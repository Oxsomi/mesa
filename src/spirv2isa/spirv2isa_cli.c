/*
 * Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
 *
 * One CLI for every backend spirv2isa was built with. There is no vendor argument: a target names
 * its own vendor, so `list` prints them all and the compile paths take whichever number that printed.
 *
 *   spirv2isa-cli list
 *   spirv2isa-cli <target> <stage> <shader.spv> <entry> [set:binding:type ...]
 *   spirv2isa-cli <target> rtpipe[-trav] <spv:stage:entry> ...
 */
#include "spirv2isa.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define S2I_SPIRV_MAGIC 0x07230203u
#define S2I_SPIRV_MAGIC_SWAPPED 0x03022307u
#define S2I_SPIRV_HEADER_WORDS 5

/*
 * Nothing here is capped at a made up number. What a run can hold is already bounded by what it was
 * given: the library says how many targets it has, and argc bounds how many bindings or ray tracing
 * shaders the command line can name, so each is sized to exactly that. A fixed cap would silently
 * drop whatever came after it, which is the same failure as parsing an argument loosely.
 */

/*
 * Read a SPIR-V module, with the structural check the bytes pay for anyway: SPIR-V is a stream of
 * 32-bit words that opens with its magic, so a file that isn't word sized, is too short to hold a
 * header, or doesn't start with that magic is refused here by name instead of failing somewhere
 * inside a backend. This is a sanity check and nothing more; validating the module is spirv-val's
 * job, not ours.
 */
static uint32_t *
read_spirv(const char *path, size_t *out_words)
{
   FILE *f = fopen(path, "rb");
   if (!f) { perror(path); return NULL; }
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);

   if (n <= 0) {
      fprintf(stderr, "%s: empty file\n", path);
      fclose(f);
      return NULL;
   }

   if (n & 3) {
      fprintf(stderr, "%s: %ld bytes is not a whole number of 32-bit words, so it isn't SPIR-V\n", path, n);
      fclose(f);
      return NULL;
   }

   if (n < (long)(S2I_SPIRV_HEADER_WORDS * sizeof(uint32_t))) {
      fprintf(stderr, "%s: %ld bytes is shorter than a SPIR-V header\n", path, n);
      fclose(f);
      return NULL;
   }

   uint32_t *b = (uint32_t *)malloc(n);
   if (!b || fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "read fail\n"); fclose(f); free(b); return NULL; }
   fclose(f);

   if (b[0] != S2I_SPIRV_MAGIC) {

      if (b[0] == S2I_SPIRV_MAGIC_SWAPPED)
         fprintf(stderr, "%s: byte swapped SPIR-V (written by a big endian producer)\n", path);

      else fprintf(stderr, "%s: not SPIR-V (magic is 0x%08x, expected 0x%08x)\n", path, b[0], S2I_SPIRV_MAGIC);

      free(b);
      return NULL;
   }

   *out_words = (size_t)n / sizeof(uint32_t);
   return b;
}

/* Targets print by the token they are named by, which is what the compile paths take back. */
static void
print_targets(FILE *out)
{
   const size_t count = s2i_targets(NULL, 0);
   s2i_target *targets = (s2i_target *)malloc(count * sizeof(*targets));

   if (!targets) {
      fprintf(out, "   (out of memory listing targets)\n");
      return;
   }

   s2i_targets(targets, count);

   for (size_t i = 0; i < count; i++) {

      const s2i_vendor vendor = S2I_TARGET_VENDOR_OF(targets[i]);
      const int built = s2i_has_vendor(vendor);

      fprintf(out, "   %-8s %-6s %s%s\n", s2i_target_id(targets[i]), s2i_vendor_name(vendor),
              built ? s2i_target_name(targets[i]) : "",
              built ? "" : "(backend not built into this library)");
   }

   free(targets);
}

static void
print_stages(FILE *out)
{
   fprintf(out, "stages:");

   for (int i = 0; i < S2I_STAGE_COUNT; i++)
      fprintf(out, " %s", s2i_stage_id((s2i_stage)i));

   fprintf(out, "\n");
}

/*
 * A target and a stage are named, never numbered: a token that isn't one of them is refused here,
 * rather than becoming whichever target or stage happens to sit at the number atoi() produced for a
 * typo (and atoi() produces 0 for anything unparseable at all).
 */
static int
parse_target(const char *arg, s2i_target *out)
{
   const s2i_target target = s2i_target_from_id(arg);

   if (!target) {
      fprintf(stderr, "unknown target '%s'\n", arg);
      print_targets(stderr);
      return 0;
   }

   if (!s2i_has_vendor(S2I_TARGET_VENDOR_OF(target))) {
      fprintf(stderr, "target '%s' needs the %s backend, which this library was built without\n", arg,
              s2i_vendor_name(S2I_TARGET_VENDOR_OF(target)));
      return 0;
   }

   *out = target;
   return 1;
}

static int
parse_stage(const char *arg, s2i_stage *out)
{
   const int stage = s2i_stage_from_id(arg);

   if (stage < 0) {
      fprintf(stderr, "unknown stage '%s'\n", arg);
      print_stages(stderr);
      return 0;
   }

   *out = (s2i_stage)stage;
   return 1;
}

/* Bindings and the RT shader specs carry real numbers, so they are parsed strictly: the whole field
 * has to be consumed and fit, or the argument is refused. */
static int
parse_u32(const char *arg, uint32_t *out)
{
   char *end = NULL;
   unsigned long value;

   if (!arg || !*arg)
      return 0;

   errno = 0;
   value = strtoul(arg, &end, 10);

   if (errno || !end || *end || value > 0xFFFFFFFFul)
      return 0;

   *out = (uint32_t)value;
   return 1;
}

static void
print_stats(FILE *out, const s2i_stats *stats)
{
   fprintf(out, "code %u B  scratch %u B  shared %u B\n", stats->code_size, stats->scratch_size,
           stats->shared_size);

   switch (stats->vendor) {

   case S2I_VENDOR_AMD:
      fprintf(out, "SGPRs %u  VGPRs %u  spilled %u/%u  instrs %u\n", stats->amd.sgprs, stats->amd.vgprs,
              stats->amd.spilled_sgprs, stats->amd.spilled_vgprs, stats->amd.instructions);
      break;

   case S2I_VENDOR_INTEL:
      fprintf(out, "GRFs %u  SIMD %u\n", stats->intel.grf_used, stats->intel.simd_width);
      break;

   default:
      break;
   }
}

static int
run_rt_pipeline(int argc, char **argv, s2i_target target, int compile_traversal)
{
   /* Every argument after the mode names one shader, so that is exactly how many there can be. */

   const size_t capacity = (size_t)(argc > 3 ? argc - 3 : 0);
   s2i_rt_shader *shaders = (s2i_rt_shader *)calloc(capacity ? capacity : 1, sizeof(*shaders));
   uint32_t **bufs = (uint32_t **)calloc(capacity ? capacity : 1, sizeof(*bufs));
   size_t n = 0;
   int status = 1;

   if (!shaders || !bufs) {
      fprintf(stderr, "out of memory\n");
      goto cleanup;
   }

   for (int i = 3; i < argc; i++) {

      /* Split from the RIGHT: the last two fields are the stage and the entrypoint, so a path that
       * contains a colon itself (a Windows drive letter) still parses. */

      char *spec = strdup(argv[i]);
      char *entryColon = spec ? strrchr(spec, ':') : NULL;
      char *stageColon = NULL;

      if (entryColon) {
         *entryColon = '\0';
         stageColon = strrchr(spec, ':');
      }

      if (!spec || !entryColon || !stageColon || !*spec || !entryColon[1] || !stageColon[1]) {
         fprintf(stderr, "bad rt shader spec '%s' (want path:stage:entry)\n", argv[i]);
         free(spec);
         status = 2;
         goto cleanup;
      }

      *stageColon = '\0';

      s2i_stage stage;

      if (!parse_stage(stageColon + 1, &stage)) {
         free(spec);
         status = 2;
         goto cleanup;
      }

      size_t words = 0;
      bufs[n] = read_spirv(spec, &words);

      if (!bufs[n]) {
         free(spec);
         goto cleanup;
      }

      shaders[n].spirv = bufs[n];
      shaders[n].spirv_words = words;
      shaders[n].stage = stage;
      shaders[n].entry = strdup(entryColon + 1);
      n++;

      free(spec);
   }

   char *isa = NULL, *msg = NULL;
   s2i_stats stats = {0};
   const char *rt_ext_env = getenv("S2I_FEATURES");

   s2i_result r =
      s2i_compile_rt_pipeline(shaders, n, 0, compile_traversal, target, NULL, 0,
                              rt_ext_env ? (s2i_features)strtoul(rt_ext_env, NULL, 0) : 0,
                              &isa, &stats, &msg);

   fprintf(stderr, "%s %s (%zu shaders) -> result %d\n", compile_traversal ? "rtpipe-trav" : "rtpipe",
           s2i_target_name(target), n, (int)r);

   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }

   if (r == S2I_OK) {

      if (isa)
         printf("%s\n", isa);

      print_stats(stderr, &stats);
      free(isa);
   }

   status = r == S2I_OK ? 0 : 1;

cleanup:

   for (size_t i = 0; i < n; i++) {
      free(bufs[i]);
      free((void *)shaders[i].entry);
   }

   free(shaders);
   free(bufs);

   return status;
}

int
main(int argc, char **argv)
{
   if (argc >= 2 && strcmp(argv[1], "list") == 0) {
      printf("targets:\n");
      print_targets(stdout);
      return 0;
   }

   /* Whole-pipeline RT mode: `<target> rtpipe[-trav] <spv:stage:entry> ...` (first shader = raygen
    * entry; the rest are its callees: closest-hit / miss / callable / ...). `rtpipe` compiles the
    * raygen monolithically (inlining the callees); `rtpipe-trav` builds + compiles the pipeline's
    * standalone BVH-traversal shader instead (the non-monolithic / function-calls artifact). */

   if (argc >= 4 && (strcmp(argv[2], "rtpipe") == 0 || strcmp(argv[2], "rtpipe-trav") == 0)) {

      s2i_target rtTarget;

      if (!parse_target(argv[1], &rtTarget))
         return 2;

      return run_rt_pipeline(argc, argv, rtTarget, strcmp(argv[2], "rtpipe-trav") == 0);
   }

   if (argc < 5) {
      fprintf(stderr, "usage: %s <target> <stage> <shader.spv> <entry> [set:binding:type ...]\n", argv[0]);
      fprintf(stderr, "       %s <target> rtpipe[-trav] <spv:stage:entry> ...\n", argv[0]);
      fprintf(stderr, "       %s list\n", argv[0]);
      fprintf(stderr, "targets:\n");
      print_targets(stderr);
      print_stages(stderr);
      fprintf(stderr, "\nS2I_NO_CAPTURE=1 leaves stderr alone: no disassembly is returned, but a backend\n"
                      "that dies inside the compiler says why instead of exiting silently.\n");
      return 2;
   }

   s2i_target target;
   s2i_stage stage;

   if (!parse_target(argv[1], &target) || !parse_stage(argv[2], &stage))
      return 2;

   const char *entry = argv[4];

   /* Optional trailing args are descriptor bindings "set:binding:type" (type = s2i_descriptor_type),
    * so we can exercise the caller-fed layout; with none, the backend uses its generic fallback. */

   /* Every trailing argument names one binding, so that is exactly how many there can be. */

   const size_t binding_capacity = (size_t)(argc > 5 ? argc - 5 : 0);
   s2i_binding *bindings = binding_capacity ?
      (s2i_binding *)calloc(binding_capacity, sizeof(*bindings)) : NULL;
   size_t binding_count = 0;

   if (binding_capacity && !bindings) {
      fprintf(stderr, "out of memory\n");
      return 1;
   }

   for (int i = 5; i < argc; i++) {

      char setId[32], bindId[32], typeId[32];
      uint32_t set = 0, bind = 0, type = 0;

      if (sscanf(argv[i], "%31[^:]:%31[^:]:%31s", setId, bindId, typeId) != 3) {
         fprintf(stderr, "bad binding '%s' (want set:binding:type)\n", argv[i]);
         free(bindings);
         return 2;
      }

      if (!parse_u32(setId, &set) || !parse_u32(bindId, &bind) || !parse_u32(typeId, &type)) {
         fprintf(stderr, "binding '%s' has a field that isn't a number\n", argv[i]);
         free(bindings);
         return 2;
      }

      if (type >= S2I_DESC_TYPE_COUNT) {
         fprintf(stderr, "binding '%s' has no descriptor type %u (0..%d)\n", argv[i], type,
                 S2I_DESC_TYPE_COUNT - 1);
         free(bindings);
         return 2;
      }

      bindings[binding_count].set = set;
      bindings[binding_count].binding = bind;
      bindings[binding_count].type = (s2i_descriptor_type)type;
      bindings[binding_count].count = 1;
      binding_count++;
   }

   size_t words = 0;
   uint32_t *spirv = read_spirv(argv[3], &words);

   if (!spirv) {
      free(bindings);
      return 1;
   }

   char *isa = NULL, *msg = NULL;
   s2i_stats stats = {0};
   s2i_info info = {0};

   /* The CLI has no oiSH, so it leans on the defensive SPIR-V scan; S2I_FEATURES=<hex> can declare an
    * s2i_feature set to exercise the primary gate (the corpus sweep translates the oiSH into one). */

   const char *ext_env = getenv("S2I_FEATURES");
   s2i_features features_used = ext_env ? (s2i_features)strtoul(ext_env, NULL, 0) : 0;

   s2i_result r = s2i_compile(spirv, words, entry, stage, target, binding_count ? bindings : NULL,
                              binding_count, features_used, &isa, &stats, &info, &msg);

   fprintf(stderr, "target: %s -> result %d\n", s2i_target_name(target), (int)r);

   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }

   if (r == S2I_OK) {

      printf("%s\n", isa ? isa : "(no isa)");
      print_stats(stderr, &stats);

      if (info.descriptors_stateless)
         fprintf(stderr, "NOTE: buffer descriptors were lowered to raw addresses; memory messages and "
                         "register pressure are not what a driver would emit\n");

      if (info.rt_mode != S2I_RT_MODE_NA) {

         const char *mode = info.rt_mode == S2I_RT_MODE_MONOLITHIC ? "monolithic"
                          : info.rt_mode == S2I_RT_MODE_FUNCTION_CALLS ? "function-calls"
                          : info.rt_mode == S2I_RT_MODE_CPS ? "cps" : "?";

         fprintf(stderr, "RT: mode=%s  can_inline=%s (whole-pipeline inlines this shader: %s)\n",
                 mode, info.rt_can_inline ? "yes" : "no",
                 info.rt_can_inline ? "yes" : "no, forces the leaner path");
      }

      free(isa);
   }

   free(spirv);
   free(bindings);
   return r == S2I_OK ? 0 : 1;
}
