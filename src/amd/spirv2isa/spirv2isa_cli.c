/*
 * Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
 * Tiny driver for iterating on spirv2isa:  spirv2isa-cli <target-index> <shader.spv> [entry]
 * (mirrors the WSL harness so we can diff self-built-RADV output against the standalone path.)
 */
#include "spirv2isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t *
read_file(const char *path, long *out_bytes)
{
   FILE *f = fopen(path, "rb");
   if (!f) { perror(path); return NULL; }
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *b = (uint32_t *)malloc(n);
   if (!b || fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "read fail\n"); fclose(f); free(b); return NULL; }
   fclose(f);
   *out_bytes = n;
   return b;
}

int
main(int argc, char **argv)
{
   /* Whole-pipeline RT mode: `<target> rtpipe[-trav] <spv:stage:entry> ...` (first shader = raygen
    * entry; the rest are its callees: closest-hit / miss / callable / ...). `rtpipe` compiles the
    * raygen monolithically (inlining the callees); `rtpipe-trav` builds + compiles the pipeline's
    * standalone BVH-traversal shader instead (the non-monolithic / function-calls artifact). */
   if (argc >= 4 && (strcmp(argv[2], "rtpipe") == 0 || strcmp(argv[2], "rtpipe-trav") == 0)) {
      int compile_traversal = strcmp(argv[2], "rtpipe-trav") == 0;
      s2i_target target = (s2i_target)atoi(argv[1]);
      s2i_rt_shader shaders[16];
      uint32_t *bufs[16] = {0};
      size_t n = 0;
      for (int i = 3; i < argc && n < 16; i++) {
         char path[512], ent[128];
         unsigned st = 0;
         if (sscanf(argv[i], "%511[^:]:%u:%127s", path, &st, ent) != 3) {
            fprintf(stderr, "bad rt shader spec '%s' (want path:stage:entry)\n", argv[i]);
            return 2;
         }
         long bytes = 0;
         bufs[n] = read_file(path, &bytes);
         if (!bufs[n])
            return 1;
         shaders[n].spirv = bufs[n];
         shaders[n].spirv_words = bytes / 4;
         shaders[n].stage = (s2i_stage)st;
         shaders[n].entry = strdup(ent);
         n++;
      }
      char *isa = NULL, *msg = NULL;
      s2i_stats stats = {0};
      s2i_result r =
         s2i_compile_rt_pipeline(shaders, n, 0, compile_traversal, target, NULL, 0, &isa, &stats, &msg);
      fprintf(stderr, "%s %s (%zu shaders) -> result %d\n", compile_traversal ? "rtpipe-trav" : "rtpipe",
              s2i_target_name(target), n, (int)r);
      if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }
      if (r == S2I_OK) {
         if (isa)
            printf("%s\n", isa);
         fprintf(stderr, "SGPRs %u  VGPRs %u  code %u B  instrs %u  scratch %u  lds %u\n", stats.sgprs,
                 stats.vgprs, stats.code_size, stats.instructions, stats.scratch_size, stats.lds_size);
         free(isa);
      }
      for (size_t i = 0; i < n; i++) { free(bufs[i]); free((void *)shaders[i].entry); }
      return r == S2I_OK ? 0 : 1;
   }

   if (argc < 5) {
      fprintf(stderr, "usage: %s <target-index 0..%d> <stage-index 0..%d> <shader.spv> <entry>\n",
              argv[0], S2I_TARGET_COUNT - 1, S2I_STAGE_COUNT - 1);
      for (int t = 0; t < S2I_TARGET_COUNT; t++)
         fprintf(stderr, "   target %d = %s\n", t, s2i_target_name((s2i_target)t));
      fprintf(stderr, "   stage: 0=vs 1=ps 2=cs 3=hs 4=ds 5=gs 6=task 7=mesh 8=raygen ...\n");
      return 2;
   }

   s2i_target target = (s2i_target)atoi(argv[1]);
   s2i_stage stage = (s2i_stage)atoi(argv[2]);
   const char *entry = argv[4];

   /* Optional trailing args are descriptor bindings "set:binding:type" (type = s2i_descriptor_type),
    * so we can exercise the caller-fed layout; with none, s2i uses its generic fallback. */
   s2i_binding bindings[64];
   size_t binding_count = 0;
   for (int i = 5; i < argc && binding_count < 64; i++) {
      unsigned set = 0, bind = 0, type = 0;
      if (sscanf(argv[i], "%u:%u:%u", &set, &bind, &type) == 3) {
         bindings[binding_count].set = set;
         bindings[binding_count].binding = bind;
         bindings[binding_count].type = (s2i_descriptor_type)type;
         bindings[binding_count].count = 1;
         binding_count++;
      }
   }

   FILE *f = fopen(argv[3], "rb");
   if (!f) { perror("fopen"); return 1; }
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *spirv = (uint32_t *)malloc(n);
   if (fread(spirv, 1, n, f) != (size_t)n) { fprintf(stderr, "read fail\n"); return 1; }
   fclose(f);

   char *isa = NULL, *msg = NULL;
   s2i_stats stats = {0};
   s2i_info info = {0};
   s2i_result r = s2i_compile(spirv, n / 4, entry, stage, target, binding_count ? bindings : NULL,
                              binding_count, &isa, &stats, &info, &msg);

   fprintf(stderr, "target: %s -> result %d\n", s2i_target_name(target), (int)r);
   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }
   if (r == S2I_OK) {
      printf("%s\n", isa ? isa : "(no isa)");
      fprintf(stderr, "SGPRs %u  VGPRs %u  code %u B  instrs %u  scratch %u  lds %u\n",
              stats.sgprs, stats.vgprs, stats.code_size, stats.instructions, stats.scratch_size, stats.lds_size);
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
   return r == S2I_OK ? 0 : 1;
}
