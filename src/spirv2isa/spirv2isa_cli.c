/*
 * Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
 *
 * One CLI for every backend spirv2isa was built with. There is no vendor argument: a target names
 * its own vendor. `list` prints the flagship tier; --verbose adds every enumerated device, and the
 * compile paths take whichever token either printed.
 *
 *   spirv2isa-cli list [--verbose]
 *   spirv2isa-cli <target> <stage> <shader.spv> <entry> [set:binding:type ...]
 *   spirv2isa-cli <target> rtpipe[-trav] <spv:stage:entry> ...
 */
#include <stdbool.h>
#include "spirv2isa.h"

/* What this tool will index, not what the library supports: the set arrays below are on
 * its stack. A caller with more sets calls the library directly. */
#define S2I_CLI_MAX_SETS 32

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define S2I_SPIRV_MAGIC 0x07230203u
#define S2I_SPIRV_MAGIC_SWAPPED 0x03022307u
#define S2I_SPIRV_HEADER_WORDS 5

/*
 * Beyond the set-array cap above, which refuses rather than truncates, nothing here is capped at a
 * made up number. What a run can hold is already bounded by what it was given: the library says how
 * many targets it has, and argc bounds how many bindings or ray tracing shaders the command line
 * can name, so each is sized to exactly that. A cap that silently dropped whatever came after it
 * would be the same failure as parsing an argument loosely.
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

/* Targets print by the token they are named by, which is what the compile paths take back. The
 * default listing is the flagship tier; --verbose adds the extended tier, and the default says how
 * many targets that hides rather than hiding them silently. */
static void
print_targets(FILE *out, int verbose)
{
   const size_t count = s2i_targets(NULL, 0);
   s2i_target *targets = (s2i_target *)malloc(count * sizeof(*targets));
   size_t hidden = 0;

   if (!targets) {
      fprintf(out, "   (out of memory listing targets)\n");
      return;
   }

   s2i_targets(targets, count);

   for (size_t i = 0; i < count; i++) {

      const s2i_vendor vendor = S2I_TARGET_VENDOR_OF(targets[i]);
      const int built = s2i_has_vendor(vendor);

      if (!verbose && !s2i_target_is_flagship(targets[i])) {
         hidden++;
         continue;
      }

      fprintf(out, "   %-10s %-6s %s%s\n", s2i_target_id(targets[i]), s2i_vendor_name(vendor),
              built ? s2i_target_name(targets[i]) : "",
              built ? "" : "(backend not built into this library)");
   }

   if (hidden)
      fprintf(out, "   (+%zu more; `list --verbose` names every device the built backends can model)\n",
              hidden);

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
      print_targets(stderr, 0);
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
print_stats(FILE *out, const char *prefix, const s2i_stats *stats)
{
   fprintf(out, "%scode %u B  scratch %u B  shared %u B\n", prefix, stats->code_size,
           stats->scratch_size, stats->shared_size);

   switch (stats->vendor) {

   case S2I_VENDOR_AMD:
      fprintf(out, "%sSGPRs %u  VGPRs %u  spilled %u/%u  instrs %u  wave %u\n", prefix,
              stats->amd.sgprs, stats->amd.vgprs, stats->amd.spilled_sgprs,
              stats->amd.spilled_vgprs,
              stats->amd.instructions, stats->amd.wave_size);
      break;

   case S2I_VENDOR_INTEL:
      fprintf(out, "%sGRFs %u  SIMD %u  instrs %u  cycles %u  spills %u/%u  sends %u  loops %u  "
                   "peak regs %u\n",
              prefix, stats->intel.grf_used, stats->intel.simd_width, stats->intel.instrs,
              stats->intel.cycles, stats->intel.spills, stats->intel.fills, stats->intel.sends,
              stats->intel.loops, stats->intel.max_live_registers);
      break;

   case S2I_VENDOR_NVIDIA:
      fprintf(out, "%sGPRs %u  instrs %u  cycles %llu  spills %u/%u  warps/SM %u\n",
              prefix, stats->nvidia.gprs, stats->nvidia.instrs,
              (unsigned long long)stats->nvidia.static_cycles,
              stats->nvidia.spills_to_mem, stats->nvidia.spills_to_reg,
              stats->nvidia.max_warps_per_sm);
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

   const s2i_pipeline pipeline = { .target = target };

   s2i_result r =
      s2i_compile_rt_pipeline(shaders, n, 0, compile_traversal, &pipeline, &isa, &stats, &msg);

   fprintf(stderr, "%s %s (%zu shaders) -> result %d\n", compile_traversal ? "rtpipe-trav" : "rtpipe",
           s2i_target_name(target), n, (int)r);

   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }

   if (r == S2I_OK) {

      /* The stats lead the artifact on stdout as comment lines, so a captured output carries its
       * register/size facts the way a golden should, and names the target that produced it (by the
       * token, which is what a caller passes back) so a snapshot filed under the wrong device is
       * visible in the file and not only in its path; stderr keeps the bare human copy. */
      printf("; target %s\n", s2i_target_id(target));
      print_stats(stdout, "; ", &stats);

      if (isa)
         printf("%s\n", isa);

      print_stats(stderr, "", &stats);
      free(isa);
   }

   status = (int)r;

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

      if (argc > 3 || (argc == 3 && strcmp(argv[2], "--verbose") != 0)) {
         fprintf(stderr, "usage: %s list [--verbose]\n", argv[0]);
         return 2;
      }

      printf("targets:\n");
      print_targets(stdout, argc == 3);
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
      fprintf(stderr, "usage: %s <target> <stage> <shader.spv> <entry> [set:binding:type ...]\n"
                      "       [gfx:<key>=<value> ...]\n", argv[0]);
      fprintf(stderr, "gfx keys: stages=vs,hs,ds,gs,ps,task,mesh (whole pipeline, default vs,ps)\n"
                      "          samples=N  sample-shading=0|1  alpha-to-coverage=0|1\n"
                      "          view-mask=N  patch-points=N\n");
      fprintf(stderr, "       %s <target> rtpipe[-trav] <spv:stage:entry> ...\n", argv[0]);
      fprintf(stderr, "       %s list [--verbose]\n", argv[0]);
      fprintf(stderr, "targets:\n");
      print_targets(stderr, 0);
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

   /* Optional "gfx:<key>=<value>" args declare the graphics pipeline this stage belongs to, which
    * is what the library keys the graphics stages from. The stage list is the whole pipeline's,
    * not just the one being compiled, so leaving the vertex stage out is refused by the library. */

   bool gfx_declared = false;
   uint32_t gfx_samples = 1, gfx_view_mask = 0, gfx_patch_points = 0;
   VkBool32 gfx_alpha_to_coverage = VK_FALSE, gfx_sample_shading = VK_FALSE;
   VkShaderStageFlags gfx_stage_mask = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

   for (int i = 5; i < argc; i++) {

      if (strncmp(argv[i], "gfx:", 4))
         continue;

      char key[32], value[96];

      if (sscanf(argv[i] + 4, "%31[^=]=%95s", key, value) != 2) {
         fprintf(stderr, "bad graphics option '%s' (want gfx:<key>=<value>)\n", argv[i]);
         return 2;
      }

      gfx_declared = true;
      uint32_t number = 0;

      if (!strcmp(key, "stages")) {

         static const struct { const char *name; VkShaderStageFlagBits bit; } names[] = {
            { "vs", VK_SHADER_STAGE_VERTEX_BIT },
            { "hs", VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT },
            { "ds", VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT },
            { "gs", VK_SHADER_STAGE_GEOMETRY_BIT },
            { "ps", VK_SHADER_STAGE_FRAGMENT_BIT },
            { "task", VK_SHADER_STAGE_TASK_BIT_EXT },
            { "mesh", VK_SHADER_STAGE_MESH_BIT_EXT },
         };

         gfx_stage_mask = 0;

         for (const char *at = value; *at; ) {

            const char *end = strchr(at, ',');
            const size_t len = end ? (size_t)(end - at) : strlen(at);
            VkShaderStageFlags found = 0;

            for (size_t n = 0; n < sizeof(names) / sizeof(names[0]); n++) {
               if (strlen(names[n].name) == len && !strncmp(at, names[n].name, len))
                  found = names[n].bit;
            }

            if (!found) {
               fprintf(stderr, "gfx:stages names no stage '%.*s'\n", (int)len, at);
               return 2;
            }

            gfx_stage_mask |= found;
            at = end ? end + 1 : at + len;
         }

      } else if (!parse_u32(value, &number)) {
         fprintf(stderr, "graphics option '%s' wants a number\n", key);
         return 2;
      } else if (!strcmp(key, "samples")) {
         gfx_samples = number;
      } else if (!strcmp(key, "view-mask")) {
         gfx_view_mask = number;
      } else if (!strcmp(key, "patch-points")) {
         gfx_patch_points = number;
      } else if (!strcmp(key, "alpha-to-coverage")) {
         gfx_alpha_to_coverage = number ? VK_TRUE : VK_FALSE;
      } else if (!strcmp(key, "sample-shading")) {
         gfx_sample_shading = number ? VK_TRUE : VK_FALSE;
      } else {
         fprintf(stderr, "unknown graphics option '%s'\n", key);
         return 2;
      }
   }

   /* Optional trailing args are descriptor bindings "set:binding:type", where type is a
    * VkDescriptorType, so the CLI exercises the same layout path a caller uses. They are collected
    * into one VkDescriptorSetLayoutCreateInfo per set, which is what the library takes. */

   /* Every trailing argument names one binding, so that is exactly how many there can be. */

   const uint32_t binding_capacity = (uint32_t)(argc > 5 ? argc - 5 : 0);

   VkDescriptorSetLayoutBinding *all_bindings = binding_capacity ?
      (VkDescriptorSetLayoutBinding *)calloc(binding_capacity, sizeof(*all_bindings)) : NULL;
   uint32_t *binding_sets = binding_capacity ?
      (uint32_t *)calloc(binding_capacity, sizeof(*binding_sets)) : NULL;

   if (binding_capacity && (!all_bindings || !binding_sets)) {
      fprintf(stderr, "out of memory\n");
      free(all_bindings);
      free(binding_sets);
      return 1;
   }

   uint32_t binding_count = 0, highest_set = 0;

   for (int i = 5; i < argc; i++) {

      if (!strncmp(argv[i], "gfx:", 4))
         continue;

      char setId[32], bindId[32], typeId[32];
      uint32_t set = 0, bind = 0, type = 0;

      if (sscanf(argv[i], "%31[^:]:%31[^:]:%31s", setId, bindId, typeId) != 3) {
         fprintf(stderr, "bad binding '%s' (want set:binding:type)\n", argv[i]);
         free(all_bindings);
         free(binding_sets);
         return 2;
      }

      if (!parse_u32(setId, &set) || !parse_u32(bindId, &bind) || !parse_u32(typeId, &type)) {
         fprintf(stderr, "binding '%s' has a field that isn't a number\n", argv[i]);
         free(all_bindings);
         free(binding_sets);
         return 2;
      }

      /* The core descriptor types are 0..10; the two extension ones are the values Vulkan gives them,
       * which is why this takes a VkDescriptorType rather than an index. */
      if (type > VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT &&
          type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK &&
          type != VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
         fprintf(stderr, "binding '%s' has no VkDescriptorType %u\n", argv[i], type);
         free(all_bindings);
         free(binding_sets);
         return 2;
      }

      if (set >= S2I_CLI_MAX_SETS) {
         fprintf(stderr, "binding '%s' names set %u, and this tool goes up to %u\n", argv[i], set,
                 S2I_CLI_MAX_SETS - 1);
         free(all_bindings);
         free(binding_sets);
         return 2;
      }

      binding_sets[binding_count] = set;
      all_bindings[binding_count] = (VkDescriptorSetLayoutBinding) {
         .binding = bind,
         .descriptorType = (VkDescriptorType)type,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_ALL,
      };
      binding_count++;

      if (set > highest_set)
         highest_set = set;
   }

   /* One create-info per set, laid out contiguously so each set's bindings are adjacent. */

   VkDescriptorSetLayoutCreateInfo set_infos[S2I_CLI_MAX_SETS];
   const VkDescriptorSetLayoutCreateInfo *set_layouts[S2I_CLI_MAX_SETS];
   VkDescriptorSetLayoutBinding *sorted = binding_capacity ?
      (VkDescriptorSetLayoutBinding *)calloc(binding_capacity, sizeof(*sorted)) : NULL;

   if (binding_capacity && !sorted) {
      fprintf(stderr, "out of memory\n");
      free(all_bindings);
      free(binding_sets);
      return 1;
   }

   const uint32_t set_layout_count = binding_count ? highest_set + 1 : 0;
   uint32_t written = 0;

   for (uint32_t set = 0; set < set_layout_count; set++) {
      const uint32_t first = written;

      for (uint32_t i = 0; i < binding_count; i++) {
         if (binding_sets[i] == set)
            sorted[written++] = all_bindings[i];
      }

      set_infos[set] = (VkDescriptorSetLayoutCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = written - first,
         .pBindings = written > first ? &sorted[first] : NULL,
      };
      set_layouts[set] = &set_infos[set];
   }

   size_t words = 0;
   uint32_t *spirv = read_spirv(argv[3], &words);

   if (!spirv) {
      free(all_bindings);
      free(binding_sets);
      free(sorted);
      return 1;
   }

   char *isa = NULL, *msg = NULL;
   s2i_stats stats = {0};
   s2i_info info = {0};



   VkPipelineShaderStageCreateInfo gfx_stage_infos[7];
   uint32_t gfx_stage_count = 0;

   static const VkShaderStageFlagBits gfx_stage_bits[] = {
      VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
      VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_TASK_BIT_EXT, VK_SHADER_STAGE_MESH_BIT_EXT,
   };

   for (size_t i = 0; i < sizeof(gfx_stage_bits) / sizeof(gfx_stage_bits[0]); i++) {

      if (!(gfx_stage_mask & gfx_stage_bits[i]))
         continue;

      gfx_stage_infos[gfx_stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      gfx_stage_infos[gfx_stage_count].pNext = NULL;
      gfx_stage_infos[gfx_stage_count].flags = 0;
      gfx_stage_infos[gfx_stage_count].stage = gfx_stage_bits[i];
      gfx_stage_infos[gfx_stage_count].module = VK_NULL_HANDLE;
      gfx_stage_infos[gfx_stage_count].pName = entry;
      gfx_stage_infos[gfx_stage_count].pSpecializationInfo = NULL;
      gfx_stage_count++;
   }

   const VkFormat gfx_color_format = VK_FORMAT_R8G8B8A8_UNORM;

   const VkPipelineRenderingCreateInfo gfx_rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .viewMask = gfx_view_mask,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &gfx_color_format,
   };

   const VkPipelineMultisampleStateCreateInfo gfx_ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)(gfx_samples ? gfx_samples : 1),
      .sampleShadingEnable = gfx_sample_shading,
      .minSampleShading = gfx_sample_shading ? 1.0f : 0.0f,
      .alphaToCoverageEnable = gfx_alpha_to_coverage,
   };

   const VkPipelineTessellationStateCreateInfo gfx_ts = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      .patchControlPoints = gfx_patch_points,
   };

   /* The fill reads these even when nothing about them is declared (rasterizer discard decides
    * whether the pipeline has rasterization at all), so a create info without them is dereferenced
    * through a null pointer inside the runtime. */
   const VkPipelineVertexInputStateCreateInfo gfx_vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };

   const VkPipelineInputAssemblyStateCreateInfo gfx_ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };

   const VkPipelineViewportStateCreateInfo gfx_vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };

   const VkPipelineRasterizationStateCreateInfo gfx_rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };

   const VkPipelineDepthStencilStateCreateInfo gfx_ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
   };

   const VkPipelineColorBlendAttachmentState gfx_cb_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };

   const VkPipelineColorBlendStateCreateInfo gfx_cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &gfx_cb_attachment,
   };

   const VkGraphicsPipelineCreateInfo gfx_ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &gfx_rendering,
      .stageCount = gfx_stage_count,
      .pStages = gfx_stage_infos,
      .pVertexInputState = &gfx_vi,
      .pInputAssemblyState = &gfx_ia,
      .pViewportState = &gfx_vp,
      .pRasterizationState = &gfx_rs,
      .pMultisampleState = &gfx_ms,
      .pDepthStencilState = &gfx_ds,
      .pColorBlendState = &gfx_cb,
      .pTessellationState = gfx_patch_points ? &gfx_ts : NULL,
   };

   const s2i_pipeline pipeline = {
      .target = target,
      .set_layouts = set_layout_count ? set_layouts : NULL,
      .set_layout_count = set_layout_count,
      .graphics = gfx_declared ? &gfx_ci : NULL,
   };

   s2i_result r = s2i_compile(spirv, words, entry, stage, &pipeline, &isa, &stats, &info, &msg);

   fprintf(stderr, "target: %s -> result %d\n", s2i_target_name(target), (int)r);

   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }

   if (info.notes) {
      fprintf(stderr, "notes:\n%s", info.notes);
      free(info.notes);
      info.notes = NULL;
   }

   if (r == S2I_OK) {

      printf("; target %s\n", s2i_target_id(target));
      print_stats(stdout, "; ", &stats);
      printf("%s\n", isa ? isa : "(no isa)");
      print_stats(stderr, "", &stats);

      if (info.graphics_specialized)
         fprintf(stderr, "graphics: pipeline state applied\n");

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
   free(all_bindings);
      free(binding_sets);
      free(sorted);
   /* The exit code is the s2i_result itself, so a spawning caller can tell the refusal kinds apart. */
   return (int)r;
}
