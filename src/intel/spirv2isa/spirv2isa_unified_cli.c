/*
 * Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
 *
 * Unified multi-vendor CLI: one binary that links BOTH backends (AMD/ACO via RADV, Intel/EU via brw)
 * and dispatches by a backend argument. This is the shape OxC3 would consume: pick the vendor at
 * runtime, one API surface. Proves the two device-free backends coexist in a single program.
 *
 *   spirv2isa-cli-all <amd|intel> <target> <stage> <shader.spv> <entry> [set:binding:type ...]
 */
#include "spirv2isa.h"          /* AMD backend (ACO) */
#include "spirv2isa_intel.h"    /* Intel backend (brw) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t *
read_spirv(const char *path, long *out_bytes)
{
   FILE *f = fopen(path, "rb");
   if (!f) { perror("fopen"); return NULL; }
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *spirv = (uint32_t *)malloc(n);
   if (!spirv || fread(spirv, 1, n, f) != (size_t)n) { fprintf(stderr, "read fail\n"); fclose(f); free(spirv); return NULL; }
   fclose(f);
   *out_bytes = n;
   return spirv;
}

static int
run_intel(int target, int stage, const uint32_t *spirv, long bytes, const char *entry)
{
   char *isa = NULL, *msg = NULL;
   s2i_intel_stats st = {0};
   s2i_intel_result r = s2i_compile_intel(spirv, bytes / 4, entry, (s2i_intel_stage)stage,
                                          (s2i_intel_target)target, &isa, &st, &msg);
   fprintf(stderr, "[intel] %s -> result %d\n", s2i_intel_target_name((s2i_intel_target)target), (int)r);
   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }
   if (r == S2I_INTEL_OK) {
      if (isa) printf("%s\n", isa);
      fprintf(stderr, "GRF %u  program %u B  scratch %u  shared %u  simd %u\n", st.grf_used,
              st.program_size, st.scratch_size, st.shared_size, st.simd_width);
      free(isa);
   }
   return r == S2I_INTEL_OK ? 0 : 1;
}

static int
run_amd(int target, int stage, const uint32_t *spirv, long bytes, const char *entry, char **binding_args,
        int binding_arg_count)
{
   s2i_binding bindings[64];
   size_t bc = 0;
   for (int i = 0; i < binding_arg_count && bc < 64; i++) {
      unsigned s = 0, b = 0, t = 0;
      if (sscanf(binding_args[i], "%u:%u:%u", &s, &b, &t) == 3) {
         bindings[bc].set = s;
         bindings[bc].binding = b;
         bindings[bc].type = (s2i_descriptor_type)t;
         bindings[bc].count = 1;
         bc++;
      }
   }

   char *isa = NULL, *msg = NULL;
   s2i_stats st = {0};
   s2i_info info = {0};
   s2i_result r = s2i_compile(spirv, bytes / 4, entry, (s2i_stage)stage, (s2i_target)target,
                              bc ? bindings : NULL, bc, &isa, &st, &info, &msg);
   fprintf(stderr, "[amd] %s -> result %d\n", s2i_target_name((s2i_target)target), (int)r);
   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }
   if (r == S2I_OK) {
      if (isa) printf("%s\n", isa);
      fprintf(stderr, "SGPRs %u  VGPRs %u  code %u B  instrs %u\n", st.sgprs, st.vgprs, st.code_size,
              st.instructions);
      if (info.rt_mode != S2I_RT_MODE_NA) {
         const char *mode = info.rt_mode == S2I_RT_MODE_MONOLITHIC ? "monolithic"
                          : info.rt_mode == S2I_RT_MODE_FUNCTION_CALLS ? "function-calls" : "cps";
         fprintf(stderr, "RT: mode=%s  can_inline=%s\n", mode, info.rt_can_inline ? "yes" : "no");
      }
      free(isa);
   }
   return r == S2I_OK ? 0 : 1;
}

int
main(int argc, char **argv)
{
   if (argc < 6) {
      fprintf(stderr, "usage: %s <amd|intel> <target> <stage> <shader.spv> <entry> [set:binding:type ...]\n", argv[0]);
      fprintf(stderr, "  amd:   targets 0..%d (gfx803..gfx1201); stages 0=vs 1=ps 2=cs 8=raygen 9=callable 10=miss ...\n", S2I_TARGET_COUNT - 1);
      fprintf(stderr, "  intel: targets 0..%d (SKL..LNL); stages 0=vs 1=ps 2=cs\n", S2I_INTEL_TARGET_COUNT - 1);
      return 2;
   }

   const int is_intel = strcmp(argv[1], "intel") == 0;
   if (!is_intel && strcmp(argv[1], "amd") != 0) {
      fprintf(stderr, "unknown backend '%s' (use amd|intel)\n", argv[1]);
      return 2;
   }

   int target = atoi(argv[2]);
   int stage = atoi(argv[3]);
   const char *entry = argv[5];

   long bytes = 0;
   uint32_t *spirv = read_spirv(argv[4], &bytes);
   if (!spirv)
      return 1;

   int rc = is_intel ? run_intel(target, stage, spirv, bytes, entry)
                     : run_amd(target, stage, spirv, bytes, entry, argv + 6, argc - 6);

   free(spirv);
   return rc;
}
