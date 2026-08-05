/*
 * Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
 * Tiny driver for spirv2isa (Intel): spirv2isa-intel-cli <target> <stage> <shader.spv> <entry>
 * Set INTEL_DEBUG=cs (or the stage) to have brw print the EU disassembly to stderr.
 */
#include "spirv2isa_intel.h"
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char **argv)
{
   if (argc < 5) {
      fprintf(stderr, "usage: %s <target 0..%d> <stage 0..%d> <shader.spv> <entry>\n",
              argv[0], S2I_INTEL_TARGET_COUNT - 1, S2I_INTEL_STAGE_COUNT - 1);
      for (int t = 0; t < S2I_INTEL_TARGET_COUNT; t++)
         fprintf(stderr, "   target %d = %s\n", t, s2i_intel_target_name((s2i_intel_target)t));
      fprintf(stderr, "   stage: 0=vs 1=ps 2=cs 3=gs 4=hull 5=domain\n");
      return 2;
   }

   s2i_intel_target target = (s2i_intel_target)atoi(argv[1]);
   s2i_intel_stage stage = (s2i_intel_stage)atoi(argv[2]);
   const char *entry = argv[4];

   FILE *f = fopen(argv[3], "rb");
   if (!f) { perror("fopen"); return 1; }
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *spirv = (uint32_t *)malloc(n);
   if (fread(spirv, 1, n, f) != (size_t)n) { fprintf(stderr, "read fail\n"); return 1; }
   fclose(f);

   char *isa = NULL, *msg = NULL;
   s2i_intel_stats stats = {0};
   s2i_intel_result r = s2i_compile_intel(spirv, n / 4, entry, stage, target, &isa, &stats, &msg);

   fprintf(stderr, "target: %s -> result %d\n", s2i_intel_target_name(target), (int)r);
   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }
   if (r == S2I_INTEL_OK) {
      if (isa)
         printf("%s\n", isa);
      fprintf(stderr, "GRF %u  program %u B  scratch %u  shared %u  simd %u\n", stats.grf_used,
              stats.program_size, stats.scratch_size, stats.shared_size, stats.simd_width);
      free(isa);
   }
   free(spirv);
   return r == S2I_INTEL_OK ? 0 : 1;
}
