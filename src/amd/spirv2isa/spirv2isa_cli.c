/*
 * Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
 * Tiny driver for iterating on spirv2isa:  spirv2isa-cli <target-index> <shader.spv> [entry]
 * (mirrors the WSL harness so we can diff self-built-RADV output against the standalone path.)
 */
#include "spirv2isa.h"
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char **argv)
{
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
   s2i_result r = s2i_compile(spirv, n / 4, entry, stage, target, &isa, &stats, &msg);

   fprintf(stderr, "target: %s -> result %d\n", s2i_target_name(target), (int)r);
   if (msg) { fprintf(stderr, "message: %s\n", msg); free(msg); }
   if (r == S2I_OK) {
      printf("%s\n", isa ? isa : "(no isa)");
      fprintf(stderr, "SGPRs %u  VGPRs %u  code %u B  instrs %u  scratch %u  lds %u\n",
              stats.sgprs, stats.vgprs, stats.code_size, stats.instructions, stats.scratch_size, stats.lds_size);
      free(isa);
   }
   free(spirv);
   return r == S2I_OK ? 0 : 1;
}
