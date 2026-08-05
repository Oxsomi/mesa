/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa (Intel): standalone SPIR-V -> Intel EU ISA via Mesa's brw compiler, with NO device.
 *
 * The Intel analog of the AMD/ACO path (src/amd/spirv2isa). Intel's compiler is a pure function of
 * an intel_device_info (built from a PCI id, device-free) plus NIR: brw_compiler_create(devinfo)
 * then brw_compile(). Built only when -Dspirv2isa=true. Exposed as plain C for OxC3 to link.
 *
 * This is the first slice: compute, descriptor-free modules (arithmetic / builtins / push consts).
 * Descriptor (binding-table) lowering and graphics/RT stages follow, mirroring the AMD backend.
 */
#ifndef SPIRV2ISA_INTEL_H
#define SPIRV2ISA_INTEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One representative PCI id per Intel graphics generation (ISA within a gen is essentially the
 * same; the brw compiler keys off intel_device_info which we build from the id). */
typedef enum s2i_intel_target {
   S2I_INTEL_GEN9_SKL = 0,   /* Skylake        (Gen9,   0x1912) */
   S2I_INTEL_GEN11_ICL,      /* Ice Lake       (Gen11,  0x8a52) */
   S2I_INTEL_GEN12_TGL,      /* Tiger Lake     (Xe-LP,  0x9a49) */
   S2I_INTEL_XE_HPG_DG2,     /* Arc A770 / DG2 (Xe-HPG, 0x56a0) */
   S2I_INTEL_XE_MTL,         /* Meteor Lake    (Xe-LPG, 0x7d40) */
   S2I_INTEL_XE2_LNL,        /* Lunar Lake     (Xe2,    0x64a0) */
   S2I_INTEL_TARGET_COUNT
} s2i_intel_target;

typedef enum s2i_intel_stage {
   S2I_INTEL_STAGE_VERTEX = 0,
   S2I_INTEL_STAGE_PIXEL,
   S2I_INTEL_STAGE_COMPUTE,
   S2I_INTEL_STAGE_GEOMETRY,
   S2I_INTEL_STAGE_HULL,       /* tessellation control */
   S2I_INTEL_STAGE_DOMAIN,     /* tessellation evaluation */
   S2I_INTEL_STAGE_COUNT
} s2i_intel_stage;

typedef struct s2i_intel_stats {
   uint32_t grf_used;       /* GRF (general register file) registers used */
   uint32_t program_size;   /* bytes of EU machine code */
   uint32_t scratch_size;   /* per-thread scratch bytes */
   uint32_t shared_size;    /* SLM / shared-local-memory bytes */
   uint32_t simd_width;     /* dispatch width (compute): 8 / 16 / 32 */
} s2i_intel_stats;

typedef enum s2i_intel_result {
   S2I_INTEL_OK = 0,
   S2I_INTEL_BAD_SPIRV,
   S2I_INTEL_BAD_DEVICE,     /* unknown PCI id / device info build failed */
   S2I_INTEL_NO_ENTRYPOINT,  /* spirv_to_nir found no such entry */
   S2I_INTEL_COMPILE_FAILED  /* brw_compile failed (see *message) */
} s2i_intel_result;

/*
 * Compile one entrypoint of a SPIR-V module to Intel EU ISA text (+ optional stats), for `target`.
 * Caller supplies entry + stage (never scanned from the SPIR-V).
 *   isa_text : out, malloc'd disassembly text (caller frees) on OK; may be NULL to skip.
 *   stats    : out, optional (may be NULL).
 *   message  : out, optional malloc'd diagnostic on error (caller frees if non-NULL).
 */
s2i_intel_result s2i_compile_intel(const uint32_t *spirv, size_t spirv_words, const char *entry,
                                   s2i_intel_stage stage, s2i_intel_target target, char **isa_text,
                                   s2i_intel_stats *stats, char **message);

const char *s2i_intel_target_name(s2i_intel_target target);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_INTEL_H */
