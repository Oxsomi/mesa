/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa: standalone SPIR-V -> AMD ISA compiler using Mesa's ACO backend, with NO VkDevice.
 *
 * This is an *additive* layer on top of the (Oxsomi) Mesa fork: it links RADV's device-independent
 * compile objects (radv_compiler_info / radv_postprocess_nir / radv_declare_shader_args) + ACO, and
 * drives them offline. It is built only when the `spirv2isa` meson option is enabled, so it never
 * affects a normal Mesa/OxC3 build. Exposed as plain C so OxC3 (CMake, static) can link it via a
 * conan bridge, exactly like the dxc package.
 *
 * Backends: AMD/ACO first. The same shape (SPIR-V -> spirv_to_nir -> lower -> backend -> disasm)
 * extends to Intel/Mali/Adreno/NVIDIA later behind the same C API (add targets + a backend switch).
 */
#ifndef SPIRV2ISA_H
#define SPIRV2ISA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A compile target = an amd_gfx_level + a representative radeon_family (matches the drm-shim's set,
 * which is one/two reps per architecture; ISA within an arch is essentially identical). */
typedef enum s2i_target {
   S2I_TARGET_GFX8_POLARIS10 = 0, /* RX 580  (GCN4)  */
   S2I_TARGET_GFX9_VEGA10,        /* RX Vega (GCN5)  */
   S2I_TARGET_GFX10_NAVI10,       /* RX 5700 XT (RDNA1) */
   S2I_TARGET_GFX10_3_NAVI21,     /* RDNA2 (6700 XT class) */
   S2I_TARGET_GFX11_NAVI31,       /* RX 7900 XTX (RDNA3) */
   S2I_TARGET_GFX12_GFX1201,      /* RX 9070 XT (RDNA4) */
   S2I_TARGET_COUNT
} s2i_target;

/* Shader stage, supplied by the caller (OxC3 maps its ESHPipelineStage to this, we never re-derive
 * it from the SPIR-V). Covers every stage OxC3 emits. Values are stable ABI; extend by appending. */
typedef enum s2i_stage {
   S2I_STAGE_VERTEX = 0,
   S2I_STAGE_PIXEL,          /* = fragment */
   S2I_STAGE_COMPUTE,
   S2I_STAGE_HULL,           /* = tessellation control */
   S2I_STAGE_DOMAIN,         /* = tessellation evaluation */
   S2I_STAGE_GEOMETRY,
   S2I_STAGE_TASK,           /* = amplification */
   S2I_STAGE_MESH,
   S2I_STAGE_RAYGEN,
   S2I_STAGE_CALLABLE,
   S2I_STAGE_MISS,
   S2I_STAGE_CLOSEST_HIT,
   S2I_STAGE_ANY_HIT,
   S2I_STAGE_INTERSECTION,
   S2I_STAGE_COUNT
} s2i_stage;

typedef struct s2i_stats {
   uint32_t sgprs, vgprs;             /* actual usage (pre-scheduling), comparable to LLPC's used counts */
   uint32_t spilled_sgprs, spilled_vgprs;
   uint32_t code_size;                /* bytes */
   uint32_t lds_size, scratch_size;   /* bytes */
   uint32_t instructions;
} s2i_stats;

typedef enum s2i_result {
   S2I_OK = 0,
   S2I_BAD_SPIRV,        /* not a SPIR-V module / parse failure */
   S2I_UNSUPPORTED_CAP,  /* module uses a capability the target doesn't support (see *message) */
   S2I_NO_ENTRYPOINT,    /* named entry not found in the module */
   S2I_COMPILE_FAILED    /* lowering/ACO failed (see *message) */
} s2i_result;

/*
 * Compile one entrypoint of a SPIR-V module to AMD ISA text (+ optional stats), for `target`.
 * The caller (OxC3) supplies `entry` and `stage`, we never scan the SPIR-V to derive them, since
 * OxC3 already knows both from the oiSH; that keeps this layer thin and non-fragile.
 *   entry     : entrypoint name (required, OxC3 always has it).
 *   stage     : the pipeline stage (OxC3 maps ESHPipelineStage -> s2i_stage).
 *   isa_text  : out, malloc'd disassembly text (caller frees) on S2I_OK.
 *   stats     : out, optional (may be NULL).
 *   message   : out, optional diagnostic string on error, malloc'd (caller frees if non-NULL).
 * spirv_to_nir validates against the target's supported capability set, so an unsupported feature
 * yields a clean S2I_UNSUPPORTED_CAP + message instead of a crash.
 */
s2i_result s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                       s2i_target target, char **isa_text, s2i_stats *stats, char **message);

/*
 * Pre-flight feature check for a cross-vendor/-target support matrix, WITHOUT re-parsing SPIR-V:
 * the caller passes the SpvCapability values it already knows the module uses (OxC3 translates its
 * ESHExtension/feature set to these), and we return a malloc'd newline-separated list of the ones
 * `target` does NOT support (or NULL if all supported). Caller frees.
 */
char *s2i_unsupported_caps(const uint32_t *caps_used, size_t caps_count, s2i_target target);

/* Human-readable target name, e.g. "gfx1100 (RDNA3, RX 7900 XTX)". Static string. */
const char *s2i_target_name(s2i_target target);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_H */
