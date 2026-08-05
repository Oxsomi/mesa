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

/*
 * Descriptor binding, supplied by the caller (OxC3 has the full binding layout from its oiSH
 * reflection). We build the RADV descriptor set layouts the SPIR-V descriptor lowering needs from
 * these, instead of guessing a generic layout. The type drives how the descriptor is lowered (a
 * buffer, an image, a sampler, an acceleration structure...); the set/binding place it so the
 * SPIR-V's DescriptorSet/Binding decorations resolve. Mirrors VkDescriptorType 1:1.
 */
typedef enum s2i_descriptor_type {
   S2I_DESC_SAMPLER = 0,
   S2I_DESC_COMBINED_IMAGE_SAMPLER,
   S2I_DESC_SAMPLED_IMAGE,
   S2I_DESC_STORAGE_IMAGE,
   S2I_DESC_UNIFORM_TEXEL_BUFFER,
   S2I_DESC_STORAGE_TEXEL_BUFFER,
   S2I_DESC_UNIFORM_BUFFER,
   S2I_DESC_STORAGE_BUFFER,
   S2I_DESC_UNIFORM_BUFFER_DYNAMIC,
   S2I_DESC_STORAGE_BUFFER_DYNAMIC,
   S2I_DESC_INPUT_ATTACHMENT,
   S2I_DESC_INLINE_UNIFORM_BLOCK,
   S2I_DESC_ACCELERATION_STRUCTURE,
   S2I_DESC_TYPE_COUNT
} s2i_descriptor_type;

typedef struct s2i_binding {
   uint32_t set;                 /* descriptor set index (0..31) */
   uint32_t binding;             /* binding number within the set */
   uint32_t count;               /* array size; 0 is treated as 1 */
   s2i_descriptor_type type;
} s2i_binding;

typedef struct s2i_stats {
   uint32_t sgprs, vgprs;             /* actual usage (pre-scheduling), comparable to LLPC's used counts */
   uint32_t spilled_sgprs, spilled_vgprs;
   uint32_t code_size;                /* bytes */
   uint32_t lds_size, scratch_size;   /* bytes */
   uint32_t instructions;
} s2i_stats;

/* How a ray tracing shader was lowered. In a real pipeline RADV inlines the whole pipeline into the
 * raygen (monolithic) only if every shader is inlinable and there are < 50 of them; otherwise each
 * shader compiles standalone (the leaner FUNCTION_CALLS / CPS path) and links through the SBT plus a
 * shared traversal shader. NA for non-RT stages. */
typedef enum s2i_rt_mode {
   S2I_RT_MODE_NA = 0,
   S2I_RT_MODE_MONOLITHIC,      /* callees inlined into raygen; no separate traversal shader */
   S2I_RT_MODE_FUNCTION_CALLS,  /* standalone/lean; pipeline links via SBT + a traversal shader */
   S2I_RT_MODE_CPS              /* continuation-passing variant of the standalone path */
} s2i_rt_mode;

/* Extra facts about the compile the caller may want to surface (all optional; pass NULL to skip). */
typedef struct s2i_info {
   s2i_rt_mode rt_mode;        /* how this shader was compiled (NA for non-RT) */
   uint8_t rt_can_inline;      /* 1 if a whole-pipeline compile could inline this shader: raygen/any-hit/
                                * intersection always, miss/closest-hit unless they recurse (traceRay),
                                * callable never. 0 otherwise / not an RT shader. */
   uint8_t graphics_specialized; /* 1 if a graphics PSO state was applied (baked-in), 0 = unlinked/dynamic */
} s2i_info;

typedef enum s2i_result {
   S2I_OK = 0,
   S2I_BAD_SPIRV,        /* not a SPIR-V module / bad arguments */
   S2I_UNSUPPORTED_CAP,  /* module uses a capability the target doesn't support (see *message) */
   S2I_NO_ENTRYPOINT,    /* named entry not found in the module */
   S2I_COMPILE_FAILED    /* lowering/ACO failed (see *message) */
} s2i_result;

/*
 * Compile one entrypoint of a SPIR-V module to AMD ISA text (+ optional stats), for `target`.
 * The caller (OxC3) supplies everything it already knows from the oiSH so this layer never has to
 * scan the SPIR-V: the entrypoint name, the stage, and the descriptor binding layout. That keeps it
 * thin and non-fragile.
 *   entry         : entrypoint name (required, OxC3 always has it).
 *   stage         : the pipeline stage (OxC3 maps ESHPipelineStage -> s2i_stage).
 *   bindings      : descriptor bindings the module uses (may be NULL for none / a quick test, in
 *                   which case a permissive generic layout is synthesized as a fallback).
 *   binding_count : number of entries in `bindings`.
 *   isa_text      : out, malloc'd disassembly text (caller frees) on S2I_OK.
 *   stats         : out, optional (may be NULL).
 *   info          : out, optional (may be NULL) compile facts: RT mode / inlinability / specialization.
 *   message       : out, optional diagnostic string on error, malloc'd (caller frees if non-NULL).
 */
s2i_result s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                       s2i_target target, const s2i_binding *bindings, size_t binding_count,
                       char **isa_text, s2i_stats *stats, s2i_info *info, char **message);

/* One shader of a ray tracing pipeline, for the whole-pipeline (monolithic) compile below. */
typedef struct s2i_rt_shader {
   const uint32_t *spirv;
   size_t spirv_words;
   const char *entry;
   s2i_stage stage; /* raygen / miss / closest_hit / any_hit / intersection / callable */
} s2i_rt_shader;

/*
 * Whole-pipeline ray tracing compile: takes all the pipeline's shaders (raygen + its callees: miss,
 * hit, callable, ...).
 *   compile_traversal = 0: compile shaders[entry_index] (must be a raygen) MONOLITHICALLY, so its
 *     traceRay/executeCallable calls are inlined from the other shaders. That is the single baked
 *     shader the driver runs in monolithic mode.
 *   compile_traversal = 1: build and compile the pipeline's TRAVERSAL shader (the BVH walk that a
 *     non-monolithic / function-calls pipeline runs as a separate stage, called by the raygen);
 *     entry_index is ignored. Per-shader ISA for the others is s2i_compile (function-calls mode).
 * Each shader may use the same descriptor `bindings`. isa_text/stats/message as in s2i_compile.
 */
s2i_result s2i_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count, size_t entry_index,
                                   int compile_traversal, s2i_target target, const s2i_binding *bindings,
                                   size_t binding_count, char **isa_text, s2i_stats *stats, char **message);

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
