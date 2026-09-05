/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa: standalone SPIR-V -> ISA compiler, with NO device, across vendors.
 *
 * This is an *additive* layer on top of the (Oxsomi) Mesa fork. Each backend drives Mesa's own
 * device-independent compile path offline: AMD links RADV's compile objects (radv_compiler_info /
 * radv_postprocess_nir / radv_declare_shader_args) + ACO, Intel builds an intel_device_info from a
 * PCI id and drives brw. It is built only when the `spirv2isa` meson option is enabled, so it never
 * affects a normal Mesa build. Exposed as plain C so OxC3 (CMake, static) can link it via a conan
 * bridge, exactly like the dxc package.
 *
 * ONE API, MANY BACKENDS. Everything that means the same thing for every vendor lives here once:
 * the stage, the result, the descriptor bindings, the feature gate. Only what a vendor genuinely
 * owns carries its name: s2i_target_amd / s2i_target_intel (which architecture), s2i_stats_amd /
 * s2i_stats_intel (SGPRs and LDS vs GRFs and SLM) and s2i_rt_mode_amd (RADV's inlining taxonomy).
 * A caller picks a vendor by picking a target; s2i_compile dispatches on it.
 *
 * Duplicating the shared half per vendor is what this replaces, and it had already gone wrong: the
 * two stage enums listed the same stages in a different order, so one integer meant HULL to one
 * backend and GEOMETRY to the other. One definition cannot drift from itself.
 */
#ifndef SPIRV2ISA_H
#define SPIRV2ISA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The set of shader features a CALLER can declare, shared by every backend.
 *
 * These values are OWNED HERE. They deliberately do NOT mirror any other project's enum, and nothing
 * outside this file may assume they line up with one. A caller that has its own feature enum (OxC3 has
 * ESHExtension) TRANSLATES into these, and that translation belongs on the caller's side, next to the
 * enum it is translating from.
 *
 * That direction is the whole point. An earlier version of this gate mirrored ESHExtension bit values
 * directly: when a value was deleted there and a different feature took the freed bit, nothing failed to
 * build, and both backends silently began rejecting the new feature under the old feature's name. A
 * mirrored constant cannot notice that it went stale. Translating on the caller's side turns the same
 * change into a compile error at the translation, where whoever made it is already looking.
 *
 * Backends consume these; a feature the caller does not report simply is not gated. Values are stable
 * ABI: extend by appending, never renumber.
 */

typedef enum s2i_feature {
   S2I_FEATURE_RAY_QUERY            = 1u << 0,  /* inline ray tracing (rayQuery) in any stage */
   S2I_FEATURE_RAY_MICROMAP_OPACITY = 1u << 1,  /* opacity micromap */
   S2I_FEATURE_RAY_TRI_POSITION     = 1u << 2,  /* hit triangle vertex position fetch */
   S2I_FEATURE_RAY_REORDER          = 1u << 3,  /* shader execution reorder (SER) */
   S2I_FEATURE_BINDLESS             = 1u << 4,  /* dynamically indexed descriptor arrays */
   S2I_FEATURE_DESCRIPTOR_HEAP      = 1u << 5,  /* SM6.6 style descriptor heap */
   S2I_FEATURE_COOP_VECTOR          = 1u << 6,  /* NVIDIA cooperative vector */
   S2I_FEATURE_COOP_VECTOR_TRAINING = 1u << 7,  /* NVIDIA cooperative vector training */
   S2I_FEATURE_COOP_MATRIX          = 1u << 8,  /* KHR cooperative matrix */
   S2I_FEATURE_COOP_FP8             = 1u << 9,  /* FP8 operands for the cooperative types */
   S2I_FEATURE_MESH_TASK_TEX_DERIV  = 1u << 10, /* texture derivatives inside mesh / task */
} s2i_feature;

/* A caller-declared feature set, an OR of s2i_feature. 0 means "not declared", which leaves a backend
 * on its own thinner defences rather than gating anything. */
typedef uint32_t s2i_features;

/*
 * Reference for the caller side, which is where the translation belongs. OxC3 drops this next to
 * ESHExtension (include/formats/oiSH/sh_binaries.h) once it links this library:
 *
 *   s2i_features S2I_fromESHExtension(ESHExtension e) {
 *
 *       //Naming each enumerator is deliberate. Deleting one on the oiSH side then becomes a compile
 *       //error HERE, rather than leaving a stale bit number in the backend that quietly gates
 *       //whichever feature later inherits it.
 *
 *       static const struct { ESHExtension ext; s2i_features feature; } map[] = {
 *           { ESHExtension_RayQuery,           S2I_FEATURE_RAY_QUERY },
 *           { ESHExtension_RayMicromapOpacity, S2I_FEATURE_RAY_MICROMAP_OPACITY },
 *           { ESHExtension_RayTriPosition,     S2I_FEATURE_RAY_TRI_POSITION },
 *           { ESHExtension_RayReorder,         S2I_FEATURE_RAY_REORDER },
 *           { ESHExtension_Bindless,           S2I_FEATURE_BINDLESS },
 *           { ESHExtension_DescriptorHeap,     S2I_FEATURE_DESCRIPTOR_HEAP },
 *           { ESHExtension_CoopVec,            S2I_FEATURE_COOP_VECTOR },
 *           { ESHExtension_CoopVecTraining,    S2I_FEATURE_COOP_VECTOR_TRAINING },
 *           { ESHExtension_CoopMat,            S2I_FEATURE_COOP_MATRIX },
 *           { ESHExtension_CoopFP8,            S2I_FEATURE_COOP_FP8 },
 *           { ESHExtension_MeshTaskTexDeriv,   S2I_FEATURE_MESH_TASK_TEX_DERIV }
 *       };
 *
 *       //And ADDING one is caught here, because a new extension nobody classified must not silently
 *       //pass through a gate that has never seen it. Bump after deciding it needs no s2i_feature.
 *       _Static_assert(ESHExtension_Count == 26, "ESHExtension changed: update the s2i_feature map");
 *
 *       s2i_features f = 0;
 *
 *       for(U64 i = 0; i < sizeof(map) / sizeof(map[0]); ++i)
 *           if(e & map[i].ext)
 *               f |= map[i].feature;
 *
 *       return f;
 *   }
 *
 * Everything ESHExtension carries that is not in that table (F64, subgroup ops, Barycentrics, ...) is
 * simply not gated, which is the correct outcome: a backend only refuses what it has been shown to be
 * unable to compile.
 */

/*
 * A target packs the vendor into its high 16 bits and that vendor's architecture into its low ones,
 * so every backend numbers its own architectures from 0 and appending to one vendor never renumbers
 * another. The vendor is 1 based, which leaves 0 as no target at all rather than a valid one.
 * Values are stable ABI: extend by appending within a vendor, never renumber.
 */
typedef enum s2i_vendor {
   S2I_VENDOR_AMD = 1,
   S2I_VENDOR_INTEL,
   S2I_VENDOR_COUNT
} s2i_vendor;

#define S2I_TARGET_VENDOR_SHIFT 16
#define S2I_TARGET(vendor, index) (((int)(vendor) << S2I_TARGET_VENDOR_SHIFT) | (int)(index))
#define S2I_TARGET_VENDOR_OF(target) ((s2i_vendor)((int)(target) >> S2I_TARGET_VENDOR_SHIFT))
#define S2I_TARGET_INDEX_OF(target) ((int)(target) & 0xFFFF)

/* AMD: an amd_gfx_level + a representative radeon_family (one/two reps per architecture; ISA within
 * an architecture is essentially identical). */
typedef enum s2i_target_amd {
   S2I_TARGET_GFX8_POLARIS10 = S2I_TARGET(S2I_VENDOR_AMD, 0), /* RX 580  (GCN4)  */
   S2I_TARGET_GFX9_VEGA10    = S2I_TARGET(S2I_VENDOR_AMD, 1), /* RX Vega (GCN5)  */
   S2I_TARGET_GFX10_NAVI10   = S2I_TARGET(S2I_VENDOR_AMD, 2), /* RX 5700 XT (RDNA1) */
   S2I_TARGET_GFX10_3_NAVI21 = S2I_TARGET(S2I_VENDOR_AMD, 3), /* RDNA2 (6700 XT class) */
   S2I_TARGET_GFX11_NAVI31   = S2I_TARGET(S2I_VENDOR_AMD, 4), /* RX 7900 XTX (RDNA3) */
   S2I_TARGET_GFX12_GFX1201  = S2I_TARGET(S2I_VENDOR_AMD, 5), /* RX 9070 XT (RDNA4) */
   S2I_TARGET_AMD_COUNT      = 6
} s2i_target_amd;

/* Intel: one representative PCI id per graphics generation (ISA within a generation is essentially
 * the same; brw keys off an intel_device_info built from the id). */
typedef enum s2i_target_intel {
   S2I_TARGET_GEN9_SKL       = S2I_TARGET(S2I_VENDOR_INTEL, 0), /* Skylake        (Gen9,   0x1912) */
   S2I_TARGET_GEN11_ICL      = S2I_TARGET(S2I_VENDOR_INTEL, 1), /* Ice Lake       (Gen11,  0x8a52) */
   S2I_TARGET_GEN12_TGL      = S2I_TARGET(S2I_VENDOR_INTEL, 2), /* Tiger Lake     (Xe-LP,  0x9a49) */
   S2I_TARGET_XE_HPG_DG2     = S2I_TARGET(S2I_VENDOR_INTEL, 3), /* Arc A770 / DG2 (Xe-HPG, 0x56a0) */
   S2I_TARGET_XE_MTL         = S2I_TARGET(S2I_VENDOR_INTEL, 4), /* Meteor Lake    (Xe-LPG, 0x7d40) */
   S2I_TARGET_XE2_LNL        = S2I_TARGET(S2I_VENDOR_INTEL, 5), /* Lunar Lake     (Xe2,    0x64a0) */
   S2I_TARGET_XE2_BMG        = S2I_TARGET(S2I_VENDOR_INTEL, 6), /* Arc B580 / BMG (Xe2,    0xe20b) */
   S2I_TARGET_INTEL_COUNT    = 7
} s2i_target_intel;

/* A target of any vendor. The per-vendor enums above are the values it takes; this is the type the
 * API speaks, so a caller holding a target never has to say which vendor it came from. */
typedef int s2i_target;

/* Shader stage, supplied by the caller (OxC3 maps its ESHPipelineStage to this, we never re-derive
 * it from the SPIR-V). Covers every stage OxC3 emits, for every backend. Values are stable ABI. */
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
 * reflection). Each backend builds the descriptor set layouts its SPIR-V descriptor lowering needs
 * from these, instead of guessing a generic layout. The type drives how the descriptor is lowered (a
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

/* AMD register and memory usage. */
typedef struct s2i_stats_amd {
   uint32_t sgprs, vgprs;             /* actual usage (pre-scheduling), comparable to LLPC's used counts */
   uint32_t spilled_sgprs, spilled_vgprs;
   uint32_t code_size;                /* bytes */
   uint32_t lds_size, scratch_size;   /* bytes */
   uint32_t instructions;
} s2i_stats_amd;

/* Intel register and memory usage. */
typedef struct s2i_stats_intel {
   uint32_t grf_used;       /* GRF (general register file) registers used */
   uint32_t program_size;   /* bytes of EU machine code */
   uint32_t scratch_size;   /* per-thread scratch bytes */
   uint32_t shared_size;    /* SLM / shared-local-memory bytes */
   uint32_t simd_width;     /* dispatch width (compute): 8 / 16 / 32 */
} s2i_stats_intel;

/*
 * What a compile reports back, for whichever vendor ran it. The three facts every backend measures
 * are lifted out so a caller can print or compare them without knowing the vendor; everything a
 * vendor counts in its own units stays in its own struct, tagged by `vendor`.
 */
typedef struct s2i_stats {
   s2i_vendor vendor;             /* which member of the union below is live */
   uint32_t code_size;            /* bytes of machine code (AMD code_size / Intel program_size) */
   uint32_t scratch_size;         /* bytes of scratch */
   uint32_t shared_size;          /* bytes of on-chip shared memory (AMD LDS / Intel SLM) */
   union {
      s2i_stats_amd amd;
      s2i_stats_intel intel;
   };
} s2i_stats;

/* How a ray tracing shader was lowered by RADV. In a real pipeline RADV inlines the whole pipeline
 * into the raygen (monolithic) only if every shader is inlinable and there are < 50 of them;
 * otherwise each shader compiles standalone (the leaner FUNCTION_CALLS / CPS path) and links through
 * the SBT plus a shared traversal shader. NA for non-RT stages and for other vendors. */
typedef enum s2i_rt_mode_amd {
   S2I_RT_MODE_NA = 0,
   S2I_RT_MODE_MONOLITHIC,      /* callees inlined into raygen; no separate traversal shader */
   S2I_RT_MODE_FUNCTION_CALLS,  /* standalone/lean; pipeline links via SBT + a traversal shader */
   S2I_RT_MODE_CPS              /* continuation-passing variant of the standalone path */
} s2i_rt_mode_amd;

/* Extra facts about the compile the caller may want to surface (all optional; pass NULL to skip). */
typedef struct s2i_info {
   s2i_rt_mode_amd rt_mode;    /* how this shader was compiled (NA for non-RT and non-AMD) */
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
   S2I_COMPILE_FAILED,   /* lowering/backend failed (see *message) */
   S2I_BAD_TARGET        /* no such target, or its backend isn't built into this library */
} s2i_result;

/*
 * Compile one entrypoint of a SPIR-V module to ISA text (+ optional stats), for `target`, whose
 * vendor decides which backend runs.
 * The caller (OxC3) supplies everything it already knows from the oiSH so this layer never has to
 * scan the SPIR-V: the entrypoint name, the stage, and the descriptor binding layout. That keeps it
 * thin and non-fragile.
 *   entry         : entrypoint name (required, OxC3 always has it).
 *   stage         : the pipeline stage (OxC3 maps ESHPipelineStage -> s2i_stage).
 *   bindings      : descriptor bindings the module uses (may be NULL for none / a quick test, in
 *                   which case a permissive generic layout is synthesized as a fallback). A backend
 *                   whose descriptor lowering isn't wired yet returns S2I_UNSUPPORTED_CAP for a
 *                   module that needs them rather than compiling something that cannot bind.
 *   binding_count : number of entries in `bindings`.
 *   features_used : which features the module uses, as an OR of s2i_feature (above),
 *                   or 0 if not declared. The CALLER translates its own feature enum into these; OxC3
 *                   maps ESHExtension -> s2i_feature on its side, so that changing ESHExtension breaks
 *                   at that translation instead of silently re-pointing a bit here. This is the primary,
 *                   authoritative feature gate: a feature this target does not support (or that the
 *                   offline compiler has not wired yet) returns S2I_UNSUPPORTED_CAP with a clear
 *                   message BEFORE compiling, so it never crashes. Pass 0 to rely on the thinner
 *                   defensive SPIR-V-extension scan instead (what the CLI does).
 *   isa_text      : out, malloc'd disassembly text (caller frees) on S2I_OK.
 *   stats         : out, optional (may be NULL).
 *   info          : out, optional (may be NULL) compile facts: RT mode / inlinability / specialization.
 *   message       : out, optional diagnostic string on error, malloc'd (caller frees if non-NULL).
 */
s2i_result s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                       s2i_target target, const s2i_binding *bindings, size_t binding_count,
                       s2i_features features_used, char **isa_text, s2i_stats *stats, s2i_info *info,
                       char **message);

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
 * This is the only ray tracing entry point that is per vendor: the artifacts it produces are RADV's
 * model, and only the AMD backend implements them, so another vendor's target returns
 * S2I_UNSUPPORTED_CAP. Ray tracing itself is not AMD only: s2i_compile compiles the individual RT
 * stages on any backend that has them (Intel routes all six through brw_compile_bs), which is what
 * a caller falls back to.
 *   compile_traversal = 0: compile shaders[entry_index] (must be a raygen) MONOLITHICALLY, so its
 *     traceRay/executeCallable calls are inlined from the other shaders. That is the single baked
 *     shader the driver runs in monolithic mode.
 *   compile_traversal = 1: build and compile the pipeline's TRAVERSAL shader (the BVH walk that a
 *     non-monolithic / function-calls pipeline runs as a separate stage, called by the raygen);
 *     entry_index is ignored. Per-shader ISA for the others is s2i_compile (function-calls mode).
 * Each shader may use the same descriptor `bindings`. `features_used` is the feature set of the whole
 * PIPELINE (the union over its shaders), gated exactly as in s2i_compile.
 * isa_text/stats/message as in s2i_compile.
 */
s2i_result s2i_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count, size_t entry_index,
                                   int compile_traversal, s2i_target target, const s2i_binding *bindings,
                                   size_t binding_count, s2i_features features_used, char **isa_text,
                                   s2i_stats *stats, char **message);

/*
 * Pre-flight feature check for a cross-vendor/-target support matrix, WITHOUT re-parsing SPIR-V:
 * the caller passes the SpvCapability values it already knows the module uses (OxC3 translates its
 * ESHExtension/feature set to these), and we return a malloc'd newline-separated list of the ones
 * `target` does NOT support (or NULL if all supported). Caller frees.
 */
char *s2i_unsupported_caps(const uint32_t *caps_used, size_t caps_count, s2i_target target);

/* Human-readable target name, e.g. "gfx1100 (RDNA3, RX 7900 XTX)". Static string, "" for no such
 * target. */
const char *s2i_target_name(s2i_target target);

/*
 * The short stable token a target is named by on a command line or in a file: "gfx1100", "dg2".
 * Unlike the human name it is one word, lowercase and never reworded, so it is what a caller stores
 * and what a golden file is keyed by. Static string, "" for no such target.
 * s2i_target_from_id is its inverse, case insensitive, returning 0 (no target) for an unknown token,
 * which is the whole point: a mistyped target is refused rather than silently becoming target 0.
 */
const char *s2i_target_id(s2i_target target);
s2i_target s2i_target_from_id(const char *id);

/* The same for stages: "vs", "ps", "cs", "hs", "ds", "gs", "task", "mesh", "raygen", "callable",
 * "miss", "chit", "ahit", "isect". s2i_stage_from_id returns -1 for an unknown token. */
const char *s2i_stage_id(s2i_stage stage);
int s2i_stage_from_id(const char *id);

/* Every target this library was built with, in order, so a caller can list them without knowing
 * which backends are compiled in. Returns the count and fills `targets` when it is non-NULL and
 * `capacity` allows; pass NULL to ask for the count alone. */
size_t s2i_targets(s2i_target *targets, size_t capacity);

/* Human-readable vendor name ("AMD", "Intel"), "" for no such vendor. */
const char *s2i_vendor_name(s2i_vendor vendor);

/* Whether this library was built with `vendor`'s backend, so a caller can report a target it cannot
 * serve as unavailable rather than as unknown. */
int s2i_has_vendor(s2i_vendor vendor);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_H */
