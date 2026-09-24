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
 * the stage, the result, the descriptor bindings, the pipeline state. Only what a vendor genuinely
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

#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A target packs the vendor into its high 16 bits and that vendor's architecture into its low ones,
 * so every backend numbers its own architectures from 0 and appending to one vendor never renumbers
 * another. The vendor is 1 based, which leaves 0 as no target at all rather than a valid one.
 * Values are stable ABI: extend by appending within a vendor, never renumber.
 */
typedef enum s2i_vendor {
   S2I_VENDOR_AMD = 1,
   S2I_VENDOR_INTEL,
   S2I_VENDOR_NVIDIA,
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
   S2I_TARGET_GFX10_3_NAVI21 = S2I_TARGET(S2I_VENDOR_AMD, 3), /* RX 6800/6900 (RDNA2) */
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
   S2I_TARGET_INTEL_COUNT    = 7,
} s2i_target_intel;

/* NVIDIA, via Mesa's NVK/NAK: one flagship die per generation, mirroring the other vendors' lists.
 * Maxwell is the floor as a scope choice (older hardware OxC3 does not target), not NAK's, whose
 * encoders go back to Fermi. Datacenter dies (GA100, GH100, GB100) and SoCs are out, as their device
 * shape diverges from the discrete-graphics one this models. */
typedef enum s2i_target_nvidia {
   S2I_TARGET_GM204 = S2I_TARGET(S2I_VENDOR_NVIDIA, 0), /* GTX 980     (Maxwell,   SM52)  */
   S2I_TARGET_GP102 = S2I_TARGET(S2I_VENDOR_NVIDIA, 1), /* GTX 1080 Ti (Pascal,    SM61)  */
   S2I_TARGET_TU102 = S2I_TARGET(S2I_VENDOR_NVIDIA, 2), /* RTX 2080 Ti (Turing,    SM75)  */
   S2I_TARGET_GA102 = S2I_TARGET(S2I_VENDOR_NVIDIA, 3), /* RTX 3080/90 (Ampere,    SM86)  */
   S2I_TARGET_AD102 = S2I_TARGET(S2I_VENDOR_NVIDIA, 4), /* RTX 4090    (Ada,       SM89)  */
   S2I_TARGET_GB202 = S2I_TARGET(S2I_VENDOR_NVIDIA, 5), /* RTX 5090    (Blackwell, SM120) */

   S2I_TARGET_NVIDIA_COUNT = 6,
} s2i_target_nvidia;

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
 * Descriptor set layouts, supplied by the caller exactly as Vulkan describes them: one create-info per
 * descriptor set, indexed by set number, with NULL for a set the pipeline layout leaves empty.
 *
 * These are passed to the vendor's own layout arithmetic rather than summarised first, because the
 * summary would have to drop things that change the generated ISA: per-binding VkDescriptorBindingFlags
 * (UPDATE_AFTER_BIND and PARTIALLY_BOUND decide whether a descriptor is bindless), the set's own
 * VkDescriptorSetLayoutCreateFlags (DESCRIPTOR_BUFFER_BIT_EXT selects a different descriptor model
 * entirely), mutable descriptor types, and variable descriptor counts.
 *
 * pImmutableSamplers: presence is what matters, not the handles, which may all be VK_NULL_HANDLE
 * (they are live driver objects a caller cannot create here). Every immutable sampler is treated as a
 * plain non-ycbcr sampler; a ycbcr immutable sampler changes plane counts and therefore binding
 * indices, and is not expressible offline, so a layout built around one diverges from the driver's.
 */

/* AMD register and memory usage. */
typedef struct s2i_stats_amd {
   uint32_t sgprs, vgprs;             /* actual usage (pre-scheduling), comparable to LLPC's used counts */
   uint32_t spilled_sgprs, spilled_vgprs;
   uint32_t code_size;                /* bytes */
   uint32_t lds_size, scratch_size;   /* bytes */
   uint32_t instructions;
   uint32_t wave_size;                /* lanes per wave RADV chose for this shader: 32 or 64 */
} s2i_stats_amd;

/* Intel register and memory usage, from the same genisa_stats a driver reports through
 * VK_KHR_pipeline_executable_properties, so these numbers compare directly against that path. */
typedef struct s2i_stats_intel {
   uint32_t grf_used;       /* GRF (general register file) registers used */
   uint32_t program_size;   /* bytes of EU machine code */
   uint32_t scratch_size;   /* per-thread scratch bytes */
   uint32_t shared_size;    /* SLM / shared-local-memory bytes */
   uint32_t simd_width;     /* dispatch width (compute): 8 / 16 / 32 */
   uint32_t stack_size;     /* ray tracing: per-ray stack bytes, which scratch does not cover */
   uint32_t instrs;         /* instructions in the final stream, nops and sync nops excluded */
   uint32_t cycles;         /* the scheduler's estimated latency for one thread */
   uint32_t spills, fills;  /* register spills to scratch, and their fills */
   uint32_t sends;          /* send instructions: every memory and sampler message */
   uint32_t loops;          /* loops in the final stream */
   uint32_t max_live_registers; /* peak register pressure, what a spill is measured against */
} s2i_stats_intel;

/* NVIDIA register and memory usage, straight from NAK's own per-shader report. */
typedef struct s2i_stats_nvidia {
   uint32_t gprs;             /* general-purpose registers used */
   uint32_t instrs;           /* instructions in the final stream */
   uint64_t static_cycles;    /* cycles spent in fixed-latency instructions */
   uint32_t spills_to_mem;    /* GPR spills to memory, and their fills */
   uint32_t fills_from_mem;
   uint32_t spills_to_reg;    /* spills between register files, and their fills */
   uint32_t fills_from_reg;
   uint32_t slm_size;         /* shader local (scratch) memory bytes */
   uint32_t crs_size;         /* call/return stack bytes per warp */
   uint32_t smem_size;        /* shared memory bytes (compute) */
   uint32_t max_warps_per_sm; /* occupancy bound from static register/memory use */
   uint32_t code_size;        /* bytes of SASS */
} s2i_stats_nvidia;

/*
 * What a compile reports back, for whichever vendor ran it. The three facts every backend measures
 * are lifted out so a caller can print or compare them without knowing the vendor; everything a
 * vendor counts in its own units stays in its own struct, tagged by `vendor`.
 */
typedef struct s2i_stats {
   s2i_vendor vendor;             /* which member of the union below is live */
   uint32_t code_size;            /* bytes of machine code (AMD / Intel / NVIDIA SASS) */
   uint32_t scratch_size;         /* bytes of scratch */
   uint32_t shared_size;          /* bytes of on-chip shared memory (AMD LDS / Intel SLM / NVIDIA smem) */
   union {
      s2i_stats_amd amd;
      s2i_stats_intel intel;
      s2i_stats_nvidia nvidia;
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
   uint8_t graphics_specialized; /* 1 when s2i_pipeline.graphics was applied, 0 when the stage
                                  * compiled unlinked against the driver's dynamic defaults. */
   char *notes;                /* malloc'd, caller frees; NULL when there were none. Prose the
                                * compiler emitted, chiefly why a wider SIMD variant was rejected.
                                * Read it, do not parse it. Intel only. */
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
 * What the caller declares about the pipeline the module is compiled into, as opposed to the module
 * itself. One struct for both entry points, so the two cannot drift, and so a caller names what it
 * sets: everything left zero is "not declared", which each field documents the meaning of.
 */
typedef struct s2i_pipeline {

   /* Which architecture to compile for, and so which backend runs. Required. */
   s2i_target target;

   /* Descriptor set layouts, indexed by set. NULL (or a count of 0) synthesizes a permissive
    * layout from the module's own resources, which is the fallback for a quick test. A backend
    * that builds a descriptor layout consumes these; one that places resources from the module's
    * own decorations validates against them. Either way a binding the backend cannot lower is
    * S2I_UNSUPPORTED_CAP rather than ISA that cannot bind. */
   const VkDescriptorSetLayoutCreateInfo *const *set_layouts;
   uint32_t set_layout_count;

   /* Robust access, in the structure a Vulkan pipeline declares it with. Bounds checking moves
    * every buffer and image access, so it is an input rather than an assumption; NULL is every
    * access DISABLED, since a compile with no device has no device default to mean. */
   const VkPipelineRobustnessCreateInfo *robustness;

   /* The graphics pipeline this stage belongs to, as a Vulkan pipeline declares it. It describes
    * the WHOLE pipeline (every stage, the render targets, the sample and tessellation state); the
    * `stage` argument selects which of those stages to compile, so a create info naming only that
    * one stage is refused. NULL compiles unlinked against the driver's dynamic defaults, which is
    * what a shader-object compile does. Ignored for compute and ray tracing. */
   const VkGraphicsPipelineCreateInfo *graphics;

   /* The ray tracing pipeline these shaders belong to, as a Vulkan pipeline declares it. Only the
    * groups are read: a group says which shader it recurses into, and which any-hit and
    * intersection shaders it carries, which is what decides whether an any-hit shader is inlined
    * into the traversal at all. Its shader indices (generalShader, closestHitShader, anyHitShader,
    * intersectionShader) index the `shaders` array passed alongside, in that order, so pStages and
    * that array have to describe the same shaders. NULL synthesizes one general or hit group per
    * shader, which cannot express a real hit group, so an any-hit or intersection shader is refused
    * rather than compiled against a group that drops it. Only s2i_compile_rt_pipeline reads it. */
   const VkRayTracingPipelineCreateInfoKHR *ray_tracing;

} s2i_pipeline;

/*
 * Compile one entrypoint of a SPIR-V module to ISA text (+ optional stats), for the pipeline's
 * target, whose vendor decides which backend runs.
 * The caller (OxC3) supplies everything it already knows from the oiSH so this layer never has to
 * scan the SPIR-V: the entrypoint name, the stage, and the descriptor binding layout. That keeps it
 * thin and non-fragile.
 *   entry         : entrypoint name (required, OxC3 always has it).
 *   stage         : the pipeline stage (OxC3 maps ESHPipelineStage -> s2i_stage).
 *   pipeline      : what the caller declares about the pipeline (target, descriptor layouts,
 *                   robust access). See s2i_pipeline.
 *   isa_text      : out, malloc'd disassembly text (caller frees) on S2I_OK.
 *   stats         : out, optional (may be NULL).
 *   info          : out, optional (may be NULL) compile facts: RT mode / inlinability / specialization.
 *   message       : out, optional diagnostic string on error, malloc'd (caller frees if non-NULL).
 */
s2i_result s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                       const s2i_pipeline *pipeline, char **isa_text, s2i_stats *stats,
                       s2i_info *info, char **message);

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
 * Every shader shares the one `pipeline`, as they do on a device: it is the pipeline's, not a
 * stage's.
 * isa_text/stats/message as in s2i_compile.
 */
s2i_result s2i_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count,
                                   size_t entry_index, int compile_traversal,
                                   const s2i_pipeline *pipeline, char **isa_text, s2i_stats *stats,
                                   char **message);

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
 * and what a golden file is keyed by. Flagship tokens resolve even in a build without their
 * vendor's backend; extended-tier tokens exist only when it is built, and an extended target's
 * numeric value depends on the Mesa data the build carries, so the token, never the number, is the
 * form to store. Static string, "" for no such target.
 * s2i_target_from_id is its inverse, case insensitive, returning 0 (no target) for an unknown token,
 * which is the whole point: a mistyped target is refused rather than silently becoming target 0.
 */
const char *s2i_target_id(s2i_target target);
s2i_target s2i_target_from_id(const char *id);

/* The same for stages: "vs", "ps", "cs", "hs", "ds", "gs", "task", "mesh", "raygen", "callable",
 * "miss", "chit", "ahit", "isect". s2i_stage_from_id returns -1 for an unknown token. */
const char *s2i_stage_id(s2i_stage stage);
int s2i_stage_from_id(const char *id);

/* Every target this library can serve, in vendor order with each vendor's flagships first, so a
 * caller can list them without knowing which backends are compiled in. For a built backend this
 * includes the extended tier: every other device Mesa's own tables can model; for one that is not
 * built only its flagships appear. Returns the count and fills `targets` when it is non-NULL and
 * `capacity` allows; pass NULL to ask for the count alone. */
size_t s2i_targets(s2i_target *targets, size_t capacity);

/* Whether `target` is one of the curated flagship targets above rather than the enumerated extended
 * tier; a listing shows flagships by default. 0 for extended and for unknown targets. */
int s2i_target_is_flagship(s2i_target target);

/* Human-readable vendor name ("AMD", "Intel"), "" for no such vendor. */
const char *s2i_vendor_name(s2i_vendor vendor);

/* Whether this library was built with `vendor`'s backend, so a caller can report a target it cannot
 * serve as unavailable rather than as unknown. */
int s2i_has_vendor(s2i_vendor vendor);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_H */
