/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa (Intel): standalone SPIR-V -> Intel EU ISA via Mesa's brw compiler, no device.
 * See spirv2isa_intel.h. The brw compiler is a pure function of intel_device_info + NIR, so unlike
 * RADV we do not need to fold into the driver: we link libintel_compiler (brw) and drive it directly
 * (blorp shows the shape). intel_get_device_info_from_pci_id gives a device-free devinfo.
 */

#include "spirv2isa_intel.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>                       /* dup/dup2/close for the ISA-capture stderr redirect */

#include "util/ralloc.h"
#include "util/macros.h"
#include "util/bitset.h"                  /* BITSET_TEST/SET/CLEAR on the intel_debug global */
#include "dev/intel_device_info.h"        /* intel_get_device_info_from_pci_id */
#include "dev/intel_debug.h"              /* process_intel_debug_variable (inits INTEL_SIMD/DEBUG globals) */
#include "compiler/shader_enums.h"        /* mesa_shader_stage / MESA_SHADER_* */
#include "compiler/glsl_types.h"          /* glsl_type_singleton_init_or_ref/decref (no device inits it) */
#include "compiler/nir/nir.h"             /* nir_shader, address formats, gather_info */
#include "compiler/nir/nir_builder.h"     /* nir builder for the buffer-descriptor lowering */
#include "compiler/spirv/nir_spirv.h"     /* spirv_to_nir + options */
#include "compiler/spirv/spirv_info.h"    /* struct spirv_capabilities */
#include "brw_compiler.h"                  /* brw_compiler_create, brw_compile, prog_data/key */
#include "brw_reg.h"                       /* REG_SIZE, the push-constant granularity */
#include "brw_nir.h"                       /* brw_preprocess_nir, brw_nir_lower_cs_intrinsics, brw_nir_lower_cmat */
#include "brw_nir_rt.h"                    /* brw_nir_lower_ray_queries (inline ray tracing) */
#include "anv_nir.h"                       /* ANV's descriptor, driver-value and multiview passes */
#include "spirv2isa_stage.h"
#include "spirv2isa_intel_anv.h"           /* the inputs those passes read, built without a device */

struct s2i_intel_target_desc {
   int pci_id;
   const char *name;
};

static const struct s2i_intel_target_desc s2i_intel_targets[S2I_TARGET_INTEL_COUNT] = {
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GEN9_SKL)]   = { 0x1912, "Gen9 Skylake (SKL GT2)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GEN11_ICL)]  = { 0x8a52, "Gen11 Ice Lake (ICL GT2)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GEN12_TGL)]  = { 0x9a49, "Xe-LP Tiger Lake (TGL GT2)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE_HPG_DG2)] = { 0x56a0, "Xe-HPG DG2 (Arc A770)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE_MTL)]     = { 0x7d40, "Xe-LPG Meteor Lake (MTL)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE2_LNL)]    = { 0x64a0, "Xe2 Lunar Lake (LNL)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE2_BMG)]    = { 0xe20b, "Xe2 Battlemage (BMG G21, Arc B580)" },
};


/* brw calls compiler->shader_{debug,perf}_log during codegen with no NULL check; the driver normally
 * installs these. Without them a plain compile (no INTEL_DEBUG) calls a NULL pointer and crashes. */
static void
s2i_intel_log_noop(void *data, unsigned *id, const char *fmt, ...)
{
   (void)data;
   (void)id;
   (void)fmt;
}

const char *
s2i_intel_target_name(int target_index)
{
   if (target_index < 0 || target_index >= S2I_TARGET_INTEL_COUNT)
      return "";
   return s2i_intel_targets[target_index].name;
}

/* Defensive scan of the module's declared SPV_* extensions against a deny-list of features brw cannot
 * lower device-free and that would otherwise CRASH (in a nir pass or brw_from_nir) instead of failing
 * cleanly. Walks only the mode-setting prefix (extensions precede the first OpFunction), reading the
 * null-terminated string operand; returns the matched name (static string from `deny`) or NULL. */
static const char *
s2i_intel_first_unsupported_ext(const uint32_t *spirv, size_t words, const char *const *deny,
                                size_t deny_count)
{
   for (size_t i = 5; i < words;) {
      uint32_t op = spirv[i] & 0xFFFFu;
      uint32_t len = spirv[i] >> 16;
      if (len == 0 || i + len > words)
         break;
      if (op == 54 /* OpFunction */)
         break;
      if (op == 10 /* OpExtension */ && len > 1) {
         /* Compare within this instruction's own words so an unterminated literal in a malformed
          * module cannot read past the end of the module. */
         const char *ext = (const char *)&spirv[i + 1];
         const size_t avail = (size_t)(len - 1) * 4;
         if (memchr(ext, '\0', avail)) {
            for (size_t d = 0; d < deny_count; d++)
               if (strcmp(ext, deny[d]) == 0)
                  return deny[d];
         }
      }
      i += len;
   }
   return NULL;
}

/* Features brw cannot lower device-free and that crash rather than error. NV cooperative vector is an
 * NVIDIA vendor extension with no Mesa lowering. (KHR cooperative matrix is now wired via
 * brw_nir_lower_cmat; ray query is wired via brw_nir_lower_ray_queries, so neither is listed here.) */
static const char *const s2i_intel_unsupported_exts[] = {
   "SPV_NV_cooperative_vector",
};

/* Primary feature gate: the caller declares which features the module uses as an s2i_features mask
 * (spirv2isa.h, values owned by us), so we gate without parsing SPIR-V and without knowing the
 * caller's own feature enum. See the AMD side and the shared header for why the translation lives on
 * the caller. */
enum s2i_intel_support { S2I_INTEL_SUP_OK, S2I_INTEL_SUP_NOT_WIRED, S2I_INTEL_SUP_UNSUPPORTED };

/* Verdict for one feature on brw. Anything not named here is supported, which is now nearly everything:
 * all 14 stages (the ray tracing pipeline stages included), buffers, images, textures, bindless /
 * dynamic descriptor arrays, descriptor heap, multiview, ray query, ray triangle position fetch and
 * cooperative matrix. Only what vtn itself rejects is listed. Every entry was checked against a real
 * corpus shader with the gate bypassed; do not gate a feature that has not been observed to fail, and
 * re-check these whenever a stage or lowering is added (wiring the RT stages made two of them stale).
 * MESH_TASK_TEX_DERIV is deliberately absent but UNTESTED: OxC3 emits no SPIR-V for it today. */
static enum s2i_intel_support
s2i_intel_ext_verdict(s2i_feature bit, const char **name)
{
   switch (bit) {
   case S2I_FEATURE_COOP_VECTOR:          *name = "cooperative vector (NVIDIA)";          return S2I_INTEL_SUP_UNSUPPORTED;
   case S2I_FEATURE_COOP_VECTOR_TRAINING: *name = "cooperative vector training (NVIDIA)"; return S2I_INTEL_SUP_UNSUPPORTED;
   case S2I_FEATURE_RAY_REORDER:          *name = "shader execution reorder (SER)";       return S2I_INTEL_SUP_UNSUPPORTED;
   case S2I_FEATURE_RAY_MICROMAP_OPACITY: *name = "ray opacity micromap";                 return S2I_INTEL_SUP_UNSUPPORTED;
   default:                               return S2I_INTEL_SUP_OK;
   }
}

/* Walk the caller-declared feature set; first unsupported/not-wired feature -> clean message. */
static bool
s2i_intel_gate_extensions(s2i_features features, const char *target_name, char **message)
{
   for (uint32_t b = features; b; b &= b - 1) {
      uint32_t bit = b & (uint32_t)(-(int32_t)b);
      const char *name = NULL;
      enum s2i_intel_support s = s2i_intel_ext_verdict((s2i_feature)bit, &name);
      if (s != S2I_INTEL_SUP_OK) {
         if (message) {
            char buf[160];
            snprintf(buf, sizeof(buf), "%s is %s on %s", name,
                     s == S2I_INTEL_SUP_NOT_WIRED ? "not yet supported by this offline compiler"
                                                  : "not supported by this backend",
                     target_name);
            *message = strdup(buf);
         }
         return true;
      }
   }
   return false;
}






/* Folds the address arithmetic ANV's descriptor passes leave behind, so anv_nir_lower_ubo_loads can
 * still see a constant offset. Skipping it also strands MAX_SETS-sized temp arrays in scratch. */
static void
s2i_intel_cleanup_nir(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);
}

/* brw uses a render target's index directly as a binding table index, so those entries are reserved
 * before descriptors claim any (ANV: anv_shader_compute_fragment_rts). Only the count matters. */
static void
s2i_intel_reserve_fragment_rts(nir_shader *nir, struct anv_pipeline_bind_map *map)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return;

   const uint64_t rt_mask = nir->info.outputs_written &
                            BITFIELD64_RANGE(FRAG_RESULT_DATA0, MAX_DRAW_BUFFERS);

   map->surface_count = util_bitcount64(rt_mask);
}

/* Every intrinsic that must be gone before brw sees the shader. brw_from_nir has no default case, so
 * one left here aborts with no explanation; naming it makes that an ordinary refusal instead. */
static bool
s2i_intel_is_unlowered_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_constant:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_heap_load:
   case nir_intrinsic_image_heap_store:
   case nir_intrinsic_image_heap_atomic:
   case nir_intrinsic_image_heap_atomic_swap:
   case nir_intrinsic_image_heap_size:
   case nir_intrinsic_image_heap_samples:
   case nir_intrinsic_image_heap_sparse_load:
      return true;
   default:
      return false;
   }
}

/* Name of the first resource intrinsic nothing lowered, or NULL when the shader is ready for brw. */
static const char *
s2i_intel_first_unlowered(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;

            if (s2i_intel_is_unlowered_intrinsic(op))
               return nir_intrinsic_infos[op].name;
         }
      }
   }

   return NULL;
}

s2i_result
s2i_intel_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                  int target_index, const s2i_binding *bindings, size_t binding_count,
                  s2i_features features_used, char **isa_text, s2i_stats_intel *stats,
                  s2i_info *info, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;

   if (!spirv || spirv_words < 5 || spirv[0] != 0x07230203u)
      return S2I_BAD_SPIRV;
   if (!entry || target_index < 0 || target_index >= S2I_TARGET_INTEL_COUNT || stage < 0 ||
       stage >= S2I_STAGE_COUNT)
      return S2I_BAD_SPIRV;

   /* Primary gate: the caller-declared feature set (OxC3 translates its oiSH ESHExtension into it). */
   if (features_used && s2i_intel_gate_extensions(features_used, s2i_intel_target_name(target_index), message))
      return S2I_UNSUPPORTED_CAP;

   /* Thin defensive fallback for callers that declare no features (features_used == 0, e.g. the CLI). */
   const char *bad_ext =
      s2i_intel_first_unsupported_ext(spirv, spirv_words, s2i_intel_unsupported_exts,
                                      ARRAY_SIZE(s2i_intel_unsupported_exts));
   if (bad_ext) {
      if (message) {
         char buf[128];
         snprintf(buf, sizeof(buf), "unsupported SPIR-V extension for Intel/brw: %s", bad_ext);
         *message = strdup(buf);
      }
      return S2I_UNSUPPORTED_CAP;
   }

   /* All of compute + vertex + fragment + geometry + tessellation control/eval are wired. The graphics
    * stages run their own input lowering inside brw_compile_*; TES computes its input VUE map itself
    * when we leave it NULL. (Mesh/task are not in the s2i_stage enum yet.) */

   /* Initialize the INTEL_DEBUG / INTEL_SIMD globals a driver would set at startup; without this the
    * SIMD-width selection reads every width as disabled and nothing compiles. call_once-guarded. */
   process_intel_debug_variable();

   void *mem_ctx = ralloc_context(NULL);

   /* No device did this for us; the glsl type system uses a singleton linear allocator. */
   glsl_type_singleton_init_or_ref();

   /* 1. device-free devinfo from a PCI id, then the brw compiler from just that. */
   struct intel_device_info devinfo;
   if (!intel_get_device_info_from_pci_id(s2i_intel_targets[target_index].pci_id, &devinfo)) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      if (message)
         *message = strdup("intel_get_device_info_from_pci_id failed");
      return S2I_BAD_TARGET;
   }
   /* mem_alignment is normally filled by the i915/xe kernel driver from a DRM query; the device-free
    * PCI-id path leaves it 0, which makes brw's mem-access vectorizer compute align(end, MIN2(.., 0))
    * and hit a util_is_power_of_two assert (only when adjacent accesses actually vectorize, e.g. a ray
    * query's spilled struct).
    * This is not a free constant: brw_nir.c's vectorizer refuses to widen a global load whose result
    * would cross it, so it decides how many buffer loads merge, and therefore the instruction count.
    * Derive it the way the kernel does (i915/intel_device_info.c), which is device-free: 64 KB from
    * Xe-HPG on and for anything with local memory, 4 KB before that. */
   if (devinfo.mem_alignment == 0)
      devinfo.mem_alignment = (devinfo.verx10 >= 125 || devinfo.has_local_mem) ? 64 * 1024 : 4096;
   struct brw_compiler *compiler = brw_compiler_create(mem_ctx, &devinfo);

   /* Must exist before spirv_to_nir: the address formats below are derived from it. */
   struct anv_physical_device *pdev = rzalloc(mem_ctx, struct anv_physical_device);
   s2i_intel_anv_init_physical_device(pdev, &devinfo, compiler);

   /* Caller-owned pipeline state, and it selects the UBO/SSBO address formats, so it moves the memory
    * instructions emitted. Nothing supplies it yet; it becomes a declared input, never a derived one. */
   const enum brw_robustness_flags robust_flags = 0;
   compiler->shader_debug_log = s2i_intel_log_noop;
   compiler->shader_perf_log = s2i_intel_log_noop;

   const mesa_shader_stage ms = s2i_stage_to_mesa(stage);

   /* 2. SPIR-V -> NIR. Broad capability set so vtn never rejects a feature the module declares; a
    * feature the HW cannot do still fails later in brw. No debug callback (device would own it). */
   struct spirv_capabilities caps;
   memset(&caps, 0xff, sizeof(caps));

   struct spirv_to_nir_options spirv_opts;
   memset(&spirv_opts, 0, sizeof(spirv_opts));
   spirv_opts.capabilities = &caps;
   /* vtn sizes vulkan_resource_index from these, and ANV's lowering builds a 4x32 resource index, so
    * anything else and the two disagree about what a descriptor is. */
   spirv_opts.ubo_addr_format = anv_nir_ubo_addr_format(pdev, robust_flags);
   spirv_opts.ssbo_addr_format = anv_nir_ssbo_addr_format(pdev, robust_flags);
   spirv_opts.phys_ssbo_addr_format = nir_address_format_64bit_global;
   spirv_opts.push_const_addr_format = nir_address_format_logical;
   spirv_opts.shared_addr_format = nir_address_format_32bit_offset;

   const nir_shader_compiler_options *nir_options = &compiler->nir_options[ms];
   nir_shader *nir =
      spirv_to_nir(spirv, spirv_words, NULL, ms, entry, &spirv_opts, nir_options);
   if (!nir) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      if (message)
         *message = strdup("spirv_to_nir failed (bad entrypoint or unsupported module)");
      return S2I_NO_ENTRYPOINT;
   }
   ralloc_steal(mem_ctx, nir);
   nir->info.stage = ms;

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Cooperative matrix (KHR): brw lowers cmat ops to subgroup/DPAS ops. This must run BEFORE the
    * descriptor passes, since it emits the matrix load/store memory accesses they then lower, and it
    * needs a valid api subgroup size (ANV derives it device-side; offline we take the shader's declared
    * size or default to a DPAS-friendly 16). cmat lowering emits cmat_call -> nir_call, so inline. */
   if (ms == MESA_SHADER_COMPUTE && nir->info.cs.has_cooperative_matrix) {
      if (nir->info.api_subgroup_size == 0)
         nir->info.api_subgroup_size = 16;
      /* brw_required_dispatch_width (brw_simd_selection.cpp) derives the REQUIRED width from
       * min/max_subgroup_size and treats "both 0" as no requirement, so brw would be free to compile
       * SIMD32 for code cmat just lowered for api_subgroup_size lanes. Pin them the way ANV's
       * anv_fixup_subgroup_size does, so the compiled width matches what the lowering assumed. */
      nir->info.min_subgroup_size = nir->info.api_subgroup_size;
      nir->info.max_subgroup_size = nir->info.api_subgroup_size;
      NIR_PASS(_, nir, brw_nir_lower_cmat, nir->info.api_subgroup_size);
      NIR_PASS(_, nir, nir_opt_dce);
      bool cmat_inlined = false;
      NIR_PASS(cmat_inlined, nir, nir_inline_functions);
      nir_remove_non_entrypoints(nir);
      if (cmat_inlined) {
         bool cmat_lowered_globals = false;
         NIR_PASS(cmat_lowered_globals, nir, nir_lower_global_vars_to_local);
         if (cmat_lowered_globals)
            NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);
         NIR_PASS(_, nir, nir_opt_copy_prop_vars);
         NIR_PASS(_, nir, nir_opt_copy_prop);
      }
      NIR_PASS(_, nir, nir_opt_deref);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_function_temp, 16);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_dce);

   /* The format side of storage images, before ANV's lowering rewrites the derefs. Run it
    * UNCONDITIONALLY: gather_info's num_images / num_textures / uses_bindless all skip a variable with
    * an interface_type (nir_gather_info.c), so a descriptor-array image shader reports 0/0/0. */
   NIR_PASS(_, nir, brw_nir_lower_storage_image, compiler,
            &(struct brw_nir_lower_storage_image_opts){
               .lower_loads = true,
               .lower_stores_64bit = true,
            });

   /* 3. brw preprocessing (common); the stage-specific input lowering runs inside brw_compile_* for
    * graphics, while compute needs brw_nir_lower_cs_intrinsics done here first (brw asserts it). */
   struct brw_nir_compiler_opts opts;
   memset(&opts, 0, sizeof(opts));
   brw_preprocess_nir(compiler, nir, &opts);

   /* The view mask is caller-owned render-pass state; 0 is the single-view shader. Gated as ANV gates
    * it, because the pass asserts on compute. */
   if (ms <= MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, anv_nir_lower_multiview, 0 /* view_mask */, false /* primitive replication */);

   /* Ray query (RayQueryKHR, e.g. inline ray tracing in a compute shader): brw lowers the opaque rq_*
    * ops to its internal query representation; device-free it needs only devinfo (ANV passes exactly
    * &pdevice->info). A no-op when the shader has no ray queries. The acceleration-structure handle
    * it consumes is a descriptor, lowered with the others below. */
   NIR_PASS(_, nir, brw_nir_lower_ray_queries, &devinfo);
   /* SSA-ify the function_temp query struct it introduced, so brw sees resolved sources. */
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_dce);

   /* From here the descriptor rewriting is ANV's, in ANV's order (anv_shader_compile.c). We supply
    * the layout it reads, nothing more. */

   s2i_intel_anv_layouts layouts;

   if (!s2i_intel_anv_build_layouts(pdev, bindings, binding_count, &layouts, message)) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_UNSUPPORTED_CAP;
   }

   struct anv_pipeline_bind_map bind_map;
   memset(&bind_map, 0, sizeof(bind_map));

   /* Zero is UNKNOWN, which is neither DIRECT nor INDIRECT; almost every site tests for one of those
    * two, so leaving it builds handles with no set base and mixed strides, and never asserts. */
   bind_map.layout_type = layouts.set_count > 0 ? layouts.sets[0]->type
                                                : ANV_PIPELINE_DESCRIPTOR_SET_LAYOUT_TYPE_DIRECT;

   bind_map.surface_to_descriptor = rzalloc_array(mem_ctx, struct anv_pipeline_binding, 256);
   bind_map.sampler_to_descriptor = rzalloc_array(mem_ctx, struct anv_pipeline_binding, 256);

   s2i_intel_reserve_fragment_rts(nir, &bind_map);

   struct anv_pipeline_push_map push_map;
   memset(&push_map, 0, sizeof(push_map));

   NIR_PASS(_, nir, anv_nir_apply_pipeline_layout, pdev, robust_flags, layouts.sets,
            layouts.set_count, layouts.dynamic_offset_start, false /* device_bindable */,
            &bind_map, &push_map, mem_ctx);

   /* The passes read the layouts, they do not keep them. */
   s2i_intel_anv_free_layouts(&layouts);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo, anv_nir_ubo_addr_format(pdev, robust_flags));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo, anv_nir_ssbo_addr_format(pdev, robust_flags));

   s2i_intel_cleanup_nir(nir);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   s2i_intel_cleanup_nir(nir);

   /* anv_nir_lower_ubo_loads READS the divergence flag rather than computing it, so without the
    * analysis every UBO load looks uniform and gets a uniform-block load it must not have. */
   NIR_PASS(_, nir, nir_convert_to_lcssa, true, true);
   nir_divergence_analysis(nir);
   NIR_PASS(_, nir, anv_nir_lower_ubo_loads);
   NIR_PASS(_, nir, nir_opt_remove_phis);

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Per-stage prog_data + key. brw_compile dispatches by nir->info.stage, so we use brw's own unions
    * (brw_compiler.h) which are sized for every stage it can dispatch to. */
   union brw_any_prog_data prog_data;
   memset(&prog_data, 0, sizeof(prog_data));

   union brw_any_prog_key prog_key;
   memset(&prog_key, 0, sizeof(prog_key));

   struct brw_stage_prog_data *base_prog_data;
   struct brw_base_prog_key *base_prog_key;

   if (ms == MESA_SHADER_COMPUTE) {
      /* The driver values this introduces are lowered below, and anv_nir_compute_push_layout owns
       * push_sizes and the brw_cs_fill_push_const_info call. */
      brw_nir_lower_cs_intrinsics(nir, &devinfo, &prog_data.cs);

      base_prog_data = &prog_data.cs.base;
      base_prog_key = &prog_key.cs.base;
   } else if (ms == MESA_SHADER_FRAGMENT) {
      base_prog_data = &prog_data.fs.base;
      base_prog_key = &prog_key.fs.base;
   } else if (s2i_stage_is_rt(ms)) {
      /* All six RT stages compile through brw_compile_bs; brw_bs_prog_data nests the stage base
       * directly (like fragment, unlike the VUE stages). */
      base_prog_data = &prog_data.bs.base;
      base_prog_key = &prog_key.bs.base;
   } else if (ms == MESA_SHADER_TASK || ms == MESA_SHADER_MESH) {
      /* Task/mesh prog_data nest brw_cs_prog_data (they dispatch like compute), so the stage base is
       * two levels down. Unlike compute, brw_compile_task/mesh run brw_nir_lower_cs_intrinsics
       * themselves, so we must not do it here. A mesh shader compiled on its own gets tue_map = NULL,
       * which brw explicitly supports (task and mesh need not be compiled together). */
      base_prog_data = ms == MESA_SHADER_TASK ? &prog_data.task.base.base : &prog_data.mesh.base.base;
      base_prog_key = ms == MESA_SHADER_TASK ? &prog_key.task.base : &prog_key.mesh.base;
   } else {
      /* VUE stages (vertex / geometry / tess-ctrl / tess-eval): the prog_data nests
       * brw_vue_prog_data -> brw_stage_prog_data; the union members overlap at offset 0, so the vs
       * view addresses the shared base. brw_compile_* casts to the real per-stage prog_data/key. */
      base_prog_data = &prog_data.vs.base.base;
      base_prog_key = &prog_key.vs.base;
   }

   /* ANV's passes read the stage off prog_data and run before brw_compile would set it. */
   base_prog_data->stage = ms;

   /* None of these is optional: apply_pipeline_layout emits resource_intel that only
    * lower_resource_intel removes, and lower_driver_values emits push offsets that only
    * compute_push_layout turns into register numbers. */
   NIR_PASS(_, nir, anv_nir_lower_driver_values, pdev);
   NIR_PASS(_, nir, anv_nir_update_resource_intel_block);
   NIR_PASS(_, nir, anv_nir_shrink_push_constant_ranges);
   NIR_PASS(_, nir, anv_nir_compute_push_layout, pdev, robust_flags,
            &(struct anv_nir_push_layout_info) { 0 }, base_prog_key, base_prog_data,
            &bind_map, &push_map);
   NIR_PASS(_, nir, anv_nir_lower_resource_intel, pdev, bind_map.layout_type);

   /* 4. compile to EU ISA. Use brw's own params union: every stage's params struct starts with the
    * shared .base, and the per-stage tails differ, so a single stage's struct is NOT a safe superset
    * for the others (fragment's max_polygons sits exactly where mesh keeps a wa_18019110168 function
    * pointer, for instance). Zero the union, fill .base, and set only the fields of the stage in hand;
    * everything left zero means "not supplied", which brw handles (NULL vue_map / tue_map). */
   /* Ray tracing stages: split the shader at every point it can suspend. A traceRay or callable call
    * yields back to the RT dispatcher, so brw compiles an RT shader as a MAIN shader plus N RESUME
    * shaders (one per continuation), all handed to brw_compile_bs together. nir_lower_shader_calls does
    * the split and hands back the resume shaders; the brw passes around it lower the RT intrinsics
    * (ray/hit/instance queries, the BTD stack) in main and every resume shader alike. This mirrors
    * ANV's populate_compile_params_bs, and needs nothing device-side: only devinfo and zeroed keys. */
   nir_shader **resume_shaders = NULL;
   uint32_t num_resume_shaders = 0;
   if (s2i_stage_is_rt(ms)) {
      /* First the per-stage entry lowering, which gives each RT stage its dispatcher entry/return ABI
       * (reading the ray/hit payload it was invoked with, returning through the BTD stack). An
       * INTERSECTION shader has no lowering of its own: on Intel its any-hit runs INSIDE it, so the
       * two are merged instead. We pass NULL for the any-hit, which brw explicitly allows and which
       * matches compiling one shader at a time: reportIntersection then behaves as accept-always.
       * Without this merge nir_lower_shader_calls asserts "Any-hit shaders must be inlined". */
      switch (ms) {
      case MESA_SHADER_RAYGEN:       brw_nir_lower_raygen(nir, &devinfo); break;
      case MESA_SHADER_ANY_HIT:      brw_nir_lower_any_hit(nir, &devinfo); break;
      case MESA_SHADER_CLOSEST_HIT:  brw_nir_lower_closest_hit(nir, &devinfo); break;
      case MESA_SHADER_MISS:         brw_nir_lower_miss(nir, &devinfo); break;
      case MESA_SHADER_CALLABLE:     brw_nir_lower_callable(nir, &devinfo); break;
      case MESA_SHADER_INTERSECTION:
         brw_nir_lower_combined_intersection_any_hit(nir, NULL, &devinfo);
         break;
      default: break;
      }

      struct brw_nir_lower_shader_calls_state calls_state = {
         .devinfo = &devinfo,
         .key = &prog_key.bs,
      };
      struct brw_nir_vectorize_mem_cb_data vectorize_cb_data = { .devinfo = &devinfo };
      const nir_lower_shader_calls_options call_opts = {
         .address_format = nir_address_format_64bit_global,
         .stack_alignment = BRW_BTD_STACK_ALIGN,
         .localized_loads = true,
         .vectorizer_callback = brw_nir_should_vectorize_mem,
         .vectorizer_data = &vectorize_cb_data,
         /* No should_remat_callback: ANV rematerializes its resource_intel intrinsics across the
          * split, and our device-free descriptor lowering never emits those. NULL is handled. */
      };

      NIR_PASS(_, nir, brw_nir_lower_rt_intrinsics_pre_trace);
      NIR_PASS(_, nir, nir_lower_shader_calls, &call_opts, &resume_shaders, &num_resume_shaders,
               mem_ctx);
      NIR_PASS(_, nir, brw_nir_lower_shader_calls, &calls_state);
      NIR_PASS(_, nir, brw_nir_lower_rt_intrinsics, base_prog_key, &devinfo);

      for (uint32_t i = 0; i < num_resume_shaders; i++) {
         NIR_PASS(_, resume_shaders[i], brw_nir_lower_shader_calls, &calls_state);
         NIR_PASS(_, resume_shaders[i], brw_nir_lower_rt_intrinsics, base_prog_key, &devinfo);
      }
   }

   union brw_any_compile_params params;
   memset(&params, 0, sizeof(params));
   params.base.mem_ctx = mem_ctx;
   params.base.nir = nir;
   params.base.key = base_prog_key;
   params.base.prog_data = base_prog_data;
   if (ms == MESA_SHADER_FRAGMENT)
      params.fs.max_polygons = 1; /* one polygon per dispatch */
   if (s2i_stage_is_rt(ms)) {
      params.bs.num_resume_shaders = num_resume_shaders;
      params.bs.resume_shaders = resume_shaders;
   }

   /* Refuse by name anything no pass claimed, instead of letting brw abort on an intrinsic it
    * has no case for. A descriptor-heap module reaches here this way. */

   const char *unlowered = s2i_intel_first_unlowered(nir);

   if (unlowered) {

      if (message) {
         char buf[192];
         snprintf(buf, sizeof(buf), "%s is not lowered by this backend yet, so the module cannot be "
                                    "compiled device-free", unlowered);
         *message = strdup(buf);
      }

      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_UNSUPPORTED_CAP;
   }

   /* Capture the EU disassembly brw prints to stderr under INTEL_DEBUG. We set the stage's disasm bit
    * in the intel_debug global (NOT the NIR bit, so no NIR noise) and redirect stderr to a temp file
    * just around brw_compile, then read it back. This mutates process-global state (the intel_debug
    * bitset + fd 2), so it is not thread-safe; fine for the CLI, worth a real brw disasm API later. */
   const uint64_t dbg_flag = intel_debug_flag_for_shader_stage(ms);
   const bool dbg_was_set = BITSET_TEST(intel_debug, dbg_flag);
   FILE *cap = NULL;
   int saved_stderr = -1;
   if (isa_text && !getenv("S2I_NO_CAPTURE")) {
      if (!dbg_was_set)
         BITSET_SET(intel_debug, dbg_flag);
      cap = tmpfile();
      if (cap) {
         fflush(stderr);
         saved_stderr = dup(2);
         dup2(fileno(cap), 2);
      }
   }

   const unsigned *program = brw_compile(compiler, &params.base);

   if (isa_text) {
      if (saved_stderr >= 0) {
         fflush(stderr);
         dup2(saved_stderr, 2);
         close(saved_stderr);
      }
      if (!dbg_was_set)
         BITSET_CLEAR(intel_debug, dbg_flag);
   }

   if (!program) {
      if (message)
         *message = strdup(params.base.error_str ? params.base.error_str : "brw_compile failed");
      if (cap)
         fclose(cap);
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_COMPILE_FAILED;
   }

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->grf_used = base_prog_data->grf_used;
      stats->program_size = base_prog_data->program_size;
      stats->scratch_size = base_prog_data->total_scratch;
      stats->shared_size = base_prog_data->total_shared;
      /* Dispatch width. Compute, task and mesh all report a valid-SIMD-variant mask (bit0=8, bit1=16,
       * bit2=32) in the brw_cs_prog_data they share; the ray tracing stages carry a plain width in
       * brw_bs_prog_data instead. The fixed-function graphics stages have no single dispatch width, so
       * they keep 0. */
      if (ms == MESA_SHADER_COMPUTE || ms == MESA_SHADER_TASK || ms == MESA_SHADER_MESH) {
         const unsigned m = ms == MESA_SHADER_COMPUTE ? prog_data.cs.prog_mask
                          : ms == MESA_SHADER_TASK    ? prog_data.task.base.prog_mask
                                                      : prog_data.mesh.base.prog_mask;
         stats->simd_width = (m & 4) ? 32 : (m & 2) ? 16 : (m & 1) ? 8 : 0;
      } else if (s2i_stage_is_rt(ms)) {
         stats->simd_width = prog_data.bs.simd_size;
      }
   }

   if (cap) {
      fseek(cap, 0, SEEK_END);
      long sz = ftell(cap);
      fseek(cap, 0, SEEK_SET);
      if (isa_text && sz > 0) {
         char *buf = (char *)malloc(sz + 1);
         if (buf) {
            size_t rd = fread(buf, 1, sz, cap);
            buf[rd] = '\0';
            /* brw prints the NIR listing before the "Native code" disassembly under the same stage
             * debug flag; keep only the machine code (from the first "Native code" header). */
            const char *asm_start = strstr(buf, "Native code");
            *isa_text = strdup(asm_start ? asm_start : buf);
            free(buf);
         }
      }
      fclose(cap);
   }

   glsl_type_singleton_decref();
   ralloc_free(mem_ctx);
   return S2I_OK;
}
