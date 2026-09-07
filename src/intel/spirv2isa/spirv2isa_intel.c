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
#include "compiler/spirv/nir_spirv.h"     /* spirv_to_nir + options */
#include "compiler/spirv/spirv_info.h"    /* struct spirv_capabilities */
#include "brw_compiler.h"                  /* brw_compiler_create, brw_compile, prog_data/key */
#include "brw_reg.h"                       /* REG_SIZE, the push-constant granularity */
#include "brw_nir.h"                       /* brw_preprocess_nir, brw_nir_lower_cs_intrinsics, brw_nir_lower_cmat */
#include "brw_nir_rt.h"                    /* brw_nir_lower_ray_queries (inline ray tracing) */
#include "anv_nir.h"                       /* ANV's descriptor, driver-value and multiview passes */
#include "anv_shader.h"                     /* anv_shader_data + the offline entry points */
#include "vk_nir.h"                        /* vk_spirv_to_nir, the runtime's SPIR-V front door */
#include "vk_pipeline.h"                   /* vk_pipeline_robustness_state */
#include "util/simple_mtx.h"
#include "spirv2isa_stage.h"
#include "spirv2isa_intel_anv.h"           /* the inputs those passes read, built without a device */

/* The ISA capture mutates process globals (the intel_debug bitset and fd 2), so captures are
 * serialized; concurrent compiles would otherwise interleave their disassembly. */
static simple_mtx_t s2i_intel_capture_mtx = SIMPLE_MTX_INITIALIZER;

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

/* The descriptor type a resource variable would need in a layout. The inference is only for the
 * synthesized fallback; a caller-supplied layout is taken at its word. */
static VkDescriptorType
s2i_intel_var_descriptor_type(const nir_variable *var)
{
   const struct glsl_type *type = glsl_without_array(var->type);

   if (var->data.mode == nir_var_mem_ubo)
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

   if (var->data.mode == nir_var_mem_ssbo)
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

   if (glsl_type_is_image(type)) {
      return glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF ?
             VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   }

   if (glsl_type_is_sampler(type))
      return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

   if (glsl_type_is_texture(type)) {
      return glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF ?
             VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
   }

   return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
}

#define S2I_INTEL_RESOURCE_MODES \
   (nir_var_uniform | nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_image)

/* ANV's lowering assumes a layout covers every descriptor the module touches, because a driver is
 * handed one the validation layers already vetted. Nothing vets one here, so a supplied layout is
 * checked against the module (a miss is a refusal, not corruption two passes later), and no layout at
 * all synthesizes one from the module's own variables, which is the fallback the API promises. */
static bool
s2i_intel_resolve_layouts(nir_shader *nir,
                          const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                          uint32_t set_layout_count, void *mem_ctx,
                          const VkDescriptorSetLayoutCreateInfo *const **out_layouts,
                          uint32_t *out_count, char **message)
{
   const bool synthesize = set_layout_count == 0;

   uint32_t nbind[S2I_INTEL_ANV_MAX_SETS];
   memset(nbind, 0, sizeof(nbind));

   uint32_t highest_set = 0;
   bool any = false;

   nir_foreach_variable_with_modes(var, nir, S2I_INTEL_RESOURCE_MODES) {
      const uint32_t set = var->data.descriptor_set;
      const uint32_t binding = var->data.binding;
      const uint32_t aoa = glsl_get_aoa_size(var->type);
      const uint32_t count = aoa ? aoa : 1;

      if (set >= S2I_INTEL_ANV_MAX_SETS) {

         if (message) {
            char buf[128];
            snprintf(buf, sizeof(buf), "the module uses descriptor set %u; sets go up to %u", set,
                     S2I_INTEL_ANV_MAX_SETS - 1);
            *message = strdup(buf);
         }

         return false;
      }

      any = true;
      highest_set = MAX2(highest_set, set);
      nbind[set] = MAX2(nbind[set], binding + 1);

      if (synthesize)
         continue;

      const VkDescriptorSetLayoutBinding *found = NULL;

      if (set < set_layout_count && set_layouts[set]) {
         for (uint32_t b = 0; b < set_layouts[set]->bindingCount; b++) {
            if (set_layouts[set]->pBindings[b].binding == binding) {
               found = &set_layouts[set]->pBindings[b];
               break;
            }
         }
      }

      if (!found || found->descriptorCount < count) {

         if (message) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "the module uses descriptor set %u binding %u (count %u), which the supplied "
                     "layout does not %s", set, binding, count,
                     found ? "cover at that count" : "contain");
            *message = strdup(buf);
         }

         return false;
      }
   }

   if (!synthesize || !any) {
      *out_layouts = set_layouts;
      *out_count = synthesize ? 0 : set_layout_count;
      return true;
   }

   const uint32_t count = highest_set + 1;

   VkDescriptorSetLayoutCreateInfo *infos =
      rzalloc_array(mem_ctx, VkDescriptorSetLayoutCreateInfo, count);
   const VkDescriptorSetLayoutCreateInfo **ptrs =
      rzalloc_array(mem_ctx, const VkDescriptorSetLayoutCreateInfo *, count);

   for (uint32_t set = 0; set < count; set++) {
      infos[set].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

      if (nbind[set])
         infos[set].pBindings = rzalloc_array(mem_ctx, VkDescriptorSetLayoutBinding, nbind[set]);

      ptrs[set] = &infos[set];
   }

   nir_foreach_variable_with_modes(var, nir, S2I_INTEL_RESOURCE_MODES) {
      VkDescriptorSetLayoutCreateInfo *info = &infos[var->data.descriptor_set];
      VkDescriptorSetLayoutBinding *bindings = (VkDescriptorSetLayoutBinding *)info->pBindings;
      const uint32_t aoa = glsl_get_aoa_size(var->type);

      bool merged = false;

      for (uint32_t b = 0; b < info->bindingCount; b++) {
         if (bindings[b].binding == var->data.binding) {
            merged = true;
            break;
         }
      }

      if (merged)
         continue;

      bindings[info->bindingCount++] = (VkDescriptorSetLayoutBinding) {
         .binding = var->data.binding,
         .descriptorType = s2i_intel_var_descriptor_type(var),
         .descriptorCount = aoa ? aoa : 1,
         .stageFlags = VK_SHADER_STAGE_ALL,
      };
   }

   *out_layouts = ptrs;
   *out_count = count;
   return true;
}

s2i_result
s2i_intel_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                  int target_index, const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                             uint32_t set_layout_count,
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

   /* All of compute + vertex + fragment + geometry + tessellation control/eval are wired. The graphics
    * stages run their own input lowering inside brw_compile_*; TES computes its input VUE map itself
    * when we leave it NULL. (Mesh/task are not in the s2i_stage enum yet.) */

   /* Initialize the INTEL_DEBUG / INTEL_SIMD globals a driver would set at startup; without this the
    * SIMD-width selection reads every width as disabled and nothing compiles. call_once-guarded. */
   process_intel_debug_variable();

   void *mem_ctx = ralloc_context(NULL);

   /* No device did this for us; the glsl type system uses a singleton linear allocator. */
   glsl_type_singleton_init_or_ref();

   /* Device-free devinfo from a PCI id, then the brw compiler from just that. */
   struct intel_device_info devinfo;
   if (!intel_get_device_info_from_pci_id(s2i_intel_targets[target_index].pci_id, &devinfo)) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      if (message)
         *message = strdup("intel_get_device_info_from_pci_id failed");
      return S2I_BAD_TARGET;
   }
   /* Normally filled by the kernel from a DRM query, and left 0 by the PCI-id path. It gates brw's
    * load/store vectorizer, so it decides how many loads merge and therefore the instruction count.
    * Derived the way the kernel derives it (i915/intel_device_info.c). */
   if (devinfo.mem_alignment == 0)
      devinfo.mem_alignment = (devinfo.verx10 >= 125 || devinfo.has_local_mem) ? 64 * 1024 : 4096;
   struct brw_compiler *compiler = brw_compiler_create(mem_ctx, &devinfo);

   /* Must exist before spirv_to_nir: the address formats below are derived from it. */
   struct anv_physical_device *pdev = rzalloc(mem_ctx, struct anv_physical_device);
   s2i_intel_anv_init_physical_device(pdev, &devinfo, compiler, mem_ctx);

   /* Robust access is caller pipeline state, and each field flips a UBO/SSBO address format between
    * the offset and bounded forms, so it moves every buffer access. OxC3 does not enable it today;
    * when it does, this becomes a declared toggle on s2i_compile rather than a constant. */
   static const struct vk_pipeline_robustness_state s2i_rs_disabled = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   compiler->shader_debug_log = s2i_intel_log_noop;
   compiler->shader_perf_log = s2i_intel_log_noop;

   const mesa_shader_stage ms = s2i_stage_to_mesa(stage);

   /* Only ->physical and the mirrored fields are read; the log stub answers for the runtime. */
   struct anv_device *device = rzalloc(mem_ctx, struct anv_device);
   device->physical = pdev;
   device->info = &pdev->info;
   device->isl_dev = pdev->isl_dev;
   device->vk.physical = &pdev->vk;

   /* The runtime's own SPIR-V front door: capabilities computed from the feature tables this pdev
    * filled, the parse, then the finalization every driver-facing module has had (returns lowered,
    * functions inlined, clip and cull distances merged). A foreign module (glslang, -Od) arrives in
    * the same shape DXC output does. */
   struct spirv_to_nir_options spirv_opts =
      anv_shader_offline_spirv_options(&pdev->vk, ms, &s2i_rs_disabled);
   const struct nir_shader_compiler_options *nir_options =
      anv_shader_offline_nir_options(&pdev->vk, ms, &s2i_rs_disabled);

   /* vtn treats an unsupported capability as a warning and parses on, because a driver can rely on
    * the validation layers having already rejected the module. Nothing vets a module here, so each
    * declared capability is checked against the same set vk_spirv_to_nir computes, and a miss is a
    * refusal by name instead of a warning followed by an abort inside brw. */
   const struct spirv_capabilities target_caps =
      vk_physical_device_get_spirv_capabilities(&pdev->vk);

   for (size_t w = 5; w + 1 < spirv_words && (spirv[w] & 0xffffu) == SpvOpCapability;
        w += spirv[w] >> 16) {

      const SpvCapability cap = (SpvCapability)spirv[w + 1];

      if (spirv_capabilities_get(&target_caps, cap))
         continue;

      if (message) {
         char buf[160];
         snprintf(buf, sizeof(buf), "the module declares SPIR-V capability %u (%s), which %s does "
                  "not support", (unsigned)cap, spirv_capability_to_string(cap),
                  s2i_intel_targets[target_index].name);
         *message = strdup(buf);
      }

      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_UNSUPPORTED_CAP;
   }

   nir_shader *nir = vk_spirv_to_nir(&device->vk, spirv, spirv_words * 4, ms, entry, NULL,
                                     &spirv_opts, nir_options, false, mem_ctx);

   if (!nir) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      if (message)
         *message = strdup("vk_spirv_to_nir failed (bad entrypoint, or a capability this target "
                           "does not support; diagnostics on stderr)");
      return S2I_NO_ENTRYPOINT;
   }

   /* vk_pipeline.c's vk_set_subgroup_size, with the sizes ANV reports (get_properties_1_3): the
    * parser's 1..128 defaults would otherwise reach anv_fixup_subgroup_size, which pins cooperative
    * matrix shaders to a width the hardware does not have. A required subgroup size stays a future
    * declared input. */
   if (mesa_shader_stage_uses_workgroup(ms)) {

      if (spirv[1] < 0x10600) {
         /* Before SPIR-V 1.6 the api subgroup size is the device's, not "varying". */
         nir->info.api_subgroup_size = 32;
         nir->info.max_subgroup_size = 32;
      }

      nir->info.max_subgroup_size = MIN2(nir->info.max_subgroup_size, 32);
      nir->info.min_subgroup_size = MAX2(nir->info.min_subgroup_size, devinfo.ver >= 20 ? 16 : 8);
   }

   /* ANV's preprocess hook: sysvals to varyings, access flags, brw's preprocessing, barrier modes. */
   anv_shader_offline_preprocess(&pdev->vk, nir, &s2i_rs_disabled);

   /* ANV's lowering, called rather than mirrored: cooperative matrix, storage images, multiview, ray
    * queries, descriptors and the push-constant layout, in the order ANV maintains. */

   const VkDescriptorSetLayoutCreateInfo *const *eff_layouts = NULL;
   uint32_t eff_layout_count = 0;

   if (!s2i_intel_resolve_layouts(nir, set_layouts, set_layout_count, mem_ctx, &eff_layouts,
                                  &eff_layout_count, message)) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_UNSUPPORTED_CAP;
   }

   s2i_intel_anv_layouts layouts;

   if (!s2i_intel_anv_build_layouts(pdev, eff_layouts, eff_layout_count, &layouts, message)) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_UNSUPPORTED_CAP;
   }

   struct vk_shader_compile_info compile_info;
   memset(&compile_info, 0, sizeof(compile_info));
   compile_info.stage = ms;
   compile_info.nir = nir;
   compile_info.set_layouts = (struct vk_descriptor_set_layout **) layouts.sets;
   compile_info.set_layout_count = layouts.set_count;

   compile_info.robustness = &s2i_rs_disabled;

   struct anv_shader_data shader_data;
   memset(&shader_data, 0, sizeof(shader_data));
   shader_data.info = &compile_info;

   /* ANV's own key population. The link mask models the classic pipeline around this stage, which is
    * what this tool models: both tessellation stages linked, a fragment stage present, mesh pipelines
    * modeled only when compiling their own stages. The day s2i_compile carries graphics pipeline
    * state, the NULL state and this mask become caller data (view mask, mesh presence, coarse pixel,
    * sample counts all live there). */
   VkShaderStageFlags link_stages = 0;

   switch (ms) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_FRAGMENT:
      link_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT;
      break;
   case MESA_SHADER_TASK:
   case MESA_SHADER_MESH:
      link_stages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                    VK_SHADER_STAGE_FRAGMENT_BIT;
      break;
   default:
      break;
   }

   anv_shader_offline_populate_key(&pdev->vk, &shader_data, NULL /* graphics state */, link_stages);

   struct brw_stage_prog_data *base_prog_data;
   struct brw_base_prog_key *base_prog_key;

   if (ms == MESA_SHADER_COMPUTE) {
      base_prog_data = &shader_data.prog_data.cs.base;
      base_prog_key = &shader_data.key.cs.base;
   } else if (ms == MESA_SHADER_FRAGMENT) {
      base_prog_data = &shader_data.prog_data.fs.base;
      base_prog_key = &shader_data.key.fs.base;
   } else if (s2i_stage_is_rt(ms)) {
      /* All six RT stages compile through brw_compile_bs; brw_bs_prog_data nests the stage base
       * directly (like fragment, unlike the VUE stages). */
      base_prog_data = &shader_data.prog_data.bs.base;
      base_prog_key = &shader_data.key.bs.base;
   } else if (ms == MESA_SHADER_TASK || ms == MESA_SHADER_MESH) {
      /* Task/mesh prog_data nest brw_cs_prog_data (they dispatch like compute), so the stage base is
       * two levels down. Unlike compute, brw_compile_task/mesh run brw_nir_lower_cs_intrinsics
       * themselves, so we must not do it here. A mesh shader compiled on its own gets tue_map = NULL,
       * which brw explicitly supports (task and mesh need not be compiled together). */
      base_prog_data = ms == MESA_SHADER_TASK ? &shader_data.prog_data.task.base.base : &shader_data.prog_data.mesh.base.base;
      base_prog_key = ms == MESA_SHADER_TASK ? &shader_data.key.task.base : &shader_data.key.mesh.base;
   } else {
      /* VUE stages (vertex / geometry / tess-ctrl / tess-eval): the prog_data nests
       * brw_vue_prog_data -> brw_stage_prog_data; the union members overlap at offset 0, so the vs
       * view addresses the shared base. brw_compile_* casts to the real per-stage prog_data/key. */
      base_prog_data = &shader_data.prog_data.vs.base.base;
      base_prog_key = &shader_data.key.vs.base;
   }

   /* The lowering reads the stage off prog_data, and runs before brw_compile would set it. */
   base_prog_data->stage = ms;

   /* Assigned outside the lowering, so it stays ours. Zero is UNKNOWN, which is neither DIRECT nor
    * INDIRECT, and almost every site tests for one of those two. */
   shader_data.bind_map.layout_type =
      layouts.set_count > 0 ? layouts.sets[0]->type
                            : ANV_PIPELINE_DESCRIPTOR_SET_LAYOUT_TYPE_DIRECT;

   /* Mesh, task and the RT stages address every resource bindlessly, so ANV gives them no tables at
    * all; the sized ones use ANV's own binding table bound. */
   if (!brw_shader_stage_requires_bindless_resources(ms)) {
      shader_data.bind_map.surface_to_descriptor =
         rzalloc_array(mem_ctx, struct anv_pipeline_binding, MAX_BINDING_TABLE_SIZE);
      shader_data.bind_map.sampler_to_descriptor =
         rzalloc_array(mem_ctx, struct anv_pipeline_binding, MAX_BINDING_TABLE_SIZE);
   }

   anv_shader_lower_nir(device, mem_ctx, NULL /* graphics pipeline state */, &shader_data);

   /* The passes read the layouts, they do not keep them. */
   s2i_intel_anv_free_layouts(&layouts);


   /* Compile to EU ISA. Use brw's own params union: every stage's params struct starts with the
    * shared .base, and the per-stage tails differ, so a single stage's struct is NOT a safe superset
    * for the others (fragment's max_polygons sits exactly where mesh keeps a wa_18019110168 function
    * pointer, for instance). Zero the union, fill .base, and set only the fields of the stage in hand;
    * everything left zero means "not supplied", which brw handles (NULL vue_map / tue_map). */
   union brw_any_compile_params params;
   memset(&params, 0, sizeof(params));
   params.base.mem_ctx = mem_ctx;
   params.base.nir = nir;
   params.base.key = base_prog_key;
   params.base.prog_data = base_prog_data;
   /* brw fills the per-variant numbers only into params->stats; prog_data->program_size counts every
    * variant plus the constant data, which is not the shader's size. ANV passes the same array. */
   params.base.stats = shader_data.stats;

   /* ANV's finish: subgroup fixup, shader workarounds, the RT entry lowering and shader-call split,
    * and the per-stage compile params (the TCS key copy, TES and mesh and fragment inputs, the RT
    * resume shaders). The NULL is the intersection shader's any-hit, which becomes caller data when
    * the RT pipeline API carries hit groups. */
   anv_shader_offline_finish(device, &shader_data, NULL, &params, mem_ctx);

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
    * bitset + fd 2), so captures are serialized on s2i_intel_capture_mtx; a real brw disasm API would
    * remove the global mutation and the mutex with it. */
   const uint64_t dbg_flag = intel_debug_flag_for_shader_stage(ms);
   const bool dbg_was_set = BITSET_TEST(intel_debug, dbg_flag);
   FILE *cap = NULL;
   int saved_stderr = -1;
   bool capture_locked = false;

   if (isa_text && !getenv("S2I_NO_CAPTURE")) {
      simple_mtx_lock(&s2i_intel_capture_mtx);
      capture_locked = true;

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

   if (capture_locked)
      simple_mtx_unlock(&s2i_intel_capture_mtx);

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
      /* The widest variant brw actually compiled is the one the numbers describe; a fragment shader
       * has several, and the earlier entries are the narrower ones. */
      const struct genisa_stats *widest = NULL;

      for (unsigned i = 0; i < ARRAY_SIZE(shader_data.stats); i++) {
         if (shader_data.stats[i].dispatch_width != 0)
            widest = &shader_data.stats[i];
      }

      /* brw only fills grf_registers from Xe3 on (brw_to_binary.cpp), so that one always comes off
       * prog_data; the rest describe the variant and are only in the stats. */
      stats->grf_used = base_prog_data->grf_used;

      if (widest) {
         stats->program_size = widest->code_size;
         stats->scratch_size = widest->scratch_memory_size;
         stats->shared_size = widest->workgroup_memory_size;
         stats->simd_width = widest->dispatch_width;
      } else {
         stats->program_size = base_prog_data->program_size;
         stats->scratch_size = base_prog_data->total_scratch;
         stats->shared_size = base_prog_data->total_shared;
      }
      /* Dispatch width. Compute, task and mesh all report a valid-SIMD-variant mask (bit0=8, bit1=16,
       * bit2=32) in the brw_cs_prog_data they share; the ray tracing stages carry a plain width in
       * brw_bs_prog_data instead. The fixed-function graphics stages have no single dispatch width, so
       * they keep 0. */
      if (stats->simd_width != 0) {
         /* brw already reported it */
      } else if (ms == MESA_SHADER_COMPUTE || ms == MESA_SHADER_TASK || ms == MESA_SHADER_MESH) {
         const unsigned m = ms == MESA_SHADER_COMPUTE ? shader_data.prog_data.cs.prog_mask
                          : ms == MESA_SHADER_TASK    ? shader_data.prog_data.task.base.prog_mask
                                                      : shader_data.prog_data.mesh.base.prog_mask;
         stats->simd_width = (m & 4) ? 32 : (m & 2) ? 16 : (m & 1) ? 8 : 0;
      } else if (s2i_stage_is_rt(ms)) {
         stats->simd_width = shader_data.prog_data.bs.simd_size;
      }

      /* Ray tracing runs on a per-ray stack rather than scratch, accumulated across the main shader
       * and its resume shaders, so scratch alone makes an RT shader look free. */
      if (s2i_stage_is_rt(ms))
         stats->stack_size = shader_data.prog_data.bs.max_stack_size;
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
