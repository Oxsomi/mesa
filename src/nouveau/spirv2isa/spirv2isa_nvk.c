/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa NVIDIA backend: SPIR-V -> SASS through NVK's own lowering and NAK, with no device.
 * See spirv2isa_nvk.h. The shape mirrors the Intel backend: the runtime's SPIR-V front door with
 * real capabilities, the driver's own descriptor layout arithmetic and NIR lowering reached through
 * guarded rebuilds of the driver's files, then the compiler. NAK is public C API and returns both
 * the disassembly and its stats in the result, so there is no capture apparatus here at all.
 */

#include <stdbool.h>
#include "util/list.h"
#include "c11/threads.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "spirv2isa_nvk.h"
#include "spirv2isa_stage.h"
#include "spirv2isa_caps.h"          /* the declared-capability gate every backend shares */

#include "util/ralloc.h"
#include "compiler/spirv/spirv_info.h"
#include "vk_nir.h"
#include "vk_physical_device.h"
#include "vk_pipeline.h"
#include "spirv2isa_state.h"       /* the pipeline state a module does not carry */
#include "spirv2isa_session.h"     /* the generic setup and its one teardown */
#include "spirv2isa_layout.h"      /* the descriptor layout resolve every backend shares */

#include "compiler/nir/nir.h"

#include "nak.h"
#include "nv_device_info.h"

#include "nvk_physical_device.h"
#include "nvk_instance.h"
#include "nvk_device.h"
#include "nvk_shader.h"
#include "nvk_descriptor_set_layout.h"
#include "nvk_drirc.h"
#include "nvk_private.h"
#include "nvk_sampler.h"

/* The 3D and compute class numbers per generation, from the tree's own class headers. */
#include "clb197.h" /* MAXWELL_B */
#include "clb1c0.h" /* MAXWELL_COMPUTE_B */
#include "clc197.h" /* PASCAL_B */
#include "clc1c0.h" /* PASCAL_COMPUTE_B */
#include "clc597.h" /* TURING_A */
#include "clc5c0.h" /* TURING_COMPUTE_A */
#include "clc797.h" /* AMPERE_B */
#include "clc7c0.h" /* AMPERE_COMPUTE_B */
#include "clc997.h" /* ADA_A */
#include "clc9c0.h" /* ADA_COMPUTE_A */
#include "clce97.h" /* BLACKWELL_B */
#include "clcec0.h" /* BLACKWELL_COMPUTE_B */

/* The flagship tier first (one die per generation, the public enum's constants, what a listing
 * shows by default), then the extended tier: the rest of each generation's consumer line, named by
 * token. A row carries only what a device would have had to report and nothing here can work out:
 * everything else about the chip follows from the chipset, so it is derived rather than written
 * down. The chipset numbering is the kernel's, and each id is named in the nouveau driver's own
 * chipset table (drivers/gpu/drm/nouveau/nvkm/engine/device/base.c), which is where a new row's
 * value comes from and where an existing one can be checked. */
struct s2i_nvk_target_desc {
   uint16_t chipset;
   const char *name;
   const char *token; /* extended tier only; a flagship's token lives in the dispatcher */
};

static const struct s2i_nvk_target_desc s2i_nvk_targets[] = {
   { 0x124, "GM204 (GTX 980)"        },
   { 0x132, "GP102 (GTX 1080 Ti)"    },
   { 0x162, "TU102 (RTX 2080 Ti)"    },
   { 0x172, "GA102 (RTX 3080/90)"    },
   { 0x192, "AD102 (RTX 4090)"       },
   { 0x1b2, "GB202 (RTX 5090)"       },

   { 0x120, "GM200 (GTX 980 Ti)",    "gm200" },
   { 0x126, "GM206 (GTX 950/960)",   "gm206" },
   { 0x134, "GP104 (GTX 1070/80)",   "gp104" },
   { 0x136, "GP106 (GTX 1060)",      "gp106" },
   { 0x137, "GP107 (GTX 1050 Ti)",   "gp107" },
   { 0x138, "GP108 (GT 1030)",       "gp108" },
   { 0x164, "TU104 (RTX 2080)",      "tu104" },
   { 0x166, "TU106 (RTX 2060/70)",   "tu106" },
   { 0x168, "TU116 (GTX 1660)",      "tu116" },
   { 0x167, "TU117 (GTX 1650)",      "tu117" },
   { 0x174, "GA104 (RTX 3070)",      "ga104" },
   { 0x176, "GA106 (RTX 3060)",      "ga106" },
   { 0x177, "GA107 (RTX 3050)",      "ga107" },
   { 0x193, "AD103 (RTX 4080)",      "ad103" },
   { 0x194, "AD104 (RTX 4070)",      "ad104" },
   { 0x196, "AD106 (RTX 4060 Ti)",   "ad106" },
   { 0x197, "AD107 (RTX 4060)",      "ad107" },
   { 0x1b3, "GB203 (RTX 5080)",      "gb203" },
   { 0x1b5, "GB205 (RTX 5070)",      "gb205" },
   { 0x1b6, "GB206 (RTX 5060)",      "gb206" },
   { 0x1b7, "GB207 (RTX 5050)",      "gb207" },
};

const char *
s2i_nvk_target_name(int target_index)
{
   if (target_index < 0 || target_index >= (int)ARRAY_SIZE(s2i_nvk_targets))
      return "";

   return s2i_nvk_targets[target_index].name;
}

int
s2i_nvk_target_count(void)
{
   return (int)ARRAY_SIZE(s2i_nvk_targets);
}

const char *
s2i_nvk_target_token(int target_index)
{
   if (target_index < (int)S2I_TARGET_NVIDIA_COUNT ||
       target_index >= (int)ARRAY_SIZE(s2i_nvk_targets))
      return "";

   return s2i_nvk_targets[target_index].token;
}

/* The 3D and compute classes a chip exposes, which a real device reads off the kernel. They follow
 * the shader model, which already separates the two shapes a generation ships: the compute chip's
 * _A pair against the consumer refresh's _B (GP100 SM60 vs GP10x SM61, GA100 SM80 vs GA10x SM86).
 * An SM this does not name is refused rather than given a neighbour's classes. */
static bool
s2i_nvk_classes_for_sm(uint8_t sm, uint16_t *cls_eng3d, uint16_t *cls_compute)
{
   switch (sm) {

   case 52:
      *cls_eng3d = MAXWELL_B;
      *cls_compute = MAXWELL_COMPUTE_B;
      return true;

   case 61:
      *cls_eng3d = PASCAL_B;
      *cls_compute = PASCAL_COMPUTE_B;
      return true;

   case 75:
      *cls_eng3d = TURING_A;
      *cls_compute = TURING_COMPUTE_A;
      return true;

   case 86:
      *cls_eng3d = AMPERE_B;
      *cls_compute = AMPERE_COMPUTE_B;
      return true;

   case 89:
      *cls_eng3d = ADA_A;
      *cls_compute = ADA_COMPUTE_A;
      return true;

   case 120:
      *cls_eng3d = BLACKWELL_B;
      *cls_compute = BLACKWELL_COMPUTE_B;
      return true;

   default:
      return false;
   }
}

static bool
s2i_nvk_fill_device_info(const struct s2i_nvk_target_desc *t, struct nv_device_info *info)
{
   memset(info, 0, sizeof(*info));

   info->type = NV_DEVICE_TYPE_DIS;

   /* Shader model, occupancy, SM layout and the shared memory ladder all follow from the chipset,
    * so they come from the winsys rather than from a table here that could disagree with it. */
   nv_device_info_from_chipset(info, t->chipset);

   /* GPC and TPC counts stay unset: they only size a per-TPC runtime allocation and the
    * shaderSMCount property, neither of which a compile reaches. */

   if (!s2i_nvk_classes_for_sm(info->sm, &info->cls_eng3d, &info->cls_compute))
      return false;

   snprintf(info->device_name, sizeof(info->device_name), "%s", t->name);
   snprintf(info->chipset_name, sizeof(info->chipset_name), "NV%03x", t->chipset);
   return true;
}

/* NVK's layout arithmetic on caller create-infos, through the count/init split. Immutable sampler
 * handles are replaced by one plain stand-in sampler, exactly as on Intel: the layout init stores
 * the pointer and reads only the ycbcr conversion off it (none), so a plain non-ycbcr immutable
 * sampler lays out identically to the driver's. */
static struct nvk_descriptor_set_layout *
s2i_nvk_build_one_layout(const struct nvk_physical_device *pdev,
                         const VkDescriptorSetLayoutCreateInfo *ci,
                         struct nvk_sampler **stand_in, char **message)
{
   VkDescriptorSetLayoutCreateInfo local_ci;
   VkDescriptorSetLayoutBinding *local_bindings = NULL;
   VkSampler *stand_ins = NULL;
   bool any_immutable = false;

   for (uint32_t b = 0; b < ci->bindingCount; b++) {
      if (ci->pBindings[b].pImmutableSamplers)
         any_immutable = true;
   }

   if (any_immutable) {

      if (!*stand_in)
         *stand_in = calloc(1, sizeof(**stand_in));

      uint32_t max_count = 1;

      for (uint32_t b = 0; b < ci->bindingCount; b++)
         max_count = MAX2(max_count, ci->pBindings[b].descriptorCount);

      local_bindings = calloc(ci->bindingCount, sizeof(*local_bindings));
      stand_ins = calloc(max_count, sizeof(*stand_ins));

      if (!*stand_in || !local_bindings || !stand_ins) {

         if (message)
            *message = strdup("out of memory building a descriptor set layout");

         free(local_bindings);
         free(stand_ins);
         return NULL;
      }

      for (uint32_t i = 0; i < max_count; i++)
         stand_ins[i] = nvk_sampler_to_handle(*stand_in);

      for (uint32_t b = 0; b < ci->bindingCount; b++) {
         local_bindings[b] = ci->pBindings[b];

         if (local_bindings[b].pImmutableSamplers)
            local_bindings[b].pImmutableSamplers = stand_ins;
      }

      local_ci = *ci;
      local_ci.pBindings = local_bindings;
      ci = &local_ci;
   }

   uint32_t num_bindings, immutable_sampler_count;
   nvk_descriptor_set_layout_count(ci, &num_bindings, &immutable_sampler_count);

   /* binding[] is a flexible array member and the sampler slots follow it, one allocation the way
    * the driver's multizalloc lays it out, so freeing the layout frees the slots with it. */
   const size_t bindings_offset = sizeof(struct nvk_descriptor_set_layout);
   const size_t samplers_offset =
      bindings_offset + num_bindings * sizeof(struct nvk_descriptor_set_binding_layout);
   const size_t layout_size =
      samplers_offset + immutable_sampler_count * sizeof(struct nvk_sampler *);

   struct nvk_descriptor_set_layout *layout = calloc(1, layout_size);

   if (!layout) {
      if (message)
         *message = strdup("out of memory building a descriptor set layout");
      return NULL;
   }

   layout->vk.ref_cnt = 1;
   layout->vk.flags = ci->flags;

   struct nvk_sampler **samplers = immutable_sampler_count > 0 ?
      (struct nvk_sampler **)((char *)layout + samplers_offset) : NULL;

   nvk_descriptor_set_layout_init(pdev, ci, layout, layout->binding, samplers,
                                  num_bindings, immutable_sampler_count);

   free(local_bindings);
   free(stand_ins);

   return layout;
}

s2i_result
s2i_nvk_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                int target_index,
                const s2i_pipeline *pipeline, char **isa_text, s2i_stats_nvidia *stats,
                s2i_info *info, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;

   if (!s2i_module_args_ok(spirv, spirv_words, entry, stage) || target_index < 0 ||
       target_index >= s2i_nvk_target_count())
      return S2I_BAD_SPIRV;

   /* This tree's NVK exposes no ray extension or feature and NAK has no RT stage, so an RT stage is
    * refused by name rather than lowered into something the driver could not have produced. */
   const mesa_shader_stage ms = s2i_stage_to_mesa(stage);

   if (s2i_stage_is_rt(ms)) {
      if (message)
         *message = strdup("ray tracing stages are not wired on the NVIDIA backend yet");
      return S2I_UNSUPPORTED_CAP;
   }

   const struct s2i_nvk_target_desc *target = &s2i_nvk_targets[target_index];

   struct s2i_session session;

   if (!s2i_session_open(&session))
      return S2I_COMPILE_FAILED;

   /* Everything the one teardown below releases, declared before the first jump to it. */
   s2i_result result = S2I_OK;
   struct nak_shader_bin *bin = NULL;
   struct vk_descriptor_set_layout *layouts[NVK_MAX_SETS] = { 0 };
   uint32_t layout_count = 0;
   struct nvk_sampler *stand_in_sampler = NULL;

   /* NVK reads this for the address forms, NAK again as a robust2 mode mask at compile. */
   const struct vk_pipeline_robustness_state rs = s2i_robustness_state(pipeline->robustness);

   /* The synthesized device world: info from the target table, NAK from the info, the instance
    * carrying NVK's own driconf defaults, and the supported tables from NVK's own feature code. */
   struct nvk_instance *instance = rzalloc(session.mem_ctx, struct nvk_instance);

   /* The runtime's __vk_log_impl fires on ANY vtn diagnostic (a wrong entry name is enough) and
    * resolves the instance by walking typed object bases, then reads these lists and mutexes; on
    * zeroed state that walk dereferences the base's NULL device pointer. Typed, client-visible
    * bases (here and on the physical device and device below) plus empty lists make every
    * diagnostic fall through to mesa's own logging instead. */
   instance->vk.base.type = VK_OBJECT_TYPE_INSTANCE;
   instance->vk.base.client_visible = true;
   list_inithead(&instance->vk.debug_report.callbacks);
   mtx_init(&instance->vk.debug_report.callbacks_mutex, mtx_plain);
   list_inithead(&instance->vk.debug_utils.instance_callbacks);
   list_inithead(&instance->vk.debug_utils.callbacks);
   mtx_init(&instance->vk.debug_utils.callbacks_mutex, mtx_plain);
   nvk_drirc_defaults(&instance->drirc);
   instance->vk.app_info.api_version = VK_MAKE_API_VERSION(0, 1, 4, 0);

   struct nvk_physical_device *pdev = rzalloc(session.mem_ctx, struct nvk_physical_device);

   pdev->vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   pdev->vk.base.client_visible = true;
   /* A row whose shader model the class ladder does not name would otherwise compile against
    * whatever zeroed classes happen to mean, so it is refused here instead. */
   if (!s2i_nvk_fill_device_info(target, &pdev->info)) {

      if (message) {
         char buf[128];

         snprintf(buf, sizeof(buf), "%s has no 3D and compute class mapping for shader model %u",
                  target->name, pdev->info.sm);
         *message = strdup(buf);
      }

      result = S2I_BAD_TARGET;
      goto done;
   }

   pdev->vk.instance = &instance->vk;

   pdev->nak = nak_compiler_create(&pdev->info);

   if (!pdev->nak) {

      if (message)
         *message = strdup("nak_compiler_create failed");

      result = S2I_BAD_TARGET;
      goto done;
   }

   nvk_physical_device_offline_supported(instance, &pdev->info, &pdev->vk.supported_extensions,
                                         &pdev->vk.supported_features, &pdev->vk.properties);

   struct nvk_device *dev = rzalloc(session.mem_ctx, struct nvk_device);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.base.client_visible = true;
   /* The logger resolves any owned object through base.device, a device's own base included. */
   dev->vk.base.device = &dev->vk;
   dev->vk.physical = &pdev->vk;

   const struct spirv_capabilities target_caps =
      vk_physical_device_get_spirv_capabilities(&pdev->vk);

   {
      bool malformed = false;
      char *refusal = s2i_gate_declared_capabilities(spirv, spirv_words, &target_caps, target->name,
                                                     &malformed);

      if (refusal) {

         if (message)
            *message = refusal;
         else
            free(refusal);

         result = malformed ? S2I_BAD_SPIRV : S2I_UNSUPPORTED_CAP;
         goto done;
      }
   }

   struct spirv_to_nir_options spirv_opts =
      nvk_shader_offline_spirv_options(&pdev->vk, ms, &rs);
   const struct nir_shader_compiler_options *nir_options =
      nvk_shader_offline_nir_options(&pdev->vk, ms, &rs);

   nir_shader *nir = vk_spirv_to_nir(&dev->vk, spirv, spirv_words * 4, ms, entry, NULL,
                                     &spirv_opts, nir_options, false, session.mem_ctx);

   if (!nir) {
      if (message)
         *message = strdup("vk_spirv_to_nir failed (bad entrypoint, or a capability this target "
                           "does not support; diagnostics on stderr)");
      result = S2I_NO_ENTRYPOINT;
      goto done;
   }

   /* The driver's preprocess, which wraps NAK's: sysvals to varyings and input attachments are
    * NVK's own steps around it. */
   nvk_shader_offline_preprocess(&pdev->vk, nir, &rs);

   const VkDescriptorSetLayoutCreateInfo *const *eff_layouts = NULL;
   uint32_t eff_layout_count = 0;

   result = s2i_resolve_layouts(&nir, 1, NVK_MAX_SETS, pipeline->set_layouts,
                                pipeline->set_layout_count, session.mem_ctx, &eff_layouts,
                                &eff_layout_count, message);

   if (result != S2I_OK)
      goto done;

   bool layouts_ok = true;

   for (uint32_t set = 0; set < eff_layout_count; set++) {

      const VkDescriptorSetLayoutCreateInfo empty = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      };

      const VkDescriptorSetLayoutCreateInfo *ci = eff_layouts[set] ? eff_layouts[set] : &empty;

      struct nvk_descriptor_set_layout *layout =
         s2i_nvk_build_one_layout(pdev, ci, &stand_in_sampler, message);

      if (!layout) {
         layouts_ok = false;
         break;
      }

      layouts[set] = &layout->vk;
      layout_count = set + 1;
   }

   if (!layouts_ok) {
      result = S2I_UNSUPPORTED_CAP;
      goto done;
   }

   /* NVK's own lowering, in NVK's own order, then NAK.
    * TODO: shader_flags is a declared input this API does not carry yet. Zero reaches descriptor
    * and mesh lowering as "not indirect bindable, task shader present", which is the plain
    * pipeline default; it is wrong only for an indirect-bindable or task-less mesh shader. */
   struct nvk_cbuf_map cbuf_map;
   memset(&cbuf_map, 0, sizeof(cbuf_map));

   nvk_shader_offline_lower(dev, nir, 0 /* shader_flags */, &rs, layout_count, layouts,
                            &cbuf_map);

   /* NVK's own fragment key. With no graphics state it turns conservative-raster underestimation
    * on, which a zeroed key would not. */
   struct vk_graphics_pipeline_all_state gfx_all;
   struct vk_graphics_pipeline_state gfx_state;
   const struct vk_graphics_pipeline_state *gfx = NULL;

   if (pipeline->graphics) {

      if (!s2i_graphics_state(&dev->vk, pipeline->graphics, &gfx_all, &gfx_state, message)) {
         result = S2I_UNSUPPORTED_CAP;
         goto done;
      }

      gfx = &gfx_state;
   }

   if (info)
      info->graphics_specialized = gfx != NULL;

   struct nak_fs_key fs_key;
   nvk_shader_offline_fs_key(&fs_key, gfx);

   /* Only the ROBUST_BUFFER_ACCESS_2 forms reach NAK, the same mask nvk_shader.c derives.
    * has_task_shader is read only for a mesh shader (NAK's init_info_from_nir), and true is the
    * driver's default; it becomes real caller data with shader_flags above. */
   nir_variable_mode robust2_modes = 0;

   if (rs.uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2)
      robust2_modes |= nir_var_mem_ubo;

   if (rs.storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2)
      robust2_modes |= nir_var_mem_ssbo;

   bin = nak_compile_shader(nir, isa_text != NULL, pdev->nak, robust2_modes,
                            ms == MESA_SHADER_FRAGMENT ? &fs_key : NULL,
                            true /* has_task_shader */);

   if (!bin) {
      if (message)
         *message = strdup("nak_compile_shader failed");
      result = S2I_COMPILE_FAILED;
      goto done;
   }

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->gprs = bin->info.num_gprs;
      stats->instrs = bin->info.num_instrs;
      stats->static_cycles = bin->info.num_static_cycles;
      stats->spills_to_mem = bin->info.num_spills_to_mem;
      stats->fills_from_mem = bin->info.num_fills_from_mem;
      stats->spills_to_reg = bin->info.num_spills_to_reg;
      stats->fills_from_reg = bin->info.num_fills_from_reg;
      stats->slm_size = bin->info.slm_size;
      stats->crs_size = bin->info.crs_size;
      stats->max_warps_per_sm = bin->info.max_warps_per_sm;
      stats->code_size = bin->code_size;

      if (ms == MESA_SHADER_COMPUTE)
         stats->smem_size = bin->info.cs.smem_size;
      else if (ms == MESA_SHADER_MESH)
         stats->smem_size = bin->info.mesh.smem_size;
      else if (ms == MESA_SHADER_TASK)
         stats->smem_size = bin->info.task.smem_size;
   }

   if (isa_text && bin->asm_str)
      *isa_text = strdup(bin->asm_str);

   /* The one way out, success included. Nothing below is set until the step that produces it ran.
    */
done:

   if (bin)
      nak_shader_bin_destroy(bin);

   for (uint32_t set = 0; set < layout_count; set++)
      free(layouts[set]);

   free(stand_in_sampler);

   if (pdev->nak)
      nak_compiler_destroy(pdev->nak);

   s2i_session_close(&session);
   return result;
}
