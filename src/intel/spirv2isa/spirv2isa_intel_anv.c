/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Builds the inputs ANV's NIR lowering reads, with no device. See spirv2isa_intel_anv.h.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#include "spirv2isa_intel_anv.h"
#include "spirv2isa_desc.h"

#include "util/ralloc.h"
#include "anv_api_version.h"
#include "anv_drirc.h"
#include "vk_log.h"
#include "anv_private.h"
#include "isl/isl.h"
#include "dev/intel_device_info.h"

static_assert(S2I_INTEL_ANV_MAX_SETS == MAX_SETS,
              "spirv2isa's set limit must be ANV's, since the passes index layouts by set number");

/* ANV reads this to let ANV_DEBUG=bindless force descriptors bindless. It is defined here and never
 * parsed, so an environment variable cannot silently change the ISA this tool reports. Defining it
 * here also keeps anv_instance.c, and the whole instance and entrypoint world, out of the link. */
enum anv_debug anv_debug;

/* vk_nir logs SPIR-V diagnostics through the instance's debug messengers, which an offline compile
 * has none of; stderr is where a command line tool's diagnostics belong. */
void
__vk_log_impl(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types,
              int object_count, const void **objects_or_instance, const char *file, int line,
              const char *format, ...)
{
   (void)severity; (void)types; (void)object_count; (void)objects_or_instance;

   va_list args;
   va_start(args, format);
   fprintf(stderr, "%s:%d: ", file, line);
   vfprintf(stderr, format, args);
   fputc('\n', stderr);
   va_end(args);
}

/* ANV builds a softfp64 helper library on the device when a shader wants doubles on hardware without
 * them. Its only caller is gated behind drirc.debug.fp64_emu, which a synthesized instance leaves off,
 * so this exists to close the link and is never entered; returning NULL rather than inventing a shader
 * keeps it that way. */
nir_shader *
anv_ensure_fp64_shader(struct anv_device *device)
{
   (void)device;
   return NULL;
}

void
s2i_intel_anv_init_physical_device(struct anv_physical_device *pdev,
                                   const struct intel_device_info *devinfo,
                                   struct brw_compiler *compiler,
                                   void *mem_ctx)
{
   memset(pdev, 0, sizeof(*pdev));

   /* The lowering reads driconf options through the instance, and several of them default to true or
    * to a non-zero number: the vertex payload budget, component packing, the spilling rate, the active
    * thread barrier emulation. A zeroed instance silently answers false or zero to all of them, so the
    * options are parsed the way ANV parses them, which is also how the defaults stay in one place. */
   pdev->instance = rzalloc(mem_ctx, struct anv_instance);

   anv_drirc_defaults(&pdev->instance->drirc);

   /* The generated SPIR-V capability mapping reads the API version off the instance and the
    * supported tables off the physical device; both are what ANV itself would report. */
   pdev->instance->vk.app_info.api_version = ANV_API_VERSION;
   pdev->vk.instance = &pdev->instance->vk;

   pdev->info = *devinfo;
   pdev->compiler = compiler;

   isl_device_init(&pdev->isl_dev, &pdev->info);

   /* Mirrors anv_physical_device.c. The first two are facts about the hardware, so a PCI id decides
    * them exactly as a real device would. */
   pdev->isl_dev.buffer_length_in_aux_addr = !intel_needs_workaround(devinfo, 14019708328);
   pdev->indirect_descriptors = !intel_has_extended_bindless(devinfo);

   /* ANV's own expression (anv_physical_device.c). The kernel mode driver is the one input a PCI id
    * cannot carry, and INTEL_KMD_TYPE_INVALID is what an offline device reports, so this lands on the
    * same answer ANV gives an application whose engine name does not appear in 00-anv-defaults.conf,
    * which is every engine but one. */
   const bool platform_supports_btp_bit_rcc =
      devinfo->has_lsc &&
      (devinfo->kmd_type == INTEL_KMD_TYPE_I915 || devinfo->xe_has_state_cache_perf_fix);

   pdev->rt_change_needs_flush =
      !pdev->instance->drirc.perf.state_cache_perf_fix || !platform_supports_btp_bit_rcc;

   /* Scratch ids are a kernel query on the device path and stay zero here, which silently folds the
    * per-invocation ray query stack to a single slot. Derived from the device info instead. */
   intel_device_info_init_max_scratch_ids(&pdev->info);

   /* brw reads both off the compiler, and ANV sets both from driconf. */
   compiler->spilling_rate = pdev->instance->drirc.debug.shader_spilling_rate;
   compiler->limit_trig_input_range = pdev->instance->drirc.debug.limit_trig_input_range;

   anv_physical_device_offline_supported(pdev);
}

static struct anv_descriptor_set_layout *
s2i_intel_anv_build_one(const struct anv_physical_device *pdev,
                        const VkDescriptorSetLayoutCreateInfo *ci,
                        uint32_t dynamic_descriptor_count)
{
   uint32_t num_bindings, immutable_sampler_count;
   anv_descriptor_set_layout_count(ci, &num_bindings, &immutable_sampler_count);

   /* binding[] is a flexible array member, and the samplers follow it, so a layout is one allocation
    * the way vk_descriptor_set_layout_multizalloc makes it one for the driver. Freeing the layout
    * therefore frees the samplers with it. */
   const size_t bindings_size = num_bindings * sizeof(struct anv_descriptor_set_binding_layout);
   const size_t samplers_offset =
      ALIGN_POT(sizeof(struct anv_descriptor_set_layout) + bindings_size,
                alignof(struct anv_descriptor_set_layout_sampler));
   const size_t layout_size =
      samplers_offset + immutable_sampler_count * sizeof(struct anv_descriptor_set_layout_sampler);

   struct anv_descriptor_set_layout *set_layout = calloc(1, layout_size);

   if (!set_layout)
      return NULL;

   struct anv_descriptor_set_layout_sampler *samplers =
      immutable_sampler_count > 0 ?
      (struct anv_descriptor_set_layout_sampler *)((char *)set_layout + samplers_offset) : NULL;

   /* The Vulkan base is normally filled by the runtime while it allocates. Only two of its fields are
    * read from here: flags, which the bindless predicates test and which therefore reaches the ISA,
    * and dynamic_descriptor_count, which feeds the layout hash. */
   set_layout->vk.flags = ci->flags;
   set_layout->vk.dynamic_descriptor_count = dynamic_descriptor_count;
   set_layout->vk.ref_cnt = 1;

   anv_descriptor_set_layout_init(pdev, ci, set_layout, set_layout->binding, samplers,
                                  num_bindings, immutable_sampler_count);

   return set_layout;
}

bool
s2i_intel_anv_build_layouts(const struct anv_physical_device *pdev,
                            const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                            uint32_t set_layout_count,
                            s2i_intel_anv_layouts *out, char **message)
{
   memset(out, 0, sizeof(*out));

   if (set_layout_count > S2I_INTEL_ANV_MAX_SETS) {

      if (message) {
         char buf[128];
         snprintf(buf, sizeof(buf), "%u descriptor sets, and this backend indexes at most %u",
                  set_layout_count, S2I_INTEL_ANV_MAX_SETS);
         *message = strdup(buf);
      }

      return false;
   }

   out->set_count = set_layout_count;

   uint32_t dynamic_total = 0;

   for (uint32_t set = 0; set < set_layout_count; set++) {
      /* A set the pipeline layout leaves empty still needs a layout, because the passes index by set
       * number; an empty create-info is what an unused set means. */
      const VkDescriptorSetLayoutCreateInfo empty = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      };
      const VkDescriptorSetLayoutCreateInfo *ci = set_layouts[set] ? set_layouts[set] : &empty;

      uint32_t dynamic_in_set = 0;

      bool any_immutable = false;

      for (uint32_t b = 0; b < ci->bindingCount; b++) {

         if (ci->pBindings[b].pImmutableSamplers)
            any_immutable = true;

         if (s2i_descriptor_type_is_dynamic(ci->pBindings[b].descriptorType))
            dynamic_in_set += ci->pBindings[b].descriptorCount;
      }

      /* The handles in pImmutableSamplers are live driver objects a caller cannot have here, so every
       * one of them is replaced by a single plain sampler before ANV's arithmetic reads them. The
       * layout math consumes exactly three things off a sampler: the plane count (1 for anything
       * without a ycbcr conversion), the ycbcr conversion (none), and the embedded key (only behind a
       * flag this build refuses). A ycbcr immutable sampler changes plane counts and therefore binding
       * indices, and is NOT expressible offline; a caller using one gets a layout laid out for plane
       * count 1, which is the one divergence this substitution carries. */
      VkDescriptorSetLayoutCreateInfo local_ci;
      VkDescriptorSetLayoutBinding *local_bindings = NULL;
      VkSampler *stand_ins = NULL;

      if (any_immutable) {

         if (!out->stand_in_sampler) {
            out->stand_in_sampler = calloc(1, sizeof(*out->stand_in_sampler));

            if (out->stand_in_sampler)
               out->stand_in_sampler->state.n_planes = 1;
         }

         uint32_t max_count = 0;

         for (uint32_t b = 0; b < ci->bindingCount; b++)
            max_count = MAX2(max_count, ci->pBindings[b].descriptorCount);

         local_bindings = calloc(ci->bindingCount, sizeof(*local_bindings));
         stand_ins = calloc(max_count ? max_count : 1, sizeof(*stand_ins));

         if (!out->stand_in_sampler || !local_bindings || !stand_ins) {

            if (message)
               *message = strdup("out of memory building a descriptor set layout");

            free(local_bindings);
            free(stand_ins);
            s2i_intel_anv_free_layouts(out);
            return false;
         }

         for (uint32_t i = 0; i < (max_count ? max_count : 1); i++)
            stand_ins[i] = anv_sampler_to_handle(out->stand_in_sampler);

         for (uint32_t b = 0; b < ci->bindingCount; b++) {
            local_bindings[b] = ci->pBindings[b];

            if (local_bindings[b].pImmutableSamplers)
               local_bindings[b].pImmutableSamplers = stand_ins;
         }

         local_ci = *ci;
         local_ci.pBindings = local_bindings;
         ci = &local_ci;
      }

      /* Dynamic offsets are numbered across the whole pipeline layout, so a set starts where the
       * previous one ended. */
      out->dynamic_offset_start[set] = dynamic_total;
      dynamic_total += dynamic_in_set;

      out->sets[set] = s2i_intel_anv_build_one(pdev, ci, dynamic_in_set);

      free(local_bindings);
      free(stand_ins);

      if (!out->sets[set]) {

         if (message)
            *message = strdup("out of memory building a descriptor set layout");

         s2i_intel_anv_free_layouts(out);
         return false;
      }
   }

   return true;
}

void
s2i_intel_anv_free_layouts(s2i_intel_anv_layouts *layouts)
{
   for (uint32_t set = 0; set < layouts->set_count; set++) {

      if (!layouts->sets[set])
         continue;

      free(layouts->sets[set]);
      layouts->sets[set] = NULL;
   }

   layouts->set_count = 0;

   free(layouts->stand_in_sampler);
   layouts->stand_in_sampler = NULL;
}
