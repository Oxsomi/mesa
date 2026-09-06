/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Builds the inputs ANV's NIR lowering reads, with no device. See spirv2isa_intel_anv.h.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "spirv2isa_intel_anv.h"
#include "spirv2isa_desc.h"

#include "util/ralloc.h"
#include "anv_private.h"
#include "isl/isl.h"
#include "dev/intel_device_info.h"

static_assert(S2I_INTEL_ANV_MAX_SETS == MAX_SETS,
              "spirv2isa's set limit must be ANV's, since the passes index layouts by set number");

/* ANV reads this to let ANV_DEBUG=bindless force descriptors bindless. It is defined here and never
 * parsed, so an environment variable cannot silently change the ISA this tool reports. Defining it
 * here also keeps anv_instance.c, and the whole instance and entrypoint world, out of the link. */
enum anv_debug anv_debug;

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

   /* The lowering reads one drirc option through the instance, so there has to be one. Zeroed means
    * every debug option is off, which is the only honest answer offline: they exist to let a user
    * perturb a driver, and there is no user here. */
   pdev->instance = rzalloc(mem_ctx, struct anv_instance);

   pdev->info = *devinfo;
   pdev->compiler = compiler;

   isl_device_init(&pdev->isl_dev, &pdev->info);

   /* Mirrors anv_physical_device.c. The first two are facts about the hardware, so a PCI id decides
    * them exactly as a real device would. */
   pdev->isl_dev.buffer_length_in_aux_addr = !intel_needs_workaround(devinfo, 14019708328);
   pdev->indirect_descriptors = !intel_has_extended_bindless(devinfo);

   /* This one is not: ANV derives it from the kernel mode driver, which no PCI id carries, and it
    * selects between binding-table and fully bindless codegen. This matches ANV on a kernel with the
    * state-cache fix, which is what 00-anv-defaults.conf assumes, and is reported to the caller
    * rather than left as a default nobody can see. */
   pdev->rt_change_needs_flush = !devinfo->has_lsc;
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

      for (uint32_t b = 0; b < ci->bindingCount; b++) {

         /* A VkSampler is a live driver object, and there is no driver here. A ycbcr immutable sampler
          * changes plane counts and therefore binding indices, so this refuses rather than lays the set
          * out as if the sampler were not there. */
         if (ci->pBindings[b].pImmutableSamplers) {

            if (message) {
               char buf[160];
               snprintf(buf, sizeof(buf),
                        "set %u binding %u has immutable samplers, which are driver objects an offline "
                        "compile cannot resolve", set, ci->pBindings[b].binding);
               *message = strdup(buf);
            }

            s2i_intel_anv_free_layouts(out);
            return false;
         }

         if (s2i_descriptor_type_is_dynamic(ci->pBindings[b].descriptorType))
            dynamic_in_set += ci->pBindings[b].descriptorCount;
      }

      /* Dynamic offsets are numbered across the whole pipeline layout, so a set starts where the
       * previous one ended. */
      out->dynamic_offset_start[set] = dynamic_total;
      dynamic_total += dynamic_in_set;

      out->sets[set] = s2i_intel_anv_build_one(pdev, ci, dynamic_in_set);

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
}
