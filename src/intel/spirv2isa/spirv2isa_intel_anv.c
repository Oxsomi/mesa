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

#include "anv_private.h"
#include "isl/isl.h"
#include "dev/intel_device_info.h"

static_assert(S2I_INTEL_ANV_MAX_SETS == MAX_SETS,
              "spirv2isa's set limit must be ANV's, since the passes index layouts by set number");

/* ANV reads this to let ANV_DEBUG=bindless force descriptors bindless. It is defined here and never
 * parsed, so an environment variable cannot silently change the ISA this tool reports. Defining it
 * here also keeps anv_instance.c, and the whole instance and entrypoint world, out of the link. */
enum anv_debug anv_debug;

void
s2i_intel_anv_init_physical_device(struct anv_physical_device *pdev,
                                   const struct intel_device_info *devinfo,
                                   struct brw_compiler *compiler)
{
   memset(pdev, 0, sizeof(*pdev));

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
                        const VkDescriptorSetLayoutBinding *vk_bindings, uint32_t vk_binding_count,
                        uint32_t dynamic_descriptor_count)
{
   const VkDescriptorSetLayoutCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = vk_binding_count,
      .pBindings = vk_bindings,
   };

   uint32_t num_bindings, immutable_sampler_count;
   anv_descriptor_set_layout_count(&create_info, &num_bindings, &immutable_sampler_count);

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
   set_layout->vk.flags = create_info.flags;
   set_layout->vk.dynamic_descriptor_count = dynamic_descriptor_count;
   set_layout->vk.ref_cnt = 1;

   anv_descriptor_set_layout_init(pdev, &create_info, set_layout, set_layout->binding, samplers,
                                  num_bindings, immutable_sampler_count);

   return set_layout;
}

bool
s2i_intel_anv_build_layouts(const struct anv_physical_device *pdev,
                            const s2i_binding *bindings, size_t binding_count,
                            s2i_intel_anv_layouts *out, char **message)
{
   memset(out, 0, sizeof(*out));

   uint32_t highest_set = 0;

   for (size_t i = 0; i < binding_count; i++) {

      if (bindings[i].set >= S2I_INTEL_ANV_MAX_SETS || bindings[i].type >= S2I_DESC_TYPE_COUNT) {

         if (message) {
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "descriptor set %u binding %u has no ANV equivalent (set < %u, type < %u)",
                     bindings[i].set, bindings[i].binding, S2I_INTEL_ANV_MAX_SETS,
                     (unsigned)S2I_DESC_TYPE_COUNT);
            *message = strdup(buf);
         }

         return false;
      }

      if (bindings[i].set > highest_set)
         highest_set = bindings[i].set;
   }

   /* The passes index set_layouts[] by descriptor set number, so a gap still needs a layout; an empty
    * one is what a pipeline layout with an unused set gives them. */
   out->set_count = binding_count > 0 ? highest_set + 1 : 0;

   uint32_t dynamic_total = 0;

   /* At most every supplied binding lands in one set, so one array of that size serves every set. */
   VkDescriptorSetLayoutBinding *vk_bindings =
      binding_count > 0 ? calloc(binding_count, sizeof(*vk_bindings)) : NULL;

   if (binding_count > 0 && !vk_bindings) {

      if (message)
         *message = strdup("out of memory building a descriptor set layout");

      return false;
   }

   for (uint32_t set = 0; set < out->set_count; set++) {
      uint32_t vk_binding_count = 0;
      uint32_t dynamic_in_set = 0;

      for (size_t i = 0; i < binding_count; i++) {

         if (bindings[i].set != set)
            continue;

         const uint32_t count = bindings[i].count ? bindings[i].count : 1;

         vk_bindings[vk_binding_count++] = (VkDescriptorSetLayoutBinding) {
            .binding = bindings[i].binding,
            .descriptorType = s2i_descriptor_type_to_vk(bindings[i].type),
            .descriptorCount = count,
            .stageFlags = VK_SHADER_STAGE_ALL,
         };

         if (s2i_descriptor_type_is_dynamic(bindings[i].type))
            dynamic_in_set += count;
      }

      /* Dynamic offsets are numbered across the whole pipeline layout, so each set starts where the
       * previous one ended. */
      out->dynamic_offset_start[set] = dynamic_total;
      dynamic_total += dynamic_in_set;

      out->sets[set] = s2i_intel_anv_build_one(pdev, vk_bindings, vk_binding_count, dynamic_in_set);

      if (!out->sets[set]) {

         if (message)
            *message = strdup("out of memory building a descriptor set layout");

         free(vk_bindings);
         s2i_intel_anv_free_layouts(out);
         return false;
      }
   }

   free(vk_bindings);
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
