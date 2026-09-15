/* Copyright 2026 Oxsomi / Nielsbishere SPDX-License-Identifier: MIT The descriptor layout the
 * shaders of one pipeline are compiled against, shared by every backend, INTERNAL. Two jobs, one
 * walk: supplied layouts are checked against the modules, so a binding the caller did not
 * describe is refused by name rather than corrupting a lowering two steps later, and no layouts
 * at all synthesizes one from them, which is the fallback the public API promises. Both the
 * resource variables and the vulkan_resource_index intrinsics count. The intrinsic is all that
 * survives for a descriptor whose variable a preprocess already consumed (ray query does this to
 * the acceleration structure), and the driver lowering reads the intrinsics, so a variables-only
 * walk would miss exactly the entries that lowering asks for. The vendors differ only in where
 * they cap sets, which is why this is one implementation. What each does with the result stays
 * its own. */
#ifndef SPIRV2ISA_LAYOUT_H
#define SPIRV2ISA_LAYOUT_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vulkan/vulkan_core.h>

#include "compiler/glsl_types.h"
#include "nir.h"
#include "util/macros.h"
#include "util/ralloc.h"

#include "spirv2isa.h"

/* Every vendor here caps descriptor sets at 32, so the bookkeeping below is sized once and each
 * backend passes its own constant as the limit it enforces. */
#define S2I_MAX_SETS 32

/* The variable modes a descriptor can live in. */
#define S2I_RESOURCE_MODES (nir_var_uniform | nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_image)

/* The descriptor type a resource variable would need in a layout. The inference is only for the
 * synthesized fallback; a caller-supplied layout is taken at its word. */
static inline VkDescriptorType
s2i_var_descriptor_type(const nir_variable *var)
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

/* One descriptor reference, either recorded for the synthesis or checked against a supplied
 * layout. */
static inline bool
s2i_note_descriptor(const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                    uint32_t set_layout_count, uint32_t max_sets, bool synthesize, uint32_t set,
                    uint32_t binding, uint32_t count, uint32_t *nbind, uint32_t *highest_set,
                    bool *any, char **message)
{
   if (set >= max_sets) {

      if (message) {
         char buf[128];
         snprintf(buf, sizeof(buf), "the module uses descriptor set %u; sets go up to %u", set,
                  max_sets - 1);
         *message = strdup(buf);
      }

      return false;
   }

   *any = true;
   *highest_set = MAX2(*highest_set, set);

   /* One slot per reference rather than one per binding number: a binding is a sparse uint32
    * namespace, so sizing by the highest one would allocate for every number the module skipped.
    * Duplicates merge in the fill below, which makes this an upper bound on what it writes. */
   nbind[set]++;

   if (synthesize)
      return true;

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

   return true;
}

/* One entry per distinct binding, variables first: their array size is richer than the intrinsic's
 * count of 1, and an intrinsic names its own descriptor type where no variable is left to infer
 * one from. */
static inline void
s2i_place_bindings_one(nir_shader *nir, VkDescriptorSetLayoutCreateInfo *infos)
{
   nir_foreach_variable_with_modes(var, nir, S2I_RESOURCE_MODES) {
      VkDescriptorSetLayoutCreateInfo *info = &infos[var->data.descriptor_set];
      VkDescriptorSetLayoutBinding *bindings = (VkDescriptorSetLayoutBinding *) info->pBindings;
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
         .descriptorType = s2i_var_descriptor_type(var),
         .descriptorCount = aoa ? aoa : 1,
         .stageFlags = VK_SHADER_STAGE_ALL,
      };
   }

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            if (intrin->intrinsic != nir_intrinsic_vulkan_resource_index)
               continue;

            VkDescriptorSetLayoutCreateInfo *info = &infos[nir_intrinsic_desc_set(intrin)];
            VkDescriptorSetLayoutBinding *bindings =
               (VkDescriptorSetLayoutBinding *) info->pBindings;
            const uint32_t binding = nir_intrinsic_binding(intrin);
            bool merged = false;

            for (uint32_t b = 0; b < info->bindingCount; b++) {
               if (bindings[b].binding == binding) {
                  merged = true;
                  break;
               }
            }

            if (merged)
               continue;

            bindings[info->bindingCount++] = (VkDescriptorSetLayoutBinding) {
               .binding = binding,
               .descriptorType = (VkDescriptorType) nir_intrinsic_desc_type(intrin),
               .descriptorCount = 1,
               .stageFlags = VK_SHADER_STAGE_ALL,
            };
         }
      }
   }
}

/* The shaders of one pipeline place into one set of infos, so a binding two of them share becomes
 * one entry. */
static inline void
s2i_place_bindings(nir_shader *const *nirs, uint32_t nir_count,
                   VkDescriptorSetLayoutCreateInfo *infos)
{
   for (uint32_t n = 0; n < nir_count; n++)
      s2i_place_bindings_one(nirs[n], infos);
}

/* Every descriptor one shader references, checked against a supplied layout or counted into the
 * synthesis. */
static inline bool
s2i_note_shader(nir_shader *nir, const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                uint32_t set_layout_count, uint32_t max_sets, bool synthesize, uint32_t *nbind,
                uint32_t *highest_set, bool *any, char **message)
{
   nir_foreach_variable_with_modes(var, nir, S2I_RESOURCE_MODES) {
      const uint32_t aoa = glsl_get_aoa_size(var->type);

      if (!s2i_note_descriptor(set_layouts, set_layout_count, max_sets, synthesize,
                               var->data.descriptor_set, var->data.binding, aoa ? aoa : 1, nbind,
                               highest_set, any, message))
         return false;
   }

   /* The intrinsic carries no array size, so its count is 1 and a variable's richer count wins
    * where both describe the same binding. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            if (intrin->intrinsic != nir_intrinsic_vulkan_resource_index)
               continue;

            if (!s2i_note_descriptor(set_layouts, set_layout_count, max_sets, synthesize,
                                     nir_intrinsic_desc_set(intrin), nir_intrinsic_binding(intrin),
                                     1, nbind, highest_set, any, message))
               return false;
         }
      }
   }

   return true;
}

/*
 * Resolve what `nir_count` shaders are compiled against. They resolve together because the shaders
 * of one pipeline share one layout, so a set one of them never touches is still sized by whichever
 * one does. `max_sets` is the caller's own set limit (at most S2I_MAX_SETS). On S2I_OK,
 * *out_layouts / *out_count are either the caller's own layouts unchanged or a synthesized set
 * owned by `mem_ctx`; otherwise *message names what could not be resolved.
 */
static inline s2i_result
s2i_resolve_layouts(nir_shader *const *nirs, uint32_t nir_count, uint32_t max_sets,
                    const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                    uint32_t set_layout_count, void *mem_ctx,
                    const VkDescriptorSetLayoutCreateInfo *const **out_layouts, uint32_t *out_count,
                    char **message)
{
   const bool synthesize = set_layout_count == 0;

   assert(max_sets <= S2I_MAX_SETS);

   if (!synthesize && set_layout_count > max_sets) {

      if (message) {
         char buf[128];
         snprintf(buf, sizeof(buf), "%u descriptor sets, and this backend indexes at most %u",
                  set_layout_count, max_sets);
         *message = strdup(buf);
      }

      return S2I_UNSUPPORTED_CAP;
   }

   uint32_t nbind[S2I_MAX_SETS];
   memset(nbind, 0, sizeof(nbind));

   uint32_t highest_set = 0;
   bool any = false;

   for (uint32_t n = 0; n < nir_count; n++) {
      if (!s2i_note_shader(nirs[n], set_layouts, set_layout_count, max_sets, synthesize, nbind,
                           &highest_set, &any, message))
         return S2I_UNSUPPORTED_CAP;
   }

   if (!synthesize || !any) {
      *out_layouts = set_layouts;
      *out_count = synthesize ? 0 : set_layout_count;
      return S2I_OK;
   }

   const uint32_t count = highest_set + 1;

   VkDescriptorSetLayoutCreateInfo *infos =
      rzalloc_array(mem_ctx, VkDescriptorSetLayoutCreateInfo, count);
   const VkDescriptorSetLayoutCreateInfo **ptrs =
      rzalloc_array(mem_ctx, const VkDescriptorSetLayoutCreateInfo *, count);

   if (!infos || !ptrs) {

      if (message)
         *message = strdup("descriptor layout synthesis ran out of memory");

      return S2I_COMPILE_FAILED;
   }

   for (uint32_t set = 0; set < count; set++) {
      infos[set].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

      if (nbind[set]) {
         infos[set].pBindings = rzalloc_array(mem_ctx, VkDescriptorSetLayoutBinding, nbind[set]);

         if (!infos[set].pBindings) {

            if (message)
               *message = strdup("descriptor layout synthesis ran out of memory");

            return S2I_COMPILE_FAILED;
         }
      }

      ptrs[set] = &infos[set];
   }

   s2i_place_bindings(nirs, nir_count, infos);

   *out_layouts = ptrs;
   *out_count = count;
   return S2I_OK;
}

#endif /* SPIRV2ISA_LAYOUT_H */
