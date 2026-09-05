/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Properties of s2i_descriptor_type that every backend needs the same answer to, INTERNAL.
 *
 * Both backends build their driver's descriptor set layouts from the caller's s2i_binding list, and
 * both need the same two facts to do it: which VkDescriptorType a member means, and whether it is a
 * dynamic buffer. Answering that twice invites the two answers to drift.
 *
 * This is not in spirv2isa.h because the Vulkan mapping would drag vulkan_core.h into every consumer
 * of the public API, which is a plain C header on purpose.
 */
#ifndef SPIRV2ISA_DESC_H
#define SPIRV2ISA_DESC_H

#include <stdbool.h>

#include <vulkan/vulkan_core.h>

#include "spirv2isa.h"

/* s2i_descriptor_type is dense so it can index a table; VkDescriptorType is not, because its last two
 * members are extension values in the billions. Translating is the only correct direction, and a cast
 * reaches an UNREACHABLE inside a driver's descriptor arithmetic. */
static inline VkDescriptorType
s2i_descriptor_type_to_vk(s2i_descriptor_type type)
{
   static const VkDescriptorType to_vk[S2I_DESC_TYPE_COUNT] = {
      [S2I_DESC_SAMPLER]                = VK_DESCRIPTOR_TYPE_SAMPLER,
      [S2I_DESC_COMBINED_IMAGE_SAMPLER] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      [S2I_DESC_SAMPLED_IMAGE]          = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      [S2I_DESC_STORAGE_IMAGE]          = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      [S2I_DESC_UNIFORM_TEXEL_BUFFER]   = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      [S2I_DESC_STORAGE_TEXEL_BUFFER]   = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      [S2I_DESC_UNIFORM_BUFFER]         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      [S2I_DESC_STORAGE_BUFFER]         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      [S2I_DESC_UNIFORM_BUFFER_DYNAMIC] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
      [S2I_DESC_STORAGE_BUFFER_DYNAMIC] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
      [S2I_DESC_INPUT_ATTACHMENT]       = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      [S2I_DESC_INLINE_UNIFORM_BLOCK]   = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
      [S2I_DESC_ACCELERATION_STRUCTURE] = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
   };

   return to_vk[type];
}

/* A dynamic buffer takes its offset at bind time, so it consumes a dynamic offset slot and every
 * backend has to count those to number the slots across a pipeline layout. */
static inline bool
s2i_descriptor_type_is_dynamic(s2i_descriptor_type type)
{
   return type == S2I_DESC_UNIFORM_BUFFER_DYNAMIC || type == S2I_DESC_STORAGE_BUFFER_DYNAMIC;
}

#endif /* SPIRV2ISA_DESC_H */
