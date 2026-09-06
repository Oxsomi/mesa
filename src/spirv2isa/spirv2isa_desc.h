/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * The descriptor question every backend needs the same answer to, INTERNAL.
 */
#ifndef SPIRV2ISA_DESC_H
#define SPIRV2ISA_DESC_H

#include <stdbool.h>

#include <vulkan/vulkan_core.h>

/* A dynamic buffer takes its offset at bind time, so it consumes a dynamic offset slot, and every
 * backend has to count those to number the slots across a pipeline layout. */
static inline bool
s2i_descriptor_type_is_dynamic(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

#endif /* SPIRV2ISA_DESC_H */
