/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Pipeline state a lone SPIR-V module does not carry, shared by every backend, INTERNAL.
 *
 * Robustness decides whether buffer and image accesses are bounds checked, which address format
 * their descriptors take, and which loads may be vectorized. Graphics state decides the sample and
 * tessellation keys, the view mask and the colour attachment map. Both move the ISA, and a driver
 * reads both off the pipeline's create info; so does this.
 */
#ifndef SPIRV2ISA_STATE_H
#define SPIRV2ISA_STATE_H

#include <stdbool.h>
#include <string.h>

#include <vulkan/vulkan_core.h>

#include "vk_graphics_state.h"
#include "vk_pipeline.h"

/* What the caller declared, in the runtime's own form. No create info means DISABLED, not
 * DEVICE_DEFAULT: there is no device here whose default it could mean. The null descriptor flags
 * follow VK_EXT_robustness2's feature rather than this create info, so they stay false. */
static inline struct vk_pipeline_robustness_state
s2i_robustness_state(const VkPipelineRobustnessCreateInfo *ci)
{
   struct vk_pipeline_robustness_state rs = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED,
   };

   if (ci) {
      rs.storage_buffers = ci->storageBuffers;
      rs.uniform_buffers = ci->uniformBuffers;
      rs.vertex_inputs = ci->vertexInputs;
      rs.images = ci->images;
   }

   return rs;
}

/* The graphics state the stage compiles against, in the runtime's own form. `all` is the caller's
 * storage: with it every sub-state points into `all`, so the fill allocates nothing and needs no
 * allocator. False means the create info is not one this can fill, which the caller refuses by
 * name. */
static inline bool
s2i_graphics_state(const struct vk_device *device, const VkGraphicsPipelineCreateInfo *ci,
                   struct vk_graphics_pipeline_all_state *all,
                   struct vk_graphics_pipeline_state *state, char **message)
{
   memset(state, 0, sizeof(*state));

   /* The fill asserts that a pipeline carrying pre-rasterization state names a vertex or mesh
    * stage, so a create info listing only the stage being compiled would abort inside the runtime.
    */
   VkShaderStageFlags stages = 0;

   for (uint32_t i = 0; i < ci->stageCount; i++)
      stages |= ci->pStages[i].stage;

   if (!(stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT))) {

      if (message)
         *message = strdup("the graphics pipeline state names no vertex or mesh stage: it "
                           "describes the whole pipeline, not just the stage being compiled");

      return false;
   }

   return vk_graphics_pipeline_state_fill(device, state, ci, NULL /* driver_mv */,
                                          NULL /* driver_rp */, 0, all, NULL /* alloc */, 0,
                                          NULL /* alloc_ptr_out */) == VK_SUCCESS;
}

#endif /* SPIRV2ISA_STATE_H */
