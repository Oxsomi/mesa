/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * The inputs ANV's NIR lowering needs, built without a device, INTERNAL.
 *
 * spirv2isa runs ANV's own descriptor and driver-value passes rather than keeping a second copy of
 * rules that would drift from the driver. Those passes read an anv_physical_device and a set of
 * anv_descriptor_set_layouts; this builds both from what the caller supplies (a PCI id and a flat
 * binding list) and owns nothing else. Every rewrite of the shader stays ANV's.
 */
#ifndef SPIRV2ISA_INTEL_ANV_H
#define SPIRV2ISA_INTEL_ANV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "spirv2isa.h"

struct anv_physical_device;
struct anv_descriptor_set_layout;
struct brw_compiler;
struct intel_device_info;

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors ANV's MAX_SETS, which the implementation asserts against so the two cannot drift. Declared
 * here so this header does not drag ANV's own headers into everything that includes it. */
#define S2I_INTEL_ANV_MAX_SETS 32

/* The set layouts for one compile, in set order, so the passes can index them by descriptor set. */
typedef struct s2i_intel_anv_layouts {
   struct anv_descriptor_set_layout *sets[S2I_INTEL_ANV_MAX_SETS];
   uint32_t dynamic_offset_start[S2I_INTEL_ANV_MAX_SETS];
   uint32_t set_count;
} s2i_intel_anv_layouts;

/* Fills a zeroed anv_physical_device with the four things the lowering reads, all derived from the
 * device info the target's PCI id already gave us. `compiler` is the brw_compiler the backend built.
 * Two of the values are policy rather than fact offline; see the implementation. */
void s2i_intel_anv_init_physical_device(struct anv_physical_device *pdev,
                                        const struct intel_device_info *devinfo,
                                        struct brw_compiler *compiler,
                                        void *mem_ctx);

/* Builds one anv_descriptor_set_layout per set touched by `bindings`, using ANV's own layout
 * arithmetic. Returns false and fills `message` when a layout cannot be built offline.
 * Every set gets a layout, including the empty ones, because the passes index by set number.
 * Free with s2i_intel_anv_free_layouts. */
bool s2i_intel_anv_build_layouts(const struct anv_physical_device *pdev,
                                 const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                                 uint32_t set_layout_count,
                                 s2i_intel_anv_layouts *out, char **message);

void s2i_intel_anv_free_layouts(s2i_intel_anv_layouts *layouts);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_INTEL_ANV_H */
