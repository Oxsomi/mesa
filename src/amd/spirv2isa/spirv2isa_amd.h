/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa AMD backend (ACO via RADV), INTERNAL.
 *
 * The public API is src/spirv2isa.h; this is what the dispatcher there calls. A target arrives as a
 * vendor local index (0 .. S2I_TARGET_AMD_COUNT-1), already unpacked from the public s2i_target, so
 * this backend keeps numbering its own architectures from 0 and never has to know the vendor axis.
 *
 * spirv2isa.c is folded into RADV's own compilation so it can call the internal device-free radv_*
 * compile functions (see the meson.build next to it).
 */
#ifndef SPIRV2ISA_AMD_H
#define SPIRV2ISA_AMD_H

#include "spirv2isa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* s2i_compile for an AMD target, taking AMD's own stats. See s2i_compile for the arguments. */
s2i_result s2i_amd_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                           int target_index, const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                             uint32_t set_layout_count,
                           s2i_features features_used, char **isa_text, s2i_stats_amd *stats,
                           s2i_info *info, char **message);

/* s2i_compile_rt_pipeline for an AMD target. See s2i_compile_rt_pipeline for the arguments. */
s2i_result s2i_amd_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count, size_t entry_index,
                                       int compile_traversal, int target_index, const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                             uint32_t set_layout_count, s2i_features features_used, char **isa_text,
                                       s2i_stats_amd *stats, char **message);

/* The capabilities this target doesn't support, newline separated and malloc'd, or NULL for none. */
char *s2i_amd_unsupported_caps(const uint32_t *caps_used, size_t caps_count, int target_index);

/* Human-readable name of an AMD target, "" when the index isn't one. */
const char *s2i_amd_target_name(int target_index);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_AMD_H */
