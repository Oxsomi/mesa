/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa NVIDIA backend (SASS via NVK/NAK), INTERNAL.
 *
 * The public API is src/spirv2isa.h; this is what the dispatcher there calls. A target arrives as a
 * vendor local index (0 .. s2i_nvk_target_count()-1, the enum's flagships as the prefix), already
 * unpacked from the public s2i_target.
 * NAK is a pure function of an nv_device_info (built from the target's chipset, device-free) plus
 * NIR; NVK's own option hooks, descriptor layout arithmetic and NIR lowering run in between, the way
 * the Intel backend runs ANV's.
 *
 * The target list's floor is Maxwell, a scope choice; NAK itself encodes back to Fermi.
 */
#ifndef SPIRV2ISA_NVK_H
#define SPIRV2ISA_NVK_H

#include "spirv2isa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* s2i_compile for an NVIDIA target, taking NVIDIA's own stats. See s2i_compile for the arguments. */
s2i_result s2i_nvk_compile(const uint32_t *spirv, size_t spirv_words, const char *entry,
                           s2i_stage stage, int target_index, const s2i_pipeline *pipeline,
                           char **isa_text, s2i_stats_nvidia *stats, s2i_info *info,
                           char **message);

/* Human-readable name of an NVIDIA target, "" when the index isn't one. */
const char *s2i_nvk_target_name(int target_index);

/* Total targets across both tiers, flagships first, matching the table in spirv2isa_nvk.c. */
int s2i_nvk_target_count(void);

/* Token of an extended-tier target, "" for a flagship index (whose token the dispatcher owns) and
 * for out of range. */
const char *s2i_nvk_target_token(int target_index);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_NVK_H */
