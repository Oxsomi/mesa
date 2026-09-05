/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa Intel backend (EU ISA via brw), INTERNAL.
 *
 * The public API is src/spirv2isa.h; this is what the dispatcher there calls. A target arrives as a
 * vendor local index (0 .. S2I_TARGET_INTEL_COUNT-1), already unpacked from the public s2i_target.
 * Intel's compiler is a pure function of an intel_device_info (built from the target's PCI id,
 * device-free) plus NIR: brw_compiler_create(devinfo) then brw_compile().
 *
 * This is the first slice: compute. Buffers, images and samplers are placed from the module's own
 * set/binding decorations; anything resource shaped that no pass here claims is refused by name
 * rather than reaching brw. Graphics and RT stages follow, mirroring the AMD backend.
 */
#ifndef SPIRV2ISA_INTEL_H
#define SPIRV2ISA_INTEL_H

#include "spirv2isa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* s2i_compile for an Intel target, taking Intel's own stats. See s2i_compile for the arguments.
 * `bindings` is validated, not consumed: placement comes from the module's own decorations, so a
 * binding this backend cannot flatten is S2I_UNSUPPORTED_CAP instead of ISA that cannot bind. */
s2i_result s2i_intel_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                             int target_index, const s2i_binding *bindings, size_t binding_count,
                             s2i_features features_used, char **isa_text, s2i_stats_intel *stats,
                             s2i_info *info, char **message);

/* Human-readable name of an Intel target, "" when the index isn't one. */
const char *s2i_intel_target_name(int target_index);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_INTEL_H */
