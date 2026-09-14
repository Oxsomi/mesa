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
 * Descriptors are not lowered here: this links ANV's own passes and supplies the layout they read,
 * so the binding tables and bindless handles are the ones the driver would build rather than a second
 * implementation of the same rules. Anything that still reaches brw unlowered is refused by name.
 */
#ifndef SPIRV2ISA_INTEL_H
#define SPIRV2ISA_INTEL_H

#include "spirv2isa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* s2i_compile for an Intel target, taking Intel's own stats. See s2i_compile for the arguments.
 * `bindings` is the layout every resource is placed from, so a module that uses a descriptor the
 * caller did not describe cannot be compiled. */
s2i_result s2i_intel_compile(const uint32_t *spirv, size_t spirv_words, const char *entry,
                             s2i_stage stage, int target_index, const s2i_pipeline *pipeline,
                             char **isa_text, s2i_stats_intel *stats, s2i_info *info,
                             char **message);

/* Human-readable name of an Intel target, "" when the index isn't one. */
const char *s2i_intel_target_name(int target_index);

/* Total targets across both tiers, flagships first: the enum's constants, then every other Gen9+
 * PCI id Mesa's device database knows (minus preproduction force-probe ids), enumerated at
 * runtime. */
int s2i_intel_target_count(void);

/* Token of an extended-tier target, "" for a flagship index (whose token the dispatcher owns) and
 * for out of range. */
const char *s2i_intel_target_token(int target_index);

#ifdef __cplusplus
}
#endif

#endif /* SPIRV2ISA_INTEL_H */
