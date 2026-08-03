/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa: standalone SPIR-V -> AMD ISA via Mesa ACO, no VkDevice. See spirv2isa.h.
 *
 * The caller (OxC3) supplies the entrypoint, stage and (for the pre-flight) the capability set - it
 * already has all of this from the oiSH, so this layer never re-scans the SPIR-V. That keeps it thin
 * and non-fragile, and it means every stage OxC3 emits is handled (RADV has per-stage lowering).
 *
 * Flow (all device-free; RADV factored radv_compiler_info out of radv_device precisely for this):
 *   1. spirv_to_nir(spirv, entry, mesa_stage, opts)          -> NIR   (validates caps -> clean errors)
 *   2. build radv_compiler_info(gfx_level, family) + radv_shader_stage + a synthesized layout
 *   3. radv_declare_shader_args + radv_postprocess_nir       -> RADV's exact per-stage lowering
 *   4. aco_compile_shader(..., callback)                     -> ISA config + code + disasm + stats
 */

#include "spirv2isa.h"

#include "util/macros.h"
#include "amd_family.h"            /* enum amd_gfx_level, enum radeon_family */
#include "compiler/shader_enums.h" /* mesa_shader_stage / MESA_SHADER_* */

/* --- target table -------------------------------------------------------------------------- */

struct s2i_target_desc {
   enum amd_gfx_level gfx_level;
   enum radeon_family family;
   const char *name;
};

static const struct s2i_target_desc s2i_targets[S2I_TARGET_COUNT] = {
   [S2I_TARGET_GFX8_POLARIS10] = { GFX8,    CHIP_POLARIS10, "gfx803 (GCN4, RX 580)" },
   [S2I_TARGET_GFX9_VEGA10]    = { GFX9,    CHIP_VEGA10,    "gfx900 (GCN5, RX Vega)" },
   [S2I_TARGET_GFX10_NAVI10]   = { GFX10,   CHIP_NAVI10,    "gfx1010 (RDNA1, RX 5700 XT)" },
   [S2I_TARGET_GFX10_3_NAVI21] = { GFX10_3, CHIP_NAVI21,    "gfx1030 (RDNA2, RX 6800/6700 XT class)" },
   [S2I_TARGET_GFX11_NAVI31]   = { GFX11,   CHIP_NAVI31,    "gfx1100 (RDNA3, RX 7900 XTX)" },
   [S2I_TARGET_GFX12_GFX1201]  = { GFX12,   CHIP_GFX1201,   "gfx1201 (RDNA4, RX 9070 XT)" },
};

const char *
s2i_target_name(s2i_target target)
{
   if (target < 0 || target >= S2I_TARGET_COUNT)
      return "(invalid)";
   return s2i_targets[target].name;
}

/* --- stage mapping (caller's s2i_stage -> Mesa mesa_shader_stage; we never derive it ourselves) --- */

static const mesa_shader_stage s2i_mesa_stage[S2I_STAGE_COUNT] = {
   [S2I_STAGE_VERTEX]       = MESA_SHADER_VERTEX,
   [S2I_STAGE_PIXEL]        = MESA_SHADER_FRAGMENT,
   [S2I_STAGE_COMPUTE]      = MESA_SHADER_COMPUTE,
   [S2I_STAGE_HULL]         = MESA_SHADER_TESS_CTRL,
   [S2I_STAGE_DOMAIN]       = MESA_SHADER_TESS_EVAL,
   [S2I_STAGE_GEOMETRY]     = MESA_SHADER_GEOMETRY,
   [S2I_STAGE_TASK]         = MESA_SHADER_TASK,
   [S2I_STAGE_MESH]         = MESA_SHADER_MESH,
   [S2I_STAGE_RAYGEN]       = MESA_SHADER_RAYGEN,
   [S2I_STAGE_CALLABLE]     = MESA_SHADER_CALLABLE,
   [S2I_STAGE_MISS]         = MESA_SHADER_MISS,
   [S2I_STAGE_CLOSEST_HIT]  = MESA_SHADER_CLOSEST_HIT,
   [S2I_STAGE_ANY_HIT]      = MESA_SHADER_ANY_HIT,
   [S2I_STAGE_INTERSECTION] = MESA_SHADER_INTERSECTION,
};

/* --- public API ---------------------------------------------------------------------------- */

/*
 * TODO(iterate in WSL against the built static libs): the Mesa-backed body. Sketch of the real calls
 * (nothing gates on stage - RADV's per-stage lowering handles all of them):
 *
 *   const struct s2i_target_desc *t = &s2i_targets[target];
 *   mesa_shader_stage ms = s2i_mesa_stage[stage];
 *
 *   struct radv_compiler_info ci = {0};
 *   ci.ac = ac_get_compiler_info(t->gfx_level, t->family);       // src/amd/common; device-free
 *   radv_fill_default_compiler_key(&ci);                         // sane offline defaults (!use_llvm, ...)
 *   ci.spirv_caps = ac_get_spirv_caps(t->gfx_level, t->family);  // per-target supported cap set
 *   radv_fill_nir_options(&ci, ms);
 *
 *   const struct spirv_to_nir_options so = { .capabilities = &ci.spirv_caps, ...radv_shader.c:503... };
 *   nir_shader *nir = spirv_to_nir(spirv, spirv_words, NULL, ms, entry, &so, &ci.nir_options[ms]);
 *   if (!nir) return S2I_UNSUPPORTED_CAP;                        // reported via debug cb, no crash
 *
 *   struct radv_shader_stage st = {0}; st.nir = nir; st.stage = ms;
 *   radv_nir_shader_info_pass(&ci, nir, &st.info);               // wave/workgroup/etc from nir->info
 *   synthesize_descriptor_layout(nir, &st);                      // from module's declared resources
 *   radv_declare_shader_args(&ci, &st, ...);
 *   radv_postprocess_nir(&ci, NULL, &st);
 *
 *   struct aco_compiler_options ao; radv_aco_fill_compiler_options(&ao, &ci, &st.key, ...);
 *   struct aco_shader_info ai;      radv_aco_convert_shader_info(&ai, &st.info, &st.args, &ci);
 *   aco_compile_shader(&ao, &ai, 1, &nir, &st.args.ac, s2i_aco_cb, &out);   // out -> isa_text + stats
 */
s2i_result
s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
            s2i_target target, char **isa_text, s2i_stats *stats, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;

   if (!spirv || spirv_words < 5 || spirv[0] != 0x07230203u)
      return S2I_BAD_SPIRV;
   if (!entry || target < 0 || target >= S2I_TARGET_COUNT || stage < 0 || stage >= S2I_STAGE_COUNT)
      return S2I_BAD_SPIRV;

   const struct s2i_target_desc *t = &s2i_targets[target];
   mesa_shader_stage mesa_stage = s2i_mesa_stage[stage];
   (void)t;
   (void)mesa_stage;
   (void)stats;

   /* TODO(iterate): Mesa-backed compile (see block comment above). Not yet wired. */
   return S2I_COMPILE_FAILED;
}

char *
s2i_unsupported_caps(const uint32_t *caps_used, size_t caps_count, s2i_target target)
{
   (void)caps_used;
   (void)caps_count;
   (void)target;
   /* TODO(iterate): build the target's spirv_capabilities (ac_get_spirv_caps for gfx_level), then for
    * each caller-supplied SpvCapability not present in it, append its name to a malloc'd list. */
   return NULL;
}
