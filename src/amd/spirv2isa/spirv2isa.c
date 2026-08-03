/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa: standalone SPIR-V -> AMD ISA via Mesa ACO, no VkDevice. See spirv2isa.h.
 *
 * This file is folded into RADV's own compilation (see src/amd/vulkan/meson.build) so it can call
 * the internal, device-free radv_* compile functions. The whole compute compile is one call:
 * radv_compile_cs(compiler_info, cs_stage, is_internal, dbg). We just build a device-free
 * radv_compiler_info (ac_fill_compiler_info with a NULL device gives the ac_compiler_info from just
 * gfx_level+family) and a minimal radv_shader_stage, then read stats from the binary config and the
 * ISA text from the debug info. The caller (OxC3) supplies entry+stage; we never scan the SPIR-V.
 */

#include "spirv2isa.h"

#include <stdlib.h>
#include <string.h>

#include "util/macros.h"
#include "util/ralloc.h"
#include "amd_family.h"                 /* enum amd_gfx_level, enum radeon_family */
#include "ac_gpu_info.h"                /* radeon_info + ac_fill_compiler_info (device-free) */
#include "compiler/shader_enums.h"      /* mesa_shader_stage / MESA_SHADER_* */
#include "compiler/glsl_types.h"        /* glsl_type_singleton_init_or_ref/decref (no device inits it) */
#include "radv_shader.h"                /* radv_compiler_info, radv_shader_stage, radv_get_nir_options, ... */
#include "radv_pipeline_compute.h"      /* radv_compile_cs */
#include "radv_descriptor_set.h"        /* radv_descriptor_set_layout (synthesized offline) */

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

/* --- stage mapping (caller's s2i_stage -> Mesa mesa_shader_stage; we never derive it) ------- */

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

   /* Only compute is wired for now (radv_compile_cs); other stages need the graphics pipeline key. */
   if (stage != S2I_STAGE_COMPUTE)
      return S2I_COMPILE_FAILED;

   /* No device did this for us; the glsl type system uses a singleton linear allocator. */
   glsl_type_singleton_init_or_ref();

   const struct s2i_target_desc *t = &s2i_targets[target];

   /* 1. device-free ac_compiler_info from just gfx_level + family (NULL device_info). */
   struct radeon_info rad;
   memset(&rad, 0, sizeof(rad));
   rad.gfx_level = t->gfx_level;
   rad.family = t->family;
   ac_fill_compiler_info(&rad, NULL, false);

   /* 2. radv_compiler_info: point at the ac info, fill per-stage nir_options + a permissive cap set. */
   struct radv_compiler_info ci;
   memset(&ci, 0, sizeof(ci));
   ci.ac = &rad.compiler_info;
   radv_get_nir_options(&ci);
   /* TODO(caps): translate OxC3's known caps; permissive-ish for now so basic modules pass. */
   memset(&ci.spirv_caps, 0, sizeof(ci.spirv_caps));
   ci.spirv_caps.Shader = true;

   /* Subgroup / wave size (the physical device normally supplies these; RADV divides by wave_size). */
   ci.subgroup_size = 64;
   ci.min_subgroup_size = (t->gfx_level >= GFX10) ? 32 : 64;
   ci.max_subgroup_size = 64;
   ci.key.cs_wave_size = (t->gfx_level >= GFX10) ? 32 : 64;

   /* 3. minimal compute stage: caller-supplied SPIR-V + entry; keep_executable_info -> we get disasm. */
   struct radv_shader_stage cs;
   memset(&cs, 0, sizeof(cs));
   cs.stage = MESA_SHADER_COMPUTE;
   cs.spirv.data = (const char *)spirv;
   cs.spirv.size = spirv_words * 4;
   cs.entrypoint = entry;
   cs.key.keep_executable_info = true;

   /* Minimal generic descriptor layout so buffer-using shaders lower without a real pipeline layout:
    * one set of storage buffers (covers RWByteAddressBuffer etc.). This is a placeholder; the real
    * design is OxC3 passing the exact binding types from its reflection, like it does entry/stage/caps. */
   const uint32_t s2i_nbind = 32;
   struct radv_descriptor_set_layout *dsl =
      calloc(1, sizeof(*dsl) + s2i_nbind * sizeof(struct radv_descriptor_set_binding_layout));
   dsl->binding_count = s2i_nbind;
   dsl->size = s2i_nbind * 16;
   for (uint32_t i = 0; i < s2i_nbind; i++) {
      dsl->binding[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      dsl->binding[i].array_size = 1;
      dsl->binding[i].offset = i * 16;
      dsl->binding[i].size = 16;
   }
   cs.layout.num_sets = 1;
   cs.layout.set[0].layout = dsl;

   struct radv_shader_debug_info dbg;
   memset(&dbg, 0, sizeof(dbg));

   struct radv_shader_binary *bin = radv_compile_cs(&ci, &cs, false, &dbg);
   if (!bin) {
      if (message)
         *message = strdup("radv_compile_cs failed");
      free(dsl);
      glsl_type_singleton_decref();
      return S2I_COMPILE_FAILED;
   }

   /* keep_executable_info made ACO record the asm into the binary; pull it out into dbg (disasm_string).
    * Without LLVM this is ACO's own listing (print_program) rather than a byte-encoded disassembly. */
   radv_parse_binary_debug_info(&ci, bin, &dbg);

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->sgprs = bin->config.num_sgprs;
      stats->vgprs = bin->config.num_vgprs;
      stats->scratch_size = bin->config.scratch_bytes_per_wave;
      stats->lds_size = bin->config.lds_size;
   }

   if (isa_text && dbg.disasm_string)
      *isa_text = strdup(dbg.disasm_string);

   free(dbg.disasm_string);
   free(dbg.ir_string);
   free(dbg.nir_string);
   free(bin);
   free(dsl);
   glsl_type_singleton_decref();
   return S2I_OK;
}

char *
s2i_unsupported_caps(const uint32_t *caps_used, size_t caps_count, s2i_target target)
{
   (void)caps_used;
   (void)caps_count;
   (void)target;
   /* TODO(iterate): build the target's spirv_capabilities and diff against the caller-supplied set. */
   return NULL;
}
