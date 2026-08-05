/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa: standalone SPIR-V -> AMD ISA via Mesa ACO, no VkDevice. See spirv2isa.h.
 *
 * This file is folded into RADV's own compilation (see src/amd/vulkan/meson.build) so it can call
 * the internal, device-free radv_* compile functions. All three stage classes go through one common
 * setup (a device-free radv_compiler_info from just gfx_level+family) and then a per-class compile:
 *   - compute:  radv_compile_cs (radv_pipeline_compute.h)
 *   - graphics: radv_graphics_shaders_compile (radv_pipeline_graphics.h), single unlinked stage
 *   - ray tracing: not yet wired (radv_rt_nir_to_asm is static; needs its monolithic path replicated)
 * The caller (OxC3) supplies entry + stage + the descriptor binding layout; we never scan the SPIR-V.
 */

#include "spirv2isa.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "util/macros.h"
#include "util/ralloc.h"
#include "util/list.h"                  /* list_inithead for the empty debug report below */
#include "vk_debug_report.h"            /* struct vk_debug_report (vtn reports through it) */
#include "amd_family.h"                 /* enum amd_gfx_level, enum radeon_family */
#include "ac_gpu_info.h"                /* radeon_info + ac_fill_compiler_info (device-free) */
#include "compiler/shader_enums.h"      /* mesa_shader_stage / MESA_SHADER_* */
#include "compiler/glsl_types.h"        /* glsl_type_singleton_init_or_ref/decref (no device inits it) */
#include "compiler/spirv/spirv.h"       /* SpvCapability* for s2i_unsupported_caps */
#include "nir.h"                        /* NIR_PASS + nir passes for the ray tracing path */
#include "nir_serialize.h"              /* nir_serialize (RT pipeline: callee NIR -> cache handle) */
#include "ac_nir.h"                     /* ac_nir_lower_indirect_derefs (RT preprocessing) */
#include "util/blob.h"                  /* blob for the NIR serialization */
#include "vk_alloc.h"                   /* vk_default_allocator for the stub vk_device */
#include "vk_device.h"                  /* struct vk_device (stub, only .alloc used) */
#include "vk_pipeline_cache.h"          /* vk_raw_data_cache_object_create (needs only device->alloc) */
#include "radv_constants.h"             /* MAX_SETS, MAX_RTS, RADV_*_DESC_SIZE */
#include "radv_shader.h"                /* radv_compiler_info, radv_shader_stage, radv_get_nir_options, ... */
#include "radv_shader_args.h"           /* radv_declare_shader_args (ray tracing path) */
#include "radv_shader_info.h"           /* radv_nir_shader_info_init/pass (ray tracing path) */
#include "radv_pipeline.h"              /* radv_postprocess_nir (ray tracing path) */
#include "radv_pipeline_compute.h"      /* radv_compile_cs */
#include "radv_pipeline_graphics.h"     /* radv_graphics_shaders_compile, radv_graphics_state_key */
#include "radv_pipeline_rt.h"           /* radv_ray_tracing_pipeline (synthesized offline) */
#include "radv_descriptor_set.h"        /* radv_descriptor_set_layout (synthesized offline) */
#include "tools/radv_rra.h"             /* radv_rra_trace_data (the RT traversal derefs ci->rra_trace) */
#include "nir/radv_nir.h"                       /* radv_nir_lower_call_abi */
#include "nir/radv_nir_rt_stage_monolithic.h"   /* radv_nir_lower_rt_io/abi_monolithic */
#include "nir/radv_nir_rt_stage_functions.h"    /* radv_get_rt_shader_entrypoint */
#include "nir/radv_nir_rt_traversal_shader.h"   /* radv_build_traversal_shader (non-monolithic pipeline) */

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

enum s2i_stage_class { S2I_CLASS_COMPUTE, S2I_CLASS_GRAPHICS, S2I_CLASS_RT };

static enum s2i_stage_class
s2i_class_of(s2i_stage stage)
{
   switch (stage) {
   case S2I_STAGE_COMPUTE:
      return S2I_CLASS_COMPUTE;
   case S2I_STAGE_RAYGEN:
   case S2I_STAGE_CALLABLE:
   case S2I_STAGE_MISS:
   case S2I_STAGE_CLOSEST_HIT:
   case S2I_STAGE_ANY_HIT:
   case S2I_STAGE_INTERSECTION:
      return S2I_CLASS_RT;
   default:
      return S2I_CLASS_GRAPHICS;
   }
}

/* --- descriptor layout (built from the caller's bindings; RADV indexes binding[binding_number]) - */

struct s2i_desc_info {
   VkDescriptorType vk;
   uint32_t size;   /* per-element descriptor size, mirroring RADV's own sizing */
};

static const struct s2i_desc_info s2i_desc[S2I_DESC_TYPE_COUNT] = {
   [S2I_DESC_SAMPLER]                = { VK_DESCRIPTOR_TYPE_SAMPLER,                RADV_SAMPLER_DESC_SIZE },
   [S2I_DESC_COMBINED_IMAGE_SAMPLER] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, RADV_STORAGE_IMAGE_DESC_SIZE + RADV_SAMPLER_DESC_SIZE },
   [S2I_DESC_SAMPLED_IMAGE]          = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          RADV_STORAGE_IMAGE_DESC_SIZE },
   [S2I_DESC_STORAGE_IMAGE]          = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          RADV_STORAGE_IMAGE_DESC_SIZE },
   [S2I_DESC_UNIFORM_TEXEL_BUFFER]   = { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_STORAGE_TEXEL_BUFFER]   = { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_UNIFORM_BUFFER]         = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_STORAGE_BUFFER]         = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_UNIFORM_BUFFER_DYNAMIC] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_STORAGE_BUFFER_DYNAMIC] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_INPUT_ATTACHMENT]       = { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       RADV_STORAGE_IMAGE_DESC_SIZE },
   [S2I_DESC_INLINE_UNIFORM_BLOCK]   = { VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,   RADV_BUFFER_DESC_SIZE },
   [S2I_DESC_ACCELERATION_STRUCTURE] = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, RADV_ACCEL_STRUCT_DESC_SIZE },
};

/* Every descriptor set layout we allocate for one compile, so we can free them all afterwards. */
struct s2i_layout_alloc {
   struct radv_descriptor_set_layout *owned[MAX_SETS + 1];
   uint32_t count;
};

static struct radv_descriptor_set_layout *
s2i_alloc_set_layout(uint32_t binding_count, struct s2i_layout_alloc *a)
{
   struct radv_descriptor_set_layout *dsl =
      calloc(1, sizeof(*dsl) + binding_count * sizeof(struct radv_descriptor_set_binding_layout));
   if (dsl && a->count < ARRAY_SIZE(a->owned))
      a->owned[a->count++] = dsl;
   return dsl;
}

/* Build one RADV descriptor set layout per set the caller referenced and attach them to `layout`.
 * Unused sets in [0, MAX_SETS) get a shared empty layout so any stray access stays in-bounds. The
 * exact offsets/sizes only need to be self-consistent for offline reflection; what matters for the
 * ISA is each binding's type (drives the descriptor-load lowering) and its set/binding placement. */
static bool
s2i_build_fed_layout(const s2i_binding *bindings, size_t n, struct radv_shader_layout *layout,
                     struct s2i_layout_alloc *a)
{
   uint32_t nbind[MAX_SETS] = {0};
   bool used[MAX_SETS] = {false};

   for (size_t i = 0; i < n; i++) {
      if (bindings[i].set >= MAX_SETS || bindings[i].type >= S2I_DESC_TYPE_COUNT)
         continue;
      used[bindings[i].set] = true;
      if (bindings[i].binding + 1 > nbind[bindings[i].set])
         nbind[bindings[i].set] = bindings[i].binding + 1;
   }

   struct radv_descriptor_set_layout *empty = s2i_alloc_set_layout(0, a);
   if (!empty)
      return false;

   layout->num_sets = MAX_SETS;
   for (uint32_t s = 0; s < MAX_SETS; s++) {
      if (!used[s]) {
         layout->set[s].layout = empty;
         continue;
      }

      struct radv_descriptor_set_layout *dsl = s2i_alloc_set_layout(nbind[s], a);
      if (!dsl)
         return false;
      dsl->binding_count = nbind[s];

      uint32_t offset = 0;
      for (uint32_t b = 0; b < nbind[s]; b++) {
         const s2i_binding *fb = NULL;
         for (size_t i = 0; i < n; i++) {
            if (bindings[i].set == s && bindings[i].binding == b) {
               fb = &bindings[i];
               break;
            }
         }

         if (fb) {
            uint32_t arr = fb->count ? fb->count : 1;
            uint32_t sz = s2i_desc[fb->type].size;
            dsl->binding[b].type = s2i_desc[fb->type].vk;
            dsl->binding[b].array_size = arr;
            dsl->binding[b].offset = offset;
            dsl->binding[b].size = sz;
            offset += sz * arr;
         } else {
            /* gap: a binding number the caller didn't use (the shader won't reference it) */
            dsl->binding[b].type = VK_DESCRIPTOR_TYPE_SAMPLER;
            dsl->binding[b].array_size = 0;
            dsl->binding[b].offset = offset;
            dsl->binding[b].size = 0;
         }
      }
      dsl->size = offset;
      layout->set[s].layout = dsl;
   }
   return true;
}

/* Fallback when the caller passes no bindings: a permissive generic set (many storage buffers)
 * replicated across all sets, enough to let simple buffer/image modules lower for a quick test. */
static bool
s2i_build_generic_layout(struct radv_shader_layout *layout, struct s2i_layout_alloc *a)
{
   const uint32_t nbind = 64;
   struct radv_descriptor_set_layout *dsl = s2i_alloc_set_layout(nbind, a);
   if (!dsl)
      return false;
   dsl->binding_count = nbind;
   dsl->size = nbind * 32;
   for (uint32_t i = 0; i < nbind; i++) {
      dsl->binding[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      dsl->binding[i].array_size = 1;
      dsl->binding[i].offset = i * 32;
      dsl->binding[i].size = 32;
   }
   layout->num_sets = MAX_SETS;
   for (uint32_t s = 0; s < MAX_SETS; s++)
      layout->set[s].layout = dsl;
   return true;
}

static void
s2i_free_layout(struct s2i_layout_alloc *a)
{
   for (uint32_t i = 0; i < a->count; i++)
      free(a->owned[i]);
   a->count = 0;
}

/* --- common device-free compiler setup ------------------------------------------------------ */

/* Fills a device-free radv_compiler_info for `t`, plus the empty debug report and wave sizes that a
 * physical device would normally supply. Points ci->ac at *rad (must outlive ci) and ci->debug at
 * *report (must outlive the compile). */
static void
s2i_setup_compiler_info(const struct s2i_target_desc *t, struct radeon_info *rad,
                        struct radv_compiler_info *ci, struct vk_debug_report *report)
{
   memset(rad, 0, sizeof(*rad));
   rad->gfx_level = t->gfx_level;
   rad->family = t->family;
   ac_fill_compiler_info(rad, NULL, false);

   memset(ci, 0, sizeof(*ci));
   ci->ac = &rad->compiler_info;
   radv_get_nir_options(ci);

   /* Enable all SPIR-V capabilities so vtn never takes its warn path (which reports through a device
    * we do not have and would crash). A capability the hardware truly cannot do still fails cleanly
    * later in ACO. s2i_unsupported_caps() is the pre-flight that tells the caller what a target lacks. */
   memset(&ci->spirv_caps, 0xff, sizeof(ci->spirv_caps));

   /* vtn reports warnings/errors through ci.debug.debug_report; a physical device normally owns it.
    * With none, RADV passes NULL and the first vtn message dereferences it (crash). Give it a real,
    * empty report: reporting becomes a clean no-op, and a genuine vtn failure (vtn_fail longjmp) then
    * unwinds to a NULL nir we turn into S2I_COMPILE_FAILED, instead of a segfault. */
   list_inithead(&report->callbacks);
   ci->debug.debug_report = report;

   /* The monolithic RT traversal derefs ci->rra_trace->ray_history_addr; a physical device owns this.
    * Point it at a zeroed record so ray_history_addr reads 0 (no ray-history capture), not a NULL deref. */
   static struct radv_rra_trace_data s2i_no_rra;
   ci->rra_trace = &s2i_no_rra;

   /* Subgroup / wave size (the physical device normally supplies these; RADV divides by wave_size). */
   ci->subgroup_size = 64;
   ci->min_subgroup_size = (t->gfx_level >= GFX10) ? 32 : 64;
   ci->max_subgroup_size = 64;
   ci->key.cs_wave_size = (t->gfx_level >= GFX10) ? 32 : 64;
   ci->key.ps_wave_size = (t->gfx_level >= GFX10) ? 32 : 64;
   ci->key.ge_wave_size = (t->gfx_level >= GFX10) ? 32 : 64;
   ci->key.rt_wave_size = (t->gfx_level >= GFX10) ? 32 : 64;
}

/* Copy the stat block + (optionally) the ISA text out of a compiled binary. */
static void
s2i_extract(const struct radv_compiler_info *ci, struct radv_shader_binary *bin, s2i_stats *stats,
            char **isa_text)
{
   /* keep_executable_info recorded the asm and keep_statistic_info the ACO stats into the binary;
    * pull both out through a scratch dbg. Without LLVM the disasm is ACO's own listing
    * (print_program), not a byte-encoded disassembly. */
   struct radv_shader_debug_info d;
   memset(&d, 0, sizeof(d));
   if (stats || isa_text)
      radv_parse_binary_debug_info(ci, bin, &d);

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->sgprs = bin->config.num_sgprs;
      stats->vgprs = bin->config.num_vgprs;
      stats->spilled_sgprs = bin->config.spilled_sgprs;
      stats->spilled_vgprs = bin->config.spilled_vgprs;
      stats->scratch_size = bin->config.scratch_bytes_per_wave;
      stats->lds_size = bin->config.lds_size;
      if (bin->type == RADV_BINARY_TYPE_LEGACY)
         stats->code_size = ((const struct radv_shader_binary_legacy *)bin)->code_size;
      if (d.statistics)
         stats->instructions = d.statistics->instrs;
   }

   if (isa_text && d.disasm_string)
      *isa_text = strdup(d.disasm_string);

   free(d.disasm_string);
   free(d.ir_string);
   free(d.nir_string);
   free(d.statistics);
}

/* Whether the module issues an OpTraceRayKHR (recursion / needs the SBT). Must run on the NIR before
 * the RT lowering rewrites trace_ray away. */
static bool
s2i_nir_has_trace_ray(nir_shader *nir)
{
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_trace_ray)
               return true;
         }
      }
   }
   return false;
}

/* Per-shader inlinability, matching radv_gather_ray_tracing_stage_info (static in RADV): raygen /
 * any-hit / intersection always inline, callable never, miss / closest-hit unless they traceRay. */
static bool
s2i_rt_can_inline(nir_shader *nir, mesa_shader_stage stage)
{
   if (stage == MESA_SHADER_RAYGEN || stage == MESA_SHADER_ANY_HIT || stage == MESA_SHADER_INTERSECTION)
      return true;
   if (stage == MESA_SHADER_CALLABLE)
      return false;
   return !s2i_nir_has_trace_ray(nir);
}

/* Ray tracing compile for one standalone shader, replicating radv_pipeline_rt.c's (static)
 * radv_rt_spirv_to_nir + radv_rt_nir_to_asm. A lone raygen that never traces is self-contained and
 * compiles MONOLITHICALLY (nothing to inline; a zeroed radv_ray_tracing_pipeline whose create_flags
 * are 0 is all that ABI reads). Everything else (callees: callable/miss/hit/anyhit/intersection, or a
 * raygen that traces) uses the FUNCTION_CALLS ABI so its real per-shader ISA (with the RT call/return
 * convention + payload/attribute passing) comes out, instead of a monolithic wrap that drops the body.
 * A true whole-pipeline monolithic compile (feeding all groups + SBT) is still a later step. */
static struct radv_shader_binary *
s2i_compile_rt(struct radv_compiler_info *ci, const uint32_t *spirv, size_t spirv_words,
               const char *entry, mesa_shader_stage ms, struct radv_shader_layout *layout,
               s2i_info *info, char **message)
{
   struct radv_shader_stage st;
   memset(&st, 0, sizeof(st));
   st.stage = ms;
   st.spirv.data = (const char *)spirv;
   st.spirv.size = spirv_words * 4;
   st.entrypoint = entry;
   st.key.keep_executable_info = true;
   st.key.keep_statistic_info = true;
   st.layout = *layout;

   struct radv_shader_debug_info dbg;
   memset(&dbg, 0, sizeof(dbg));

   st.nir = radv_shader_spirv_to_nir(ci, &st, NULL, false);
   if (!st.nir) {
      if (message)
         *message = strdup("spirv_to_nir failed (bad entrypoint or unsupported module)");
      return NULL;
   }

   /* Inspect the module before the RT lowering rewrites trace_ray away. */
   const bool traces = s2i_nir_has_trace_ray(st.nir);
   const bool monolithic = (ms == MESA_SHADER_RAYGEN) && !traces;
   if (info) {
      info->rt_can_inline = s2i_rt_can_inline(st.nir, ms) ? 1 : 0;
      info->rt_mode = monolithic ? S2I_RT_MODE_MONOLITHIC : S2I_RT_MODE_FUNCTION_CALLS;
   }

   /* Common RT preprocessing + payload / hit-attribute sizing (mirrors radv_rt_spirv_to_nir); the
    * FUNCTION_CALLS ABI needs those sizes to lay out payload + hit-attribute register/stack space. */
   NIR_PASS(_, st.nir, ac_nir_lower_indirect_derefs);
   NIR_PASS(_, st.nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, st.nir, nir_remove_dead_variables, nir_var_function_temp, NULL);

   uint32_t hit_attrib_size = 8, payload_size = 0;
   nir_foreach_variable_with_modes (var, st.nir, nir_var_ray_hit_attrib) {
      unsigned size, align;
      glsl_get_natural_size_align_bytes(var->type, &size, &align);
      hit_attrib_size = MAX2(hit_attrib_size, var->data.driver_location + size);
   }
   NIR_PASS(_, st.nir, radv_nir_lower_hit_attrib_derefs);
   nir_foreach_variable_with_modes (var, st.nir, nir_var_shader_call_data) {
      unsigned size, align;
      glsl_get_natural_size_align_bytes(var->type, &size, &align);
      payload_size = MAX2(payload_size, size);
   }
   nir_foreach_function_impl (impl, st.nir) {
      nir_foreach_variable_in_list (var, &impl->locals) {
         unsigned size, align;
         glsl_get_natural_size_align_bytes(var->type, &size, &align);
         payload_size = MAX2(payload_size, size);
      }
   }

   if (monolithic)
      radv_nir_lower_rt_io_monolithic(st.nir);
   else
      radv_nir_lower_rt_io_functions(st.nir);

   nir_shader_gather_info(st.nir, nir_shader_get_entrypoint(st.nir));
   radv_nir_shader_info_init(st.stage, MESA_SHADER_NONE, &st.info);
   radv_nir_shader_info_pass(ci, st.nir, &st.layout, &st.key, NULL, RADV_PIPELINE_RAY_TRACING, false,
                             &st.info);

   radv_declare_shader_args(ci, NULL, &st, MESA_SHADER_NONE, &dbg);
   st.info.user_sgprs_locs = st.args.user_sgprs_locs;
   st.info.inline_push_constant_mask = st.args.ac.inline_push_const_mask;
   st.info.type = radv_is_traversal_shader(st.nir) ? RADV_SHADER_TYPE_RT_TRAVERSAL : RADV_SHADER_TYPE_DEFAULT;

   struct radv_ray_tracing_pipeline rt_pipeline;
   memset(&rt_pipeline, 0, sizeof(rt_pipeline));
   if (monolithic)
      radv_nir_lower_rt_abi_monolithic(st.nir, ci, &rt_pipeline);
   else
      radv_nir_lower_rt_abi_functions(st.nir, &st.info, payload_size, hit_attrib_size, ci, &rt_pipeline);

   /* Info can be stale after the ABI lowering; re-gather then re-run the info pass, like RADV does. */
   nir_shader_gather_info(st.nir, radv_get_rt_shader_entrypoint(st.nir));
   radv_nir_shader_info_pass(ci, st.nir, &st.layout, &st.key, NULL, RADV_PIPELINE_RAY_TRACING, false,
                             &st.info);

   radv_optimize_nir(st.nir, st.key.optimisations_disabled);
   radv_postprocess_nir(ci, NULL, &st);

   NIR_PASS(_, st.nir, radv_nir_lower_call_abi, st.info.wave_size);
   NIR_PASS(_, st.nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, st.nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, st.nir, nir_opt_copy_prop);
   NIR_PASS(_, st.nir, nir_opt_remove_phis);
   if (!st.key.optimisations_disabled && !radv_is_traversal_shader(st.nir))
      NIR_PASS(_, st.nir, nir_minimize_call_live_states);

   struct radv_shader_binary *bin = radv_shader_nir_to_asm(ci, &st, &st.nir, 1, NULL);

   free(dbg.disasm_string);
   free(dbg.ir_string);
   free(dbg.nir_string);
   if (st.nir)
      ralloc_free(st.nir);
   return bin;
}

/* --- public API ---------------------------------------------------------------------------- */

s2i_result
s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
            s2i_target target, const s2i_binding *bindings, size_t binding_count, char **isa_text,
            s2i_stats *stats, s2i_info *info, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;
   if (info) {
      info->rt_mode = S2I_RT_MODE_NA;
      info->rt_can_inline = 0;
      info->graphics_specialized = 0;
   }

   if (!spirv || spirv_words < 5 || spirv[0] != 0x07230203u)
      return S2I_BAD_SPIRV;
   if (!entry || target < 0 || target >= S2I_TARGET_COUNT || stage < 0 || stage >= S2I_STAGE_COUNT)
      return S2I_BAD_SPIRV;

   const enum s2i_stage_class cls = s2i_class_of(stage);

   /* No device did this for us; the glsl type system uses a singleton linear allocator. */
   glsl_type_singleton_init_or_ref();

   const struct s2i_target_desc *t = &s2i_targets[target];

   struct radeon_info rad;
   struct radv_compiler_info ci;
   struct vk_debug_report report;
   s2i_setup_compiler_info(t, &rad, &ci, &report);

   /* Descriptor layout: caller-fed if provided (the precise path), else a generic fallback. */
   struct radv_shader_layout layout;
   memset(&layout, 0, sizeof(layout));
   struct s2i_layout_alloc lalloc = {0};
   bool ok = (bindings && binding_count) ? s2i_build_fed_layout(bindings, binding_count, &layout, &lalloc)
                                         : s2i_build_generic_layout(&layout, &lalloc);
   if (!ok) {
      s2i_free_layout(&lalloc);
      glsl_type_singleton_decref();
      if (message)
         *message = strdup("descriptor layout allocation failed");
      return S2I_COMPILE_FAILED;
   }

   struct radv_shader_binary *bin = NULL;
   const mesa_shader_stage ms = s2i_mesa_stage[stage];

   if (cls == S2I_CLASS_COMPUTE) {
      struct radv_shader_stage cs;
      memset(&cs, 0, sizeof(cs));
      cs.stage = MESA_SHADER_COMPUTE;
      cs.spirv.data = (const char *)spirv;
      cs.spirv.size = spirv_words * 4;
      cs.entrypoint = entry;
      cs.key.keep_executable_info = true;
      cs.key.keep_statistic_info = true;
      cs.layout = layout;

      struct radv_shader_debug_info dbg;
      memset(&dbg, 0, sizeof(dbg));
      bin = radv_compile_cs(&ci, &cs, false, &dbg);
      free(dbg.disasm_string);
      free(dbg.ir_string);
      free(dbg.nir_string);
   } else if (cls == S2I_CLASS_RT) {
      bin = s2i_compile_rt(&ci, spirv, spirv_words, entry, ms, &layout, info, message);
   } else {
      /* Graphics: one unlinked stage, exactly like VK_EXT_shader_object compiles a single stage.
       * The other stages stay MESA_SHADER_NONE; gfx_state carries the shader-object dynamic defaults
       * so the (device-free) graphics compile has a valid key to reason about. */
      struct radv_shader_stage stages[MESA_VULKAN_SHADER_STAGES];
      for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         memset(&stages[i], 0, sizeof(stages[i]));
         stages[i].stage = MESA_SHADER_NONE;
         stages[i].next_stage = MESA_SHADER_NONE;
      }
      stages[ms].stage = ms;
      stages[ms].spirv.data = (const char *)spirv;
      stages[ms].spirv.size = spirv_words * 4;
      stages[ms].entrypoint = entry;
      stages[ms].key.keep_executable_info = true;
      stages[ms].key.keep_statistic_info = true;
      stages[ms].layout = layout;

      struct radv_graphics_state_key gfx;
      memset(&gfx, 0, sizeof(gfx));
      gfx.vs.has_prolog = true;
      gfx.ps.has_epilog = true;
      gfx.dynamic_rasterization_samples = true;
      gfx.dynamic_provoking_vtx_mode = true;
      gfx.smooth_lines_may_be_enabled = true;
      gfx.rs.polygon_mode_unknown = true;
      gfx.ps.exports_mrtz_via_epilog = true;
      for (uint32_t i = 0; i < MAX_RTS; i++)
         gfx.ps.epilog.color_map[i] = i;

      struct radv_shader_binary *binaries[MESA_VULKAN_SHADER_STAGES] = {NULL};
      struct radv_shader_debug_info debug[MESA_VULKAN_SHADER_STAGES] = {0};
      struct radv_shader_debug_info gs_copy_debug = {0};
      struct radv_shader_binary *gs_copy_binary = NULL;

      radv_graphics_shaders_compile(&ci, NULL, stages, &gfx, false, NULL, false, debug, binaries,
                                    &gs_copy_debug, &gs_copy_binary);

      bin = binaries[ms];

      for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         if (stages[i].nir)
            ralloc_free(stages[i].nir);
         if (binaries[i] && i != ms)
            free(binaries[i]);
         free(debug[i].disasm_string);
         free(debug[i].ir_string);
         free(debug[i].nir_string);
      }
      free(gs_copy_binary);
      free(gs_copy_debug.disasm_string);
      free(gs_copy_debug.ir_string);
      free(gs_copy_debug.nir_string);
   }

   s2i_result result = S2I_OK;
   if (!bin) {
      if (message)
         *message = strdup("compile failed (no binary)");
      result = S2I_COMPILE_FAILED;
   } else {
      s2i_extract(&ci, bin, stats, isa_text);
      free(bin);
   }

   s2i_free_layout(&lalloc);
   glsl_type_singleton_decref();
   return result;
}

/* --- whole ray tracing pipeline (monolithic) ----------------------------------------------- */

/* Serialize a NIR shader into a pipeline-cache object handle (the form radv_ray_tracing_stage.nir
 * takes). vk_raw_data_cache_object_create only touches device->alloc, so a stub vk_device suffices,
 * no real device. The monolithic inlining reads it back device-free via radv_pipeline_cache_handle_to_nir. */
static struct vk_pipeline_cache_object *
s2i_nir_to_handle(struct vk_device *stub_dev, nir_shader *nir, uint32_t index)
{
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, false);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      return NULL;
   }
   void *data;
   size_t size;
   blob_finish_get_buffer(&blob, &data, &size);

   uint8_t key[BLAKE3_KEY_LEN];
   memset(key, 0, sizeof(key));
   memcpy(key, &index, sizeof(index)); /* distinct per shader; consumed directly, not looked up */

   struct vk_raw_data_cache_object *obj =
      vk_raw_data_cache_object_create(stub_dev, key, sizeof(key), data, size);
   free(data);
   return obj ? &obj->base : NULL;
}

/* Run the RT front half on one shader: spirv_to_nir + the radv_rt_spirv_to_nir preprocessing, and
 * accumulate the pipeline-wide payload / hit-attribute sizes. Returns the NIR (caller owns) or NULL. */
static nir_shader *
s2i_rt_shader_to_nir(struct radv_compiler_info *ci, struct radv_shader_layout *layout, mesa_shader_stage ms,
                     const uint32_t *spirv, size_t spirv_words, const char *entry, uint32_t *payload_size,
                     uint32_t *hit_attrib_size)
{
   struct radv_shader_stage st;
   memset(&st, 0, sizeof(st));
   st.stage = ms;
   st.spirv.data = (const char *)spirv;
   st.spirv.size = spirv_words * 4;
   st.entrypoint = entry;
   st.layout = *layout;

   nir_shader *nir = radv_shader_spirv_to_nir(ci, &st, NULL, false);
   if (!nir)
      return NULL;

   NIR_PASS(_, nir, ac_nir_lower_indirect_derefs);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);

   nir_foreach_variable_with_modes (var, nir, nir_var_ray_hit_attrib) {
      unsigned size, align;
      glsl_get_natural_size_align_bytes(var->type, &size, &align);
      *hit_attrib_size = MAX2(*hit_attrib_size, var->data.driver_location + size);
   }
   NIR_PASS(_, nir, radv_nir_lower_hit_attrib_derefs);
   nir_foreach_variable_with_modes (var, nir, nir_var_shader_call_data) {
      unsigned size, align;
      glsl_get_natural_size_align_bytes(var->type, &size, &align);
      *payload_size = MAX2(*payload_size, size);
   }
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_variable_in_list (var, &impl->locals) {
         unsigned size, align;
         glsl_get_natural_size_align_bytes(var->type, &size, &align);
         *payload_size = MAX2(*payload_size, size);
      }
   }
   return nir;
}

s2i_result
s2i_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count, size_t entry_index,
                        int compile_traversal, s2i_target target, const s2i_binding *bindings,
                        size_t binding_count, char **isa_text, s2i_stats *stats, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;
   if (!shaders || shader_count == 0 || target < 0 || target >= S2I_TARGET_COUNT)
      return S2I_BAD_SPIRV;
   if (!compile_traversal && (entry_index >= shader_count || shaders[entry_index].stage != S2I_STAGE_RAYGEN)) {
      if (message)
         *message = strdup("whole-pipeline entry shader must be a raygen");
      return S2I_BAD_SPIRV;
   }

   glsl_type_singleton_init_or_ref();

   const struct s2i_target_desc *t = &s2i_targets[target];
   struct radeon_info rad;
   struct radv_compiler_info ci;
   struct vk_debug_report report;
   s2i_setup_compiler_info(t, &rad, &ci, &report);

   struct radv_shader_layout layout;
   memset(&layout, 0, sizeof(layout));
   struct s2i_layout_alloc lalloc = {0};
   bool ok = (bindings && binding_count) ? s2i_build_fed_layout(bindings, binding_count, &layout, &lalloc)
                                         : s2i_build_generic_layout(&layout, &lalloc);
   if (!ok) {
      s2i_free_layout(&lalloc);
      glsl_type_singleton_decref();
      if (message)
         *message = strdup("descriptor layout allocation failed");
      return S2I_COMPILE_FAILED;
   }

   /* Stub vk_device: vk_raw_data_cache_object_create only needs its allocator. */
   struct vk_device stub_dev;
   memset(&stub_dev, 0, sizeof(stub_dev));
   stub_dev.alloc = *vk_default_allocator();

   struct radv_ray_tracing_stage *rt_stages = calloc(shader_count, sizeof(*rt_stages));
   struct radv_ray_tracing_group *groups = calloc(shader_count, sizeof(*groups));
   struct radv_shader_stage entry_st;
   memset(&entry_st, 0, sizeof(entry_st));
   nir_shader *entry_nir = NULL;
   uint32_t payload_size = 0, hit_attrib_size = 8;
   s2i_result result = S2I_OK;
   struct radv_shader_binary *bin = NULL;

   if (!rt_stages || !groups) {
      result = S2I_COMPILE_FAILED;
      goto cleanup;
   }

   /* Front half for every shader: NIR + payload/attrib sizing; serialize callees into stage handles;
    * build the groups. Each shader becomes one group (general for raygen/miss/callable, a hit group
    * for hit shaders); distinct synthetic handle pointers let the monolithic inliner switch on them. */
   for (size_t i = 0; i < shader_count; i++) {
      const mesa_shader_stage ms = s2i_mesa_stage[shaders[i].stage];
      nir_shader *nir = s2i_rt_shader_to_nir(&ci, &layout, ms, shaders[i].spirv, shaders[i].spirv_words,
                                             shaders[i].entry, &payload_size, &hit_attrib_size);
      if (!nir) {
         if (message)
            *message = strdup("spirv_to_nir failed for an RT pipeline shader (bad entry?)");
         result = S2I_NO_ENTRYPOINT;
         goto cleanup;
      }

      rt_stages[i].stage = ms;
      rt_stages[i].info.can_inline = s2i_rt_can_inline(nir, ms);
      rt_stages[i].info.set_flags = 0xFFFFFFFF;
      rt_stages[i].info.unset_flags = 0xFFFFFFFF;
      rt_stages[i].nir = s2i_nir_to_handle(&stub_dev, nir, (uint32_t)i);

      const bool general = ms == MESA_SHADER_RAYGEN || ms == MESA_SHADER_MISS || ms == MESA_SHADER_CALLABLE;
      groups[i].type = general ? VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR
                               : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      groups[i].recursive_shader = (uint32_t)i;
      groups[i].any_hit_shader = VK_SHADER_UNUSED_KHR;
      groups[i].intersection_shader = VK_SHADER_UNUSED_KHR;
      groups[i].handle.recursive_shader_ptr = 0x1000ull + i * 0x100ull;
      groups[i].handle.general_index = (uint32_t)i; /* union: also closest_hit_index */

      if (!compile_traversal && i == entry_index) {
         entry_nir = nir;
         entry_st.stage = ms;
         entry_st.entrypoint = shaders[i].entry;
         entry_st.key.keep_executable_info = true;
         entry_st.key.keep_statistic_info = true;
         entry_st.layout = layout;
         entry_st.nir = nir;
      } else {
         ralloc_free(nir); /* callee NIR (and, for traversal, every shader) lives on via its handle */
      }
   }

   struct radv_ray_tracing_pipeline pipeline;
   memset(&pipeline, 0, sizeof(pipeline));
   pipeline.stages = rt_stages;
   pipeline.stage_count = (unsigned)shader_count;
   pipeline.groups = groups;
   pipeline.group_count = (unsigned)shader_count;

   struct radv_shader_debug_info dbg;
   memset(&dbg, 0, sizeof(dbg));

   if (compile_traversal) {
      /* Build + compile the BVH-traversal shader a non-monolithic (function-calls) pipeline runs as a
       * separate stage (the raygen calls it). It is an intersection-typed shader over the groups. */
      /* Zeroed info = no pipeline-forced ray flags (set_flags/unset_flags = 0, all flags runtime-tested)
       * and every const arg UNINITIALIZED, i.e. the general-case traversal. radv would AND-narrow these
       * from the real stages' constant trace flags; without per-stage flag info we want the neutral 0/0
       * (0xFFFFFFFF here would force every ray flag ON at once, a degenerate walk). */
      struct radv_ray_tracing_stage_info tinfo;
      memset(&tinfo, 0, sizeof(tinfo));
      nir_shader *trav =
         radv_build_traversal_shader(&ci, &pipeline, &tinfo, NULL, payload_size, hit_attrib_size);
      if (!trav) {
         if (message)
            *message = strdup("radv_build_traversal_shader failed");
         result = S2I_COMPILE_FAILED;
         goto cleanup;
      }
      struct radv_shader_stage tst;
      memset(&tst, 0, sizeof(tst));
      tst.stage = MESA_SHADER_INTERSECTION;
      tst.nir = trav;
      tst.entrypoint = "main";
      tst.key.keep_executable_info = true;
      tst.key.keep_statistic_info = true;
      tst.layout = layout;

      radv_nir_lower_rt_io_functions(tst.nir);
      nir_shader_gather_info(tst.nir, nir_shader_get_entrypoint(tst.nir));
      radv_nir_shader_info_init(tst.stage, MESA_SHADER_NONE, &tst.info);
      radv_nir_shader_info_pass(&ci, tst.nir, &tst.layout, &tst.key, NULL, RADV_PIPELINE_RAY_TRACING,
                                false, &tst.info);
      radv_declare_shader_args(&ci, NULL, &tst, MESA_SHADER_NONE, &dbg);
      tst.info.user_sgprs_locs = tst.args.user_sgprs_locs;
      tst.info.inline_push_constant_mask = tst.args.ac.inline_push_const_mask;
      tst.info.type = RADV_SHADER_TYPE_RT_TRAVERSAL;
      radv_nir_lower_rt_abi_functions(tst.nir, &tst.info, payload_size, hit_attrib_size, &ci, &pipeline);
      nir_shader_gather_info(tst.nir, radv_get_rt_shader_entrypoint(tst.nir));
      radv_nir_shader_info_pass(&ci, tst.nir, &tst.layout, &tst.key, NULL, RADV_PIPELINE_RAY_TRACING,
                                false, &tst.info);
      radv_optimize_nir(tst.nir, tst.key.optimisations_disabled);
      radv_postprocess_nir(&ci, NULL, &tst);
      NIR_PASS(_, tst.nir, radv_nir_lower_call_abi, tst.info.wave_size);
      NIR_PASS(_, tst.nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, tst.nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, tst.nir, nir_opt_copy_prop);
      NIR_PASS(_, tst.nir, nir_opt_remove_phis);
      bin = radv_shader_nir_to_asm(&ci, &tst, &tst.nir, 1, NULL);
      ralloc_free(trav);
   } else {
      /* Compile the raygen MONOLITHICALLY against the real pipeline: its traceRay/executeCallable now
       * inline the callee shaders (from pipeline->stages[]) instead of becoming SBT calls. */
      radv_nir_lower_rt_io_monolithic(entry_st.nir);
      nir_shader_gather_info(entry_st.nir, nir_shader_get_entrypoint(entry_st.nir));
      radv_nir_shader_info_init(entry_st.stage, MESA_SHADER_NONE, &entry_st.info);
      radv_nir_shader_info_pass(&ci, entry_st.nir, &entry_st.layout, &entry_st.key, NULL,
                                RADV_PIPELINE_RAY_TRACING, false, &entry_st.info);
      radv_declare_shader_args(&ci, NULL, &entry_st, MESA_SHADER_NONE, &dbg);
      entry_st.info.user_sgprs_locs = entry_st.args.user_sgprs_locs;
      entry_st.info.inline_push_constant_mask = entry_st.args.ac.inline_push_const_mask;
      entry_st.info.type = RADV_SHADER_TYPE_DEFAULT;

      radv_nir_lower_rt_abi_monolithic(entry_st.nir, &ci, &pipeline);

      nir_shader_gather_info(entry_st.nir, radv_get_rt_shader_entrypoint(entry_st.nir));
      radv_nir_shader_info_pass(&ci, entry_st.nir, &entry_st.layout, &entry_st.key, NULL,
                                RADV_PIPELINE_RAY_TRACING, false, &entry_st.info);
      radv_optimize_nir(entry_st.nir, entry_st.key.optimisations_disabled);
      radv_postprocess_nir(&ci, NULL, &entry_st);
      NIR_PASS(_, entry_st.nir, radv_nir_lower_call_abi, entry_st.info.wave_size);
      NIR_PASS(_, entry_st.nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, entry_st.nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, entry_st.nir, nir_opt_copy_prop);
      NIR_PASS(_, entry_st.nir, nir_opt_remove_phis);

      bin = radv_shader_nir_to_asm(&ci, &entry_st, &entry_st.nir, 1, NULL);
   }

   free(dbg.disasm_string);
   free(dbg.ir_string);
   free(dbg.nir_string);

cleanup:
   if (result == S2I_OK) {
      if (!bin) {
         if (message && !*message)
            *message = strdup("monolithic RT pipeline compile failed");
         result = S2I_COMPILE_FAILED;
      } else {
         s2i_extract(&ci, bin, stats, isa_text);
         free(bin);
      }
   }
   if (entry_nir)
      ralloc_free(entry_nir);
   if (rt_stages) {
      for (size_t i = 0; i < shader_count; i++)
         if (rt_stages[i].nir)
            vk_pipeline_cache_object_unref(&stub_dev, rt_stages[i].nir);
   }
   free(rt_stages);
   free(groups);
   s2i_free_layout(&lalloc);
   glsl_type_singleton_decref();
   return result;
}

/*
 * Caller passes the SpvCapability values it already knows the module uses (OxC3 has them from its
 * reflection). We return the ones that need a newer AMD generation than `target` supports on RADV.
 * This is a small gen-gate table, not a full device query (that needs a VkPhysicalDevice); a
 * capability not listed here is supported across all our targets (GFX8+). OxC3's own capability
 * matrix is the authoritative higher layer; this is a convenience pre-flight.
 */
static const struct {
   uint32_t cap;
   enum amd_gfx_level min;
   const char *name;
} s2i_cap_gates[] = {
   { SpvCapabilityRayQueryKHR,                    GFX10_3, "RayQueryKHR" },
   { SpvCapabilityRayTracingKHR,                  GFX10_3, "RayTracingKHR" },
   { SpvCapabilityRayTraversalPrimitiveCullingKHR, GFX10_3, "RayTraversalPrimitiveCullingKHR" },
   { SpvCapabilityRayTracingPositionFetchKHR,     GFX11,   "RayTracingPositionFetchKHR" },
   { SpvCapabilityMeshShadingEXT,                 GFX10_3, "MeshShadingEXT" },
   { SpvCapabilityFragmentBarycentricKHR,         GFX10_3, "FragmentBarycentricKHR" },
   { SpvCapabilityFragmentShadingRateKHR,         GFX10_3, "FragmentShadingRateKHR" },
   { SpvCapabilityCooperativeMatrixKHR,           GFX11,   "CooperativeMatrixKHR" },
};

char *
s2i_unsupported_caps(const uint32_t *caps_used, size_t caps_count, s2i_target target)
{
   if (!caps_used || !caps_count || target < 0 || target >= S2I_TARGET_COUNT)
      return NULL;

   const enum amd_gfx_level gfx = s2i_targets[target].gfx_level;

   char buf[2048];
   size_t n = 0;
   buf[0] = '\0';

   for (size_t i = 0; i < caps_count; i++) {
      const char *name = NULL;
      for (size_t g = 0; g < ARRAY_SIZE(s2i_cap_gates); g++) {
         if (caps_used[i] == s2i_cap_gates[g].cap) {
            if (gfx < s2i_cap_gates[g].min)
               name = s2i_cap_gates[g].name;
            break;
         }
      }
      if (name && n + strlen(name) + 2 < sizeof(buf))
         n += (size_t)snprintf(buf + n, sizeof(buf) - n, "%s%s", n ? "\n" : "", name);
   }

   return n ? strdup(buf) : NULL;
}
