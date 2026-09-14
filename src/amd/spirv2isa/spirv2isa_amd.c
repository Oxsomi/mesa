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
 *   - ray tracing: s2i_compile_rt for one shader, s2i_amd_compile_rt_pipeline for a whole one
 * The caller (OxC3) supplies entry + stage + the descriptor binding layout; we never scan the SPIR-V.
 */

#include "spirv2isa_amd.h"
#include "spirv2isa_desc.h"
#include "spirv2isa_stage.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "util/macros.h"
#include "util/ralloc.h"
#include "util/list.h"                  /* list_inithead for the empty debug report below */
#include "vk_debug_report.h"            /* struct vk_debug_report (vtn reports through it) */
#include "amd_family.h"                 /* enum amd_gfx_level, enum radeon_family */
#include "ac_gpu_info.h"                /* radeon_info + ac_fill_compiler_info (device-free) */
#include "compiler/shader_enums.h"      /* mesa_shader_stage / MESA_SHADER_* */
#include "compiler/glsl_types.h"        /* glsl_get_aoa_size and the layout walk type queries */
#include "compiler/spirv/spirv.h"       /* SpvCapability* for s2i_unsupported_caps */
#include "nir.h"                        /* NIR_PASS + nir passes for the ray tracing path */
#include "nir_serialize.h"              /* nir_serialize (RT pipeline: callee NIR -> cache handle) */
#include "ac_nir.h"                     /* ac_nir_lower_indirect_derefs (RT preprocessing) */
#include "util/blob.h"                  /* blob for the NIR serialization */
#include "vk_alloc.h"                   /* vk_default_allocator for the stub vk_device */
#include "vk_device.h"                  /* struct vk_device (stub, only .alloc used) */
#include "vk_pipeline_cache.h"          /* vk_raw_data_cache_object_create (needs only device->alloc) */
#include "radv_constants.h"             /* MAX_SETS, MAX_RTS, RADV_*_DESC_SIZE */
#include "radv_physical_device.h"       /* radv_physical_device_offline_supported */
#include "radv_instance.h"              /* RADV_API_VERSION */
#include "radv_drirc.h"                 /* radv_drirc_defaults */
#include "vk_physical_device.h"         /* vk_physical_device_get_spirv_capabilities */
#include "spirv2isa_caps.h"             /* the shared declared-capability gate */
#include "spirv2isa_state.h"            /* the pipeline state a module does not carry */
#include "spirv2isa_session.h"          /* the generic setup and its one teardown */
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

static const struct s2i_target_desc s2i_amd_targets[S2I_TARGET_AMD_COUNT] = {
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GFX8_POLARIS10)] = { GFX8,    CHIP_POLARIS10, "gfx803 (GCN4, RX 580)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GFX9_VEGA10)]    = { GFX9,    CHIP_VEGA10,    "gfx900 (GCN5, RX Vega)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GFX10_NAVI10)]   = { GFX10,   CHIP_NAVI10,    "gfx1010 (RDNA1, RX 5700 XT)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GFX10_3_NAVI21)] = { GFX10_3, CHIP_NAVI21,    "gfx1030 (RDNA2, RX 6800/6700 XT class)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GFX11_NAVI31)]   = { GFX11,   CHIP_NAVI31,    "gfx1100 (RDNA3, RX 7900 XTX)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GFX12_GFX1201)]  = { GFX12,   CHIP_GFX1201,   "gfx1201 (RDNA4, RX 9070 XT)" },
};

/*
 * The extended tier: every family from GFX8 up that Mesa's own tables know and no flagship row
 * already covers, so new hardware appears with a Mesa update rather than an edit here. Level and
 * names come from those tables too; the per-family facts come from ac_fill_compiler_info the same
 * way they do for a flagship. Built once on first use, unsynchronized like the library's other
 * process-global state.
 */
struct s2i_amd_extended_row {
   struct s2i_target_desc desc;
   char token[32];
   char name[64];
};

static struct s2i_amd_extended_row s2i_amd_extended[CHIP_LAST];
static int s2i_amd_extended_count = -1;

static void
s2i_amd_build_extended(void)
{
   if (s2i_amd_extended_count >= 0)
      return;

   int n = 0;

   for (int f = CHIP_UNKNOWN + 1; f < CHIP_LAST; ++f) {

      const enum amd_gfx_level gfx = ac_get_gfx_level((enum radeon_family)f);
      int flagship = 0;

      if (gfx < GFX8)
         continue;

      for (int i = 0; i < (int)S2I_TARGET_AMD_COUNT; ++i)
         flagship |= s2i_amd_targets[i].family == (enum radeon_family)f;

      if (flagship)
         continue;

      struct s2i_amd_extended_row *row = &s2i_amd_extended[n];

      row->desc.gfx_level = gfx;
      row->desc.family = (enum radeon_family)f;

      /* Built in a local so the name below is provably not written from its own object. */
      char token[sizeof(row->token)];
      snprintf(token, sizeof(token), "%s", ac_get_family_name((enum radeon_family)f));

      for (char *c = token; *c; ++c)
         *c = (char)tolower((unsigned char)*c);

      /* The LLVM processor name is the useful alias ("navi22 (gfx1031)"); when it is absent (a
       * family newer than the table) or identical to the token, the token alone is the name. */
      const char *llvm_name = ac_get_llvm_processor_name((enum radeon_family)f);

      if (llvm_name && llvm_name[0] && strcmp(llvm_name, token))
         snprintf(row->name, sizeof(row->name), "%s (%s)", token, llvm_name);
      else
         snprintf(row->name, sizeof(row->name), "%s", token);

      snprintf(row->token, sizeof(row->token), "%s", token);

      row->desc.name = row->name;

      ++n;
   }

   s2i_amd_extended_count = n;
}

/* Row for a target index across both tiers, NULL when the index is out of range. */
static const struct s2i_target_desc *
s2i_amd_target(int target_index)
{
   if (target_index >= 0 && target_index < (int)S2I_TARGET_AMD_COUNT)
      return &s2i_amd_targets[target_index];

   s2i_amd_build_extended();

   if (target_index >= (int)S2I_TARGET_AMD_COUNT &&
       target_index < (int)S2I_TARGET_AMD_COUNT + s2i_amd_extended_count)
      return &s2i_amd_extended[target_index - S2I_TARGET_AMD_COUNT].desc;

   return NULL;
}

int
s2i_amd_target_count(void)
{
   s2i_amd_build_extended();
   return (int)S2I_TARGET_AMD_COUNT + s2i_amd_extended_count;
}

const char *
s2i_amd_target_token(int target_index)
{
   s2i_amd_build_extended();

   if (target_index < (int)S2I_TARGET_AMD_COUNT ||
       target_index >= (int)S2I_TARGET_AMD_COUNT + s2i_amd_extended_count)
      return "";

   return s2i_amd_extended[target_index - S2I_TARGET_AMD_COUNT].token;
}

const char *
s2i_amd_target_name(int target_index)
{
   const struct s2i_target_desc *t = s2i_amd_target(target_index);

   return t ? t->name : "";
}


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

/* Per-element descriptor size, mirroring RADV's own sizing. A switch rather than a table because
 * VkDescriptorType is sparse: its last members are extension values in the billions. */
static uint32_t
s2i_desc_size(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:                return RADV_SAMPLER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return RADV_STORAGE_IMAGE_DESC_SIZE +
                                                          RADV_SAMPLER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:          return RADV_STORAGE_IMAGE_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:          return RADV_STORAGE_IMAGE_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:   return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:   return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:         return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:         return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:       return RADV_STORAGE_IMAGE_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:   return RADV_BUFFER_DESC_SIZE;
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return RADV_ACCEL_STRUCT_DESC_SIZE;
   default:                                        return 0;
   }
}

/* RADV's set layout is a header with its bindings inline, so one sized allocation on the session's
 * context, which releases them all at the end of the compile. */
static struct radv_descriptor_set_layout *
s2i_alloc_set_layout(uint32_t binding_count, void *mem_ctx)
{
   return (struct radv_descriptor_set_layout *) rzalloc_size(
      mem_ctx, sizeof(struct radv_descriptor_set_layout) +
               binding_count * sizeof(struct radv_descriptor_set_binding_layout));
}

/* Build one RADV descriptor set layout per set the caller referenced and attach them to `layout`.
 * Unused sets in [0, MAX_SETS) get a shared empty layout so any stray access stays in-bounds. The
 * offsets and sizes reach the ISA, not just reflection: they decide which descriptor a load
 * resolves to, so a layout that differs from the pipeline's produces different code. */
static bool
s2i_build_fed_layout(const VkDescriptorSetLayoutCreateInfo *const *set_layouts,
                     uint32_t set_layout_count, struct radv_shader_layout *layout, void *mem_ctx)
{
   uint32_t nbind[MAX_SETS] = {0};

   _Static_assert(MAX_SETS <= 32, "one bit per set below");
   uint32_t used = 0;

   for (uint32_t s = 0; s < set_layout_count && s < MAX_SETS; s++) {

      if (!set_layouts[s])
         continue;

      used |= 1u << s;

      for (uint32_t i = 0; i < set_layouts[s]->bindingCount; i++) {
         const uint32_t b = set_layouts[s]->pBindings[i].binding;

         if (b + 1 > nbind[s])
            nbind[s] = b + 1;
      }
   }

   struct radv_descriptor_set_layout *empty = s2i_alloc_set_layout(0, mem_ctx);
   if (!empty)
      return false;

   layout->num_sets = MAX_SETS;
   for (uint32_t s = 0; s < MAX_SETS; s++) {
      if (!(used & (1u << s))) {
         layout->set[s].layout = empty;
         continue;
      }

      struct radv_descriptor_set_layout *dsl = s2i_alloc_set_layout(nbind[s], mem_ctx);
      if (!dsl)
         return false;
      dsl->binding_count = nbind[s];

      uint32_t offset = 0;
      for (uint32_t b = 0; b < nbind[s]; b++) {
         const VkDescriptorSetLayoutBinding *fb = NULL;
         for (uint32_t i = 0; i < set_layouts[s]->bindingCount; i++) {
            if (set_layouts[s]->pBindings[i].binding == b) {
               fb = &set_layouts[s]->pBindings[i];
               break;
            }
         }

         if (fb) {
            uint32_t arr = fb->descriptorCount ? fb->descriptorCount : 1;
            uint32_t sz = s2i_desc_size(fb->descriptorType);
            dsl->binding[b].type = fb->descriptorType;
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

/* Fallback when the caller passes no layouts: one set of plain storage buffers repeated for every
 * set. This is a placeholder, NOT the module's layout. A binding the module uses as anything else
 * is described wrongly, and since offsets reach the ISA the code differs from what the real layout
 * produces. Intel and NVK synthesize the module's own layout instead; docs/roadmap.md tracks doing
 * the same here. */
static bool
s2i_build_generic_layout(struct radv_shader_layout *layout, void *mem_ctx)
{
   const uint32_t nbind = 64;
   struct radv_descriptor_set_layout *dsl = s2i_alloc_set_layout(nbind, mem_ctx);
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

/* --- common device-free compiler setup ------------------------------------------------------ */

/* The capability set a physical device would report, without having one. Allocated rather than put
 * on the stack because a radv_physical_device carries the whole property table, and freed straight
 * after: only the capability bitset outlives it. An allocation failure leaves the set empty, which
 * refuses every module by name rather than letting one through unchecked. */
static void
s2i_amd_fill_spirv_caps(const struct radeon_info *info, struct spirv_capabilities *caps,
                        bool *rt_exposed)
{
   struct radv_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct radv_instance *instance = calloc(1, sizeof(*instance));

   memset(caps, 0, sizeof(*caps));
   *rt_exposed = false;

   if (pdev && instance) {

      pdev->info = *info;
      instance->vk.app_info.api_version = RADV_API_VERSION;
      pdev->vk.instance = &instance->vk;

      /* The property fill reads driconf options straight off the instance, and the string ones are
       * char pointers: a zeroed instance takes strlen(NULL). The generated defaults are what a
       * driver starts from before any config file is read. */
      radv_drirc_defaults(&instance->drirc);

      radv_physical_device_offline_supported(pdev);
      *caps = vk_physical_device_get_spirv_capabilities(&pdev->vk);

      /* RADV sets .rayTracingPipeline in the feature struct unconditionally and hides RT behind the
       * extension list instead, so the capability set alone says yes on hardware with no BVH unit.
       * The extension is the driver's real answer. */
      *rt_exposed = pdev->vk.supported_extensions.KHR_ray_tracing_pipeline;
   }

   free(instance);
   free(pdev);
}

/* Fills a device-free radv_compiler_info for `t`, plus the empty debug report and wave sizes that a
 * physical device would normally supply, and reports through `rt_exposed` whether RADV would expose
 * ray tracing here. Points ci->ac at *rad (must outlive ci) and ci->debug at *report (must outlive
 * the compile). */
static void
s2i_setup_compiler_info(const struct s2i_target_desc *t, struct radeon_info *rad,
                        struct radv_compiler_info *ci, struct vk_debug_report *report,
                        bool *rt_exposed)
{
   memset(rad, 0, sizeof(*rad));
   rad->gfx_level = t->gfx_level;
   rad->family = t->family;

   /* Everything this tool models has a graphics ring except the CDNA compute dies; several ac
    * tables key ABI choices off (!has_graphics && family >= X), which with a zeroed flag misfires
    * for every family enumerated after the CDNA block. */
   rad->has_graphics =
      t->family != CHIP_MI100 && t->family != CHIP_MI200 && t->family != CHIP_GFX940;

   ac_fill_compiler_info(rad, NULL, false);

   memset(ci, 0, sizeof(*ci));
   ci->ac = &rad->compiler_info;

   /* ACO disassembles through LLVM's MC layer, which it can only construct from a named processor;
    * a zeroed family leaves it unable to and it prints its own IR listing instead, so the artifact
    * would be ACO's internal representation rather than the machine code this tool exists to show.
    */
   ci->debug.family = t->family;

   radv_get_nir_options(ci);

   /* The capabilities this target really has, computed the way the driver computes them: RADV fills
    * its supported tables from the device info and the instance, and the generated mapping turns
    * those into SPIR-V capabilities. A physical device is the only thing those tables hang off, so
    * one is built here and thrown away; nothing in it needs a winsys or an open device. */
   s2i_amd_fill_spirv_caps(rad, &ci->spirv_caps, rt_exposed);

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

   /* The per-stage wave size DEFAULTS a physical device supplies, matching RADV with no perftest
    * flags set: 64 for compute, pixel and geometry, 32 for ray tracing from GFX10 on.
    * They are only defaults. RADV picks each shader's real wave size in radv_shader_spirv_to_nir: a
    * workgroup of 32 or fewer invocations compiles wave32 whatever this says, as does a required
    * subgroup size, and a shader using ray queries takes the ray tracing size. The size actually
    * chosen comes back in the stats. */
   ci->subgroup_size = 64;
   ci->min_subgroup_size = (t->gfx_level >= GFX10) ? 32 : 64;
   ci->max_subgroup_size = 64;
   ci->key.cs_wave_size = 64;
   ci->key.ps_wave_size = 64;
   ci->key.ge_wave_size = 64;
   ci->key.rt_wave_size = (t->gfx_level >= GFX10) ? 32 : 64;
}

/* Every heap field RADV's compile paths hand back. A driver releases these when it serializes the
 * shader into its pipeline cache; there is no cache here, so this is where they go. */
static void
s2i_free_debug_info(struct radv_shader_debug_info *dbg)
{
   free(dbg->spirv);
   free(dbg->nir_string);
   free(dbg->disasm_string);
   free(dbg->ir_string);
   free(dbg->args_string);
   free(dbg->statistics);
   free(dbg->debug_info);
   memset(dbg, 0, sizeof(*dbg));
}

/* Copy the stat block + (optionally) the ISA text out of a compiled binary. */
static void
s2i_extract(const struct radv_compiler_info *ci, struct radv_shader_binary *bin, s2i_stats_amd *stats,
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
      stats->wave_size = bin->info.wave_size;

      if (d.statistics)
         stats->instructions = d.statistics->instrs;
   }

   if (isa_text && d.disasm_string)
      *isa_text = strdup(d.disasm_string);

   s2i_free_debug_info(&d);
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
               const struct vk_pipeline_robustness_state *rs, s2i_info *info, bool *no_entry,
               char **message)
{
   struct radv_shader_stage st;
   memset(&st, 0, sizeof(st));
   st.stage = ms;
   st.spirv.data = (const char *)spirv;
   st.spirv.size = spirv_words * 4;
   st.entrypoint = entry;
   st.key.keep_executable_info = true;
   st.key.keep_statistic_info = true;
   radv_set_stage_key_robustness(rs, ms, &st.key);
   st.layout = *layout;

   struct radv_shader_debug_info dbg;
   memset(&dbg, 0, sizeof(dbg));

   st.nir = radv_shader_spirv_to_nir(ci, &st, NULL, false);
   if (!st.nir) {
      if (message)
         *message = strdup("spirv_to_nir failed (bad entrypoint or unsupported module)");
      *no_entry = true;
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

   s2i_free_debug_info(&dbg);
   if (st.nir)
      ralloc_free(st.nir);
   return bin;
}

/* The first extension the module declares that `deny` names, or NULL. Only the mode-setting prefix
 * is walked, since OpExtension precedes the first OpFunction. Why an entry is in `deny` is
 * documented on the list itself. */
static const char *
s2i_first_unsupported_ext(const uint32_t *spirv, size_t words, const char *const *deny,
                          size_t deny_count)
{
   for (size_t i = 5; i < words;) {
      uint32_t op = spirv[i] & 0xFFFFu;
      uint32_t len = spirv[i] >> 16;
      if (len == 0 || i + len > words)
         break;
      if (op == 54 /* OpFunction */) /* extensions/capabilities are all before the first function */
         break;
      if (op == 10 /* OpExtension */ && len > 1) {
         /* The literal is NUL-padded inside this instruction's own words; compare with an explicit
          * bound so a malformed module with an unterminated literal cannot read past the module. */
         const char *ext = (const char *)&spirv[i + 1];
         const size_t avail = (size_t)(len - 1) * 4;
         if (memchr(ext, '\0', avail)) {
            for (size_t d = 0; d < deny_count; d++)
               if (strcmp(ext, deny[d]) == 0)
                  return deny[d];
         }
      }
      i += len;
   }
   return NULL;
}

/* The extensions RADV advertises that this path still cannot lower, which crash inside spirv_to_nir
 * rather than failing. An entry earns its place ONLY while both halves hold: the driver advertises
 * it, so the capability gate lets it through, and this path cannot compile it. Anything the driver
 * does not advertise leaves its feature false and the gate refuses it by capability, which is how
 * Intel and NVK refuse the same modules carrying no list at all. */
static const char *const s2i_amd_unsupported_exts[] = {
   "SPV_EXT_descriptor_heap",
};

/* --- public API ---------------------------------------------------------------------------- */

s2i_result
s2i_amd_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                int target_index, const s2i_pipeline *pipeline, char **isa_text,
                s2i_stats_amd *stats, s2i_info *info, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;
   if (info) {
      info->rt_mode = S2I_RT_MODE_NA;
      info->rt_can_inline = 0;
      info->graphics_specialized = 0;
      info->notes = NULL;
   }

   if (!s2i_module_args_ok(spirv, spirv_words, entry, stage) || target_index < 0 ||
       target_index >= s2i_amd_target_count())
      return S2I_BAD_SPIRV;

   /* Runs before the capability gate below because these are the ones that gate lets through. */
   const char *bad_ext =
      s2i_first_unsupported_ext(spirv, spirv_words, s2i_amd_unsupported_exts,
                                ARRAY_SIZE(s2i_amd_unsupported_exts));
   if (bad_ext) {
      if (message) {
         char buf[128];
         snprintf(buf, sizeof(buf), "unsupported SPIR-V extension for AMD/RADV: %s", bad_ext);
         *message = strdup(buf);
      }
      return S2I_UNSUPPORTED_CAP;
   }

   /* Mesh and task are NGG-only stages that do not exist before GFX10.3. Enabling key.use_ngg for them
    * (needed so radv_get_user_data_0 accepts a mesh stage) walks an older target straight into ACO's NGG
    * path and crashes, so reject here rather than compile something the target cannot express. */
   if ((s2i_stage_to_mesa(stage) == MESA_SHADER_MESH || s2i_stage_to_mesa(stage) == MESA_SHADER_TASK) &&
       s2i_amd_target(target_index)->gfx_level < GFX10_3) {
      if (message) {
         char buf[160];
         snprintf(buf, sizeof(buf), "mesh and task shaders need GFX10.3 or newer; %s cannot run them",
                  s2i_amd_target_name(target_index));
         *message = strdup(buf);
      }
      return S2I_UNSUPPORTED_CAP;
   }

   const enum s2i_stage_class cls = s2i_class_of(stage);

   struct s2i_session session;

   if (!s2i_session_open(&session))
      return S2I_COMPILE_FAILED;

   s2i_result result = S2I_OK;
   struct radv_shader_binary *bin = NULL;

   const struct s2i_target_desc *t = s2i_amd_target(target_index);

   struct radeon_info rad;
   struct radv_compiler_info ci;
   struct vk_debug_report report;
   bool rt_exposed = false;
   s2i_setup_compiler_info(t, &rad, &ci, &report, &rt_exposed);

   /* A target RADV would not expose ray tracing on has no BVH unit to run it, so its RT ISA would
    * be for hardware that cannot execute it. */
   if (cls == S2I_CLASS_RT && !rt_exposed) {

      if (message) {
         char buf[160];
         snprintf(buf, sizeof(buf),
                  "%s has no ray tracing on RADV, so it has no ray tracing stages",
                  s2i_amd_target_name(target_index));
         *message = strdup(buf);
      }

      result = S2I_UNSUPPORTED_CAP;
      goto done;
   }

   /* Every capability the module declares, against the set this target really has. vtn only warns
    * on one it lacks and parses on, so without this a refusal arrives as a crash inside a lowering
    * pass, or as ISA for something the hardware cannot do. */
   {
      bool malformed = false;
      char *refusal = s2i_gate_declared_capabilities(spirv, spirv_words, &ci.spirv_caps,
                                                     s2i_amd_target_name(target_index), &malformed);

      if (refusal) {

         if (message)
            *message = refusal;
         else
            free(refusal);

         result = malformed ? S2I_BAD_SPIRV : S2I_UNSUPPORTED_CAP;
         goto done;
      }
   }

   /* Descriptor layout: caller-fed if provided (the precise path), else a generic fallback. */
   struct radv_shader_layout layout;
   memset(&layout, 0, sizeof(layout));

   if (!((pipeline->set_layouts && pipeline->set_layout_count)
            ? s2i_build_fed_layout(pipeline->set_layouts, pipeline->set_layout_count, &layout,
                                   session.mem_ctx)
            : s2i_build_generic_layout(&layout, session.mem_ctx))) {
      if (message)
         *message = strdup("descriptor layout allocation failed");
      result = S2I_COMPILE_FAILED;
      goto done;
   }

   /* RADV carries robustness as stage key bits, so it reaches the compile through whichever stage
    * struct the class below builds. */
   const struct vk_pipeline_robustness_state rs = s2i_robustness_state(pipeline->robustness);

   const mesa_shader_stage ms = s2i_stage_to_mesa(stage);

   /* Distinguishes a module vtn could not parse (a wrong entrypoint name is the usual cause) from a
    * backend that ran and produced nothing, which are different answers to the caller. */
   bool no_entry = false;

   if (cls == S2I_CLASS_COMPUTE) {
      struct radv_shader_stage cs;
      memset(&cs, 0, sizeof(cs));
      cs.stage = MESA_SHADER_COMPUTE;
      cs.spirv.data = (const char *)spirv;
      cs.spirv.size = spirv_words * 4;
      cs.entrypoint = entry;
      cs.key.keep_executable_info = true;
      cs.key.keep_statistic_info = true;
      radv_set_stage_key_robustness(&rs, MESA_SHADER_COMPUTE, &cs.key);
      cs.layout = layout;

      struct radv_shader_debug_info dbg;
      memset(&dbg, 0, sizeof(dbg));
      bin = radv_compile_cs(&ci, &cs, false, &dbg);
      no_entry = cs.nir == NULL;
      /* radv_compile_cs makes the NIR itself, so this is the only owner of it. */
      if (cs.nir)
         ralloc_free(cs.nir);
      s2i_free_debug_info(&dbg);
   } else if (cls == S2I_CLASS_RT) {
      bin = s2i_compile_rt(&ci, spirv, spirv_words, entry, ms, &layout, &rs, info, &no_entry,
                           message);
   } else {
      /* TODO: pipeline->graphics is not consumed here yet. RADV turns a create info into its own
       * key with radv_generate_graphics_state_key, static in radv_pipeline_graphics.c, so it needs
       * a carve export the way radv_physical_device_offline_supported did.
       * Graphics: one unlinked stage, exactly like VK_EXT_shader_object compiles a single stage.
       * The other stages stay MESA_SHADER_NONE; gfx_state carries the shader-object dynamic defaults
       * so the (device-free) graphics compile has a valid key to reason about. */

      /* NGG is a device property, not a per-stage one: a real device sets key.use_ngg from
       * pdev->use_ngg, which is on from GFX10 (Navi14 excepted) and unconditional from GFX11
       * (radv_physical_device.c). With it set, radv_fill_shader_info_ngg picks the NGG stage by
       * RADV's own rules, including the GFX10 fallbacks; with it clear, a vertex shader compiles
       * down the legacy path and ends in a parameter export, which GFX11 replaced with the
       * attribute ring, so the ISA would name an export target the hardware does not have. A stage
       * compiled alone carries no next_stage, so a lone vertex shader is taken for the last vertex
       * stage; which stage actually precedes or follows this one is a declared input. */
      ci.key.use_ngg = s2i_amd_target(target_index)->gfx_level >= GFX10 &&
                       s2i_amd_target(target_index)->family != CHIP_NAVI14;

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
      radv_set_stage_key_robustness(&rs, ms, &stages[ms].key);
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
      no_entry = stages[ms].nir == NULL;

      for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         if (stages[i].nir)
            ralloc_free(stages[i].nir);
         if (binaries[i] && i != ms)
            free(binaries[i]);
         s2i_free_debug_info(&debug[i]);
      }
      free(gs_copy_binary);
      s2i_free_debug_info(&gs_copy_debug);
   }

   if (!bin) {

      if (message && !*message)
         *message = strdup(no_entry ? "vtn could not parse the module for this entrypoint"
                                    : "compile failed (no binary)");

      result = no_entry ? S2I_NO_ENTRYPOINT : S2I_COMPILE_FAILED;
   } else {
      s2i_extract(&ci, bin, stats, isa_text);
      free(bin);
   }

   /* The one way out, success included; the layouts are the session context's, so it frees them. */
done:

   s2i_session_close(&session);
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
                     const uint32_t *spirv, size_t spirv_words, const char *entry,
                     const struct vk_pipeline_robustness_state *rs, uint32_t *payload_size,
                     uint32_t *hit_attrib_size)
{
   struct radv_shader_stage st;
   memset(&st, 0, sizeof(st));
   st.stage = ms;
   st.spirv.data = (const char *)spirv;
   st.spirv.size = spirv_words * 4;
   st.entrypoint = entry;
   radv_set_stage_key_robustness(rs, ms, &st.key);
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
s2i_amd_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count, size_t entry_index,
                            int compile_traversal, int target_index,
                            const s2i_pipeline *pipeline, char **isa_text,
                            s2i_stats_amd *stats, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;
   if (!shaders || shader_count == 0 || target_index < 0 || target_index >= s2i_amd_target_count())
      return S2I_BAD_SPIRV;
   if (!compile_traversal && (entry_index >= shader_count || shaders[entry_index].stage != S2I_STAGE_RAYGEN)) {
      if (message)
         *message = strdup("whole-pipeline entry shader must be a raygen");
      return S2I_BAD_SPIRV;
   }

   /* The same two gates s2i_compile applies, and every shader has to pass them, not just the entry:
    * the features spirv_to_nir cannot lower abort inside it rather than failing. */
   for (size_t i = 0; i < shader_count; i++) {
      if (!s2i_module_args_ok(shaders[i].spirv, shaders[i].spirv_words, shaders[i].entry,
                              shaders[i].stage))
         return S2I_BAD_SPIRV;
      const char *bad = s2i_first_unsupported_ext(shaders[i].spirv, shaders[i].spirv_words,
                                                  s2i_amd_unsupported_exts,
                                                  ARRAY_SIZE(s2i_amd_unsupported_exts));
      if (bad) {
         if (message) {
            char buf[128];
            snprintf(buf, sizeof(buf), "unsupported SPIR-V extension for AMD/RADV: %s", bad);
            *message = strdup(buf);
         }
         return S2I_UNSUPPORTED_CAP;
      }
   }

   struct s2i_session session;

   if (!s2i_session_open(&session))
      return S2I_COMPILE_FAILED;

   const struct s2i_target_desc *t = s2i_amd_target(target_index);
   struct radeon_info rad;
   struct radv_compiler_info ci;
   struct vk_debug_report report;
   bool rt_exposed = false;
   s2i_setup_compiler_info(t, &rad, &ci, &report, &rt_exposed);

   if (!rt_exposed) {

      if (message) {
         char buf[160];
         snprintf(buf, sizeof(buf),
                  "%s has no ray tracing on RADV, so it has no pipeline to compile",
                  s2i_amd_target_name(target_index));
         *message = strdup(buf);
      }

      s2i_session_close(&session);
      return S2I_UNSUPPORTED_CAP;
   }

   /* One robustness for the whole pipeline, as on a device. */
   const struct vk_pipeline_robustness_state rs = s2i_robustness_state(pipeline->robustness);

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

   struct radv_shader_layout layout;
   memset(&layout, 0, sizeof(layout));

   if (!rt_stages || !groups) {
      result = S2I_COMPILE_FAILED;
      goto cleanup;
   }

   if (!((pipeline->set_layouts && pipeline->set_layout_count)
            ? s2i_build_fed_layout(pipeline->set_layouts, pipeline->set_layout_count, &layout,
                                   session.mem_ctx)
            : s2i_build_generic_layout(&layout, session.mem_ctx))) {
      if (message)
         *message = strdup("descriptor layout allocation failed");
      result = S2I_COMPILE_FAILED;
      goto cleanup;
   }

   /* Front half for every shader: NIR + payload/attrib sizing; serialize callees into stage handles;
    * build the groups. Each shader becomes one group (general for raygen/miss/callable, a hit group
    * for hit shaders); distinct synthetic handle pointers let the monolithic inliner switch on them. */
   for (size_t i = 0; i < shader_count; i++) {
      const mesa_shader_stage ms = s2i_stage_to_mesa(shaders[i].stage);
      nir_shader *nir = s2i_rt_shader_to_nir(&ci, &layout, ms, shaders[i].spirv,
                                             shaders[i].spirv_words, shaders[i].entry, &rs,
                                             &payload_size, &hit_attrib_size);
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
         radv_set_stage_key_robustness(&rs, ms, &entry_st.key);
         entry_st.layout = layout;
         entry_st.nir = nir;
      } else {
         ralloc_free(nir); /* callee NIR (and, for traversal, every shader) lives on via its handle */
      }
   }

   struct radv_ray_tracing_pipeline rt_pipeline;
   memset(&rt_pipeline, 0, sizeof(rt_pipeline));
   rt_pipeline.stages = rt_stages;
   rt_pipeline.stage_count = (unsigned)shader_count;
   rt_pipeline.groups = groups;
   rt_pipeline.group_count = (unsigned)shader_count;

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
         radv_build_traversal_shader(&ci, &rt_pipeline, &tinfo, NULL, payload_size,
                                     hit_attrib_size);
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
      radv_set_stage_key_robustness(&rs, MESA_SHADER_INTERSECTION, &tst.key);
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
      radv_nir_lower_rt_abi_functions(tst.nir, &tst.info, payload_size, hit_attrib_size, &ci,
                                      &rt_pipeline);
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
       * inline the callee shaders (from rt_pipeline.stages[]) instead of becoming SBT calls. */
      radv_nir_lower_rt_io_monolithic(entry_st.nir);
      nir_shader_gather_info(entry_st.nir, nir_shader_get_entrypoint(entry_st.nir));
      radv_nir_shader_info_init(entry_st.stage, MESA_SHADER_NONE, &entry_st.info);
      radv_nir_shader_info_pass(&ci, entry_st.nir, &entry_st.layout, &entry_st.key, NULL,
                                RADV_PIPELINE_RAY_TRACING, false, &entry_st.info);
      radv_declare_shader_args(&ci, NULL, &entry_st, MESA_SHADER_NONE, &dbg);
      entry_st.info.user_sgprs_locs = entry_st.args.user_sgprs_locs;
      entry_st.info.inline_push_constant_mask = entry_st.args.ac.inline_push_const_mask;
      entry_st.info.type = RADV_SHADER_TYPE_DEFAULT;

      radv_nir_lower_rt_abi_monolithic(entry_st.nir, &ci, &rt_pipeline);

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

   s2i_free_debug_info(&dbg);

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
   s2i_session_close(&session);
   return result;
}

/*
 * Caller passes the SpvCapability values it already knows the module uses (OxC3 has them from its
 * reflection). We return the ones that need a newer AMD generation than `target_index` supports on RADV.
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
s2i_amd_unsupported_caps(const uint32_t *caps_used, size_t caps_count, int target_index)
{
   if (!caps_used || !caps_count || target_index < 0 || target_index >= s2i_amd_target_count())
      return NULL;

   const enum amd_gfx_level gfx = s2i_amd_target(target_index)->gfx_level;
   const enum radeon_family family = s2i_amd_target(target_index)->family;

   char buf[2048];
   size_t n = 0;
   buf[0] = '\0';

   for (size_t i = 0; i < caps_count; i++) {
      const char *name = NULL;
      for (size_t g = 0; g < ARRAY_SIZE(s2i_cap_gates); g++) {
         if (caps_used[i] == s2i_cap_gates[g].cap) {
      /* GFX1013 is GFX10 with ray tracing hardware, predating the GFX10_3 floor the RT rows use. */
      if (family == CHIP_GFX1013 &&
          (s2i_cap_gates[i].cap == SpvCapabilityRayQueryKHR ||
           s2i_cap_gates[i].cap == SpvCapabilityRayTracingKHR ||
           s2i_cap_gates[i].cap == SpvCapabilityRayTraversalPrimitiveCullingKHR))
         continue;

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
