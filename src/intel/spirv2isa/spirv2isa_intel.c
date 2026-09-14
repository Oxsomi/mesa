/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa (Intel): standalone SPIR-V -> Intel EU ISA via Mesa's brw compiler, no device.
 * See spirv2isa_intel.h. The brw compiler is a pure function of intel_device_info + NIR, so unlike
 * RADV we do not need to fold into the driver: we link libintel_compiler (brw) and drive it directly
 * (blorp shows the shape). intel_get_device_info_from_pci_id gives a device-free devinfo.
 */

#include "spirv2isa_intel.h"

#include <stdarg.h>                       /* va_list for the perf-log sink below */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>                       /* dup/dup2/close for the ISA-capture stderr redirect */

#include "util/ralloc.h"
#include "util/macros.h"
#include "util/bitset.h"                  /* BITSET_TEST/SET/CLEAR on the intel_debug global */
#include "dev/intel_device_info.h"        /* intel_get_device_info_from_pci_id */
#include "dev/intel_debug.h"              /* process_intel_debug_variable (inits INTEL_SIMD/DEBUG globals) */
#include "compiler/shader_enums.h"        /* mesa_shader_stage / MESA_SHADER_* */
#include "compiler/nir/nir.h"             /* nir_shader, address formats, gather_info */
#include "compiler/spirv/nir_spirv.h"     /* spirv_to_nir + options */
#include "compiler/spirv/spirv_info.h"    /* struct spirv_capabilities */
#include "brw_compiler.h"                  /* brw_compiler_create, brw_compile, prog_data/key */
#include "brw_reg.h"                       /* REG_SIZE, the push-constant granularity */
#include "brw_nir.h"                       /* brw_preprocess_nir, brw_nir_lower_cs_intrinsics, brw_nir_lower_cmat */
#include "brw_nir_rt.h"                    /* brw_nir_lower_ray_queries (inline ray tracing) */
#include "anv_nir.h"                       /* ANV's descriptor, driver-value and multiview passes */
#include "anv_shader.h"                     /* anv_shader_data + the offline entry points */
#include "vk_nir.h"                        /* vk_spirv_to_nir, the runtime's SPIR-V front door */
#include "vk_pipeline.h"                   /* vk_pipeline_robustness_state */
#include "spirv2isa_state.h"       /* the pipeline state a module does not carry */
#include "spirv2isa_session.h"     /* the generic setup and its one teardown */
#include "spirv2isa_layout.h"      /* the descriptor layout resolve every backend shares */
#include "util/simple_mtx.h"
#include "spirv2isa_stage.h"
#include "spirv2isa_caps.h"          /* the declared-capability gate every backend shares */
#include "spirv2isa_intel_anv.h"           /* the inputs those passes read, built without a device */

/* The ISA capture mutates process globals (the intel_debug bitset and fd 2), so captures are
 * serialized; concurrent compiles would otherwise interleave their disassembly. */
static simple_mtx_t s2i_intel_capture_mtx = SIMPLE_MTX_INITIALIZER;

struct s2i_intel_target_desc {
   int pci_id;
   const char *name;
};

static const struct s2i_intel_target_desc s2i_intel_targets[S2I_TARGET_INTEL_COUNT] = {
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GEN9_SKL)]   = { 0x1912, "Gen9 Skylake (SKL GT2)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GEN11_ICL)]  = { 0x8a52, "Gen11 Ice Lake (ICL GT2)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_GEN12_TGL)]  = { 0x9a49, "Xe-LP Tiger Lake (TGL GT2)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE_HPG_DG2)] = { 0x56a0, "Xe-HPG DG2 (Arc A770)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE_MTL)]     = { 0x7d40, "Xe-LPG Meteor Lake (MTL)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE2_LNL)]    = { 0x64a0, "Xe2 Lunar Lake (LNL)" },
   [S2I_TARGET_INDEX_OF(S2I_TARGET_XE2_BMG)]    = { 0xe20b, "Xe2 Battlemage (BMG G21, Arc B580)" },
};

/*
 * The extended tier: every Gen9+ PCI id in Mesa's own probe database (the same X-macro list the
 * driver builds its device table from), so new ids appear with a Mesa update rather than an edit
 * here. The build filter drops preproduction (force-probe) ids, ids that predate the backend's
 * Gen9 floor, and ids a flagship row already names; the info-fill check is defensive, since the
 * ids come from that fill's own table. Tokens are the PCI id itself. Built
 * once on first use, unsynchronized like the library's other process-global state.
 */
static const struct {
   int pci_id;
   const char *name;
   int force_probe; /* preproduction id the driver refuses without INTEL_FORCE_PROBE; not listed */
} s2i_intel_pci_db[] = {
#define FORCE_PROBE 1
#define CHIPSET(id, family, fam_str, name, ...) { id, name, __VA_ARGS__ + 0 },
#include "pci_ids/iris_pci_ids.h"
#undef CHIPSET
#undef FORCE_PROBE
};

static struct {
   struct s2i_intel_target_desc row;
   char token[8];
} s2i_intel_extended[ARRAY_SIZE(s2i_intel_pci_db)];
static int s2i_intel_extended_count = -1;

static void
s2i_intel_build_extended(void)
{
   if (s2i_intel_extended_count >= 0)
      return;

   int n = 0;

   for (size_t i = 0; i < ARRAY_SIZE(s2i_intel_pci_db); ++i) {

      struct intel_device_info devinfo;
      int flagship = 0;

      for (int j = 0; j < (int)S2I_TARGET_INTEL_COUNT; ++j)
         flagship |= s2i_intel_targets[j].pci_id == s2i_intel_pci_db[i].pci_id;

      if (flagship || s2i_intel_pci_db[i].force_probe ||
          !intel_get_device_info_from_pci_id(s2i_intel_pci_db[i].pci_id, &devinfo) || devinfo.ver < 9)
         continue;

      s2i_intel_extended[n].row.pci_id = s2i_intel_pci_db[i].pci_id;
      s2i_intel_extended[n].row.name = s2i_intel_pci_db[i].name;
      snprintf(s2i_intel_extended[n].token, sizeof(s2i_intel_extended[n].token), "0x%04x",
               (unsigned)s2i_intel_pci_db[i].pci_id);
      ++n;
   }

   s2i_intel_extended_count = n;
}

/* Row for a target index across both tiers, NULL when the index is out of range. */
static const struct s2i_intel_target_desc *
s2i_intel_target(int target_index)
{
   if (target_index >= 0 && target_index < (int)S2I_TARGET_INTEL_COUNT)
      return &s2i_intel_targets[target_index];

   s2i_intel_build_extended();

   if (target_index >= (int)S2I_TARGET_INTEL_COUNT &&
       target_index < (int)S2I_TARGET_INTEL_COUNT + s2i_intel_extended_count)
      return &s2i_intel_extended[target_index - S2I_TARGET_INTEL_COUNT].row;

   return NULL;
}

int
s2i_intel_target_count(void)
{
   s2i_intel_build_extended();
   return (int)S2I_TARGET_INTEL_COUNT + s2i_intel_extended_count;
}

const char *
s2i_intel_target_token(int target_index)
{
   s2i_intel_build_extended();

   if (target_index < (int)S2I_TARGET_INTEL_COUNT ||
       target_index >= (int)S2I_TARGET_INTEL_COUNT + s2i_intel_extended_count)
      return "";

   return s2i_intel_extended[target_index - S2I_TARGET_INTEL_COUNT].token;
}

/* brw calls compiler->shader_{debug,perf}_log during codegen with no NULL check, so both must exist.
 * The debug one is dropped: it prints the same numbers brw writes into genisa_stats a few lines
 * later, which the stats below read from the struct instead. */
static void
s2i_intel_log_noop(void *data, unsigned *id, const char *fmt, ...)
{
   (void)data;
   (void)id;
   (void)fmt;
}

/* Where the perf log accumulates, for the length of one compile. */
struct s2i_intel_log_sink {
   char *notes;   /* on the session context, never NULL once opened */
};

/* The perf log is worth keeping: it is the only place brw says why it rejected a WIDER SIMD variant
 * than the one it shipped, and when a shader spilled. A total failure already reaches the caller
 * through params.base.error_str, but a partial one (SIMD32 refused, SIMD16 compiled) is invisible
 * otherwise, and it is the answer to "why did I get this dispatch width". */
static void
s2i_intel_perf_log(void *data, unsigned *id, const char *fmt, ...)
{
   struct s2i_intel_log_sink *sink = (struct s2i_intel_log_sink *)data;

   (void)id;

   if (!sink || !sink->notes)
      return;

   va_list args;
   va_start(args, fmt);
   ralloc_vasprintf_append(&sink->notes, fmt, args);
   va_end(args);
}

const char *
s2i_intel_target_name(int target_index)
{
   const struct s2i_intel_target_desc *t = s2i_intel_target(target_index);

   return t ? t->name : "";
}

/* Every intrinsic that must be gone before brw sees the shader. brw_from_nir has no default case, so
 * one left here aborts with no explanation; naming it makes that an ordinary refusal instead. */
static bool
s2i_intel_is_unlowered_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_constant:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_heap_load:
   case nir_intrinsic_image_heap_store:
   case nir_intrinsic_image_heap_atomic:
   case nir_intrinsic_image_heap_atomic_swap:
   case nir_intrinsic_image_heap_size:
   case nir_intrinsic_image_heap_samples:
   case nir_intrinsic_image_heap_sparse_load:
      return true;
   default:
      return false;
   }
}

/* Name of the first resource intrinsic nothing lowered, or NULL when the shader is ready for brw. */
static const char *
s2i_intel_first_unlowered(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;

            if (s2i_intel_is_unlowered_intrinsic(op))
               return nir_intrinsic_infos[op].name;
         }
      }
   }

   return NULL;
}

s2i_result
s2i_intel_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
                  int target_index, const s2i_pipeline *pipeline, char **isa_text,
                  s2i_stats_intel *stats, s2i_info *info, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;

   if (!s2i_module_args_ok(spirv, spirv_words, entry, stage) || target_index < 0 ||
       target_index >= s2i_intel_target_count())
      return S2I_BAD_SPIRV;

   /* Initialize the INTEL_DEBUG / INTEL_SIMD globals a driver would set at startup; without this the
    * SIMD-width selection reads every width as disabled and nothing compiles. call_once-guarded. */
   process_intel_debug_variable();

   struct s2i_session session;

   if (!s2i_session_open(&session))
      return S2I_COMPILE_FAILED;

   /* Everything the one teardown below releases, declared before the first jump to it. */
   s2i_result result = S2I_OK;
   FILE *cap = NULL;
   s2i_intel_anv_layouts layouts;
   bool layouts_built = false;

   /* Device-free devinfo from a PCI id, then the brw compiler from just that. */
   struct intel_device_info devinfo;
   if (!intel_get_device_info_from_pci_id(s2i_intel_target(target_index)->pci_id, &devinfo)) {

      if (message)
         *message = strdup("intel_get_device_info_from_pci_id failed");

      result = S2I_BAD_TARGET;
      goto done;
   }

   /* Normally filled by the kernel from a DRM query, and left 0 by the PCI-id path. It gates brw's
    * load/store vectorizer, so it decides how many loads merge and therefore the instruction count.
    * Derived the way the kernel derives it (i915/intel_device_info.c). */
   if (devinfo.mem_alignment == 0)
      devinfo.mem_alignment = (devinfo.verx10 >= 125 || devinfo.has_local_mem) ? 64 * 1024 : 4096;

   struct brw_compiler *compiler = brw_compiler_create(session.mem_ctx, &devinfo);

   /* Must exist before spirv_to_nir: the address formats below are derived from it. */
   struct anv_physical_device *pdev = rzalloc(session.mem_ctx, struct anv_physical_device);
   s2i_intel_anv_init_physical_device(pdev, &devinfo, compiler, session.mem_ctx);

   /* Each field flips a UBO/SSBO address format between the offset and bounded forms. */
   const struct vk_pipeline_robustness_state rs = s2i_robustness_state(pipeline->robustness);
   compiler->shader_debug_log = s2i_intel_log_noop;
   compiler->shader_perf_log = s2i_intel_perf_log;

   const mesa_shader_stage ms = s2i_stage_to_mesa(stage);

   /* Only ->physical, the mirrored fields and the object base are read; logging resolves the
    * instance through the base (see the instance setup in spirv2isa_intel_anv.c). */
   struct anv_device *device = rzalloc(session.mem_ctx, struct anv_device);
   device->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   device->vk.base.client_visible = true;
   /* The logger resolves any owned object through base.device, a device's own base included. */
   device->vk.base.device = &device->vk;
   device->physical = pdev;
   device->info = &pdev->info;
   device->isl_dev = pdev->isl_dev;
   device->vk.physical = &pdev->vk;

   /* The runtime's own SPIR-V front door: capabilities computed from the feature tables this pdev
    * filled, the parse, then the finalization every driver-facing module has had (returns lowered,
    * functions inlined, clip and cull distances merged). A foreign module (glslang, -Od) arrives in
    * the same shape DXC output does. */
   struct spirv_to_nir_options spirv_opts =
      anv_shader_offline_spirv_options(&pdev->vk, ms, &rs);
   const struct nir_shader_compiler_options *nir_options =
      anv_shader_offline_nir_options(&pdev->vk, ms, &rs);

   /* Every capability the module declares, against the set this target really has. */
   const struct spirv_capabilities target_caps =
      vk_physical_device_get_spirv_capabilities(&pdev->vk);

   {
      bool malformed = false;
      char *refusal = s2i_gate_declared_capabilities(spirv, spirv_words, &target_caps,
                                                     s2i_intel_target(target_index)->name, &malformed);

      if (refusal) {

         if (message)
            *message = refusal;
         else
            free(refusal);

         result = malformed ? S2I_BAD_SPIRV : S2I_UNSUPPORTED_CAP;
         goto done;
      }
   }

   nir_shader *nir = vk_spirv_to_nir(&device->vk, spirv, spirv_words * 4, ms, entry, NULL,
                                     &spirv_opts, nir_options, false, session.mem_ctx);

   if (!nir) {
      if (message)
         *message = strdup("vk_spirv_to_nir failed (bad entrypoint, or a capability this target "
                           "does not support; diagnostics on stderr)");
      result = S2I_NO_ENTRYPOINT;
      goto done;
   }

   /* vk_pipeline.c's vk_set_subgroup_size, with the sizes ANV reports (get_properties_1_3): the
    * parser's 1..128 defaults would otherwise reach anv_fixup_subgroup_size, which pins cooperative
    * matrix shaders to a width the hardware does not have. A required subgroup size stays a future
    * declared input. */
   if (mesa_shader_stage_uses_workgroup(ms)) {

      if (spirv[1] < 0x10600) {
         /* Before SPIR-V 1.6 the api subgroup size is the device's, not "varying". */
         nir->info.api_subgroup_size = 32;
         nir->info.max_subgroup_size = 32;
      }

      nir->info.max_subgroup_size = MIN2(nir->info.max_subgroup_size, 32);
      nir->info.min_subgroup_size = MAX2(nir->info.min_subgroup_size, devinfo.ver >= 20 ? 16 : 8);
   }

   /* ANV's preprocess hook: sysvals to varyings, access flags, brw's preprocessing, barrier modes. */
   anv_shader_offline_preprocess(&pdev->vk, nir, &rs);

   const VkDescriptorSetLayoutCreateInfo *const *eff_layouts = NULL;
   uint32_t eff_layout_count = 0;

   result = s2i_resolve_layouts(nir, S2I_INTEL_ANV_MAX_SETS, pipeline->set_layouts,
                                pipeline->set_layout_count, session.mem_ctx, &eff_layouts,
                                &eff_layout_count, message);

   if (result != S2I_OK)
      goto done;

   if (!s2i_intel_anv_build_layouts(pdev, eff_layouts, eff_layout_count, &layouts, message)) {
      result = S2I_UNSUPPORTED_CAP;
      goto done;
   }

   layouts_built = true;

   struct vk_shader_compile_info compile_info;
   memset(&compile_info, 0, sizeof(compile_info));
   compile_info.stage = ms;
   compile_info.nir = nir;
   compile_info.set_layouts = (struct vk_descriptor_set_layout **) layouts.sets;
   compile_info.set_layout_count = layouts.set_count;

   compile_info.robustness = &rs;

   struct anv_shader_data shader_data;
   memset(&shader_data, 0, sizeof(shader_data));
   shader_data.info = &compile_info;

   /* ANV's own key population. The link mask models the classic pipeline around this stage, which is
    * what this tool models: both tessellation stages linked, a fragment stage present, mesh pipelines
    * modeled only when compiling their own stages. The day s2i_compile carries graphics pipeline
    * state, the NULL state and this mask become caller data (view mask, mesh presence, coarse pixel,
    * sample counts all live there). */
   VkShaderStageFlags link_stages = 0;

   switch (ms) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_FRAGMENT:
      link_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT;
      break;
   case MESA_SHADER_TASK:
   case MESA_SHADER_MESH:
      link_stages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                    VK_SHADER_STAGE_FRAGMENT_BIT;
      break;
   default:
      break;
   }

   struct vk_graphics_pipeline_all_state gfx_all;
   struct vk_graphics_pipeline_state gfx_state;
   const struct vk_graphics_pipeline_state *gfx = NULL;

   if (pipeline->graphics) {

      if (!s2i_graphics_state(&device->vk, pipeline->graphics, &gfx_all, &gfx_state, message)) {
         result = S2I_UNSUPPORTED_CAP;
         goto done;
      }

      gfx = &gfx_state;
   }

   anv_shader_offline_populate_key(&pdev->vk, &shader_data, gfx, link_stages);

   if (info)
      info->graphics_specialized = gfx != NULL;

   struct brw_stage_prog_data *base_prog_data;
   struct brw_base_prog_key *base_prog_key;

   if (ms == MESA_SHADER_COMPUTE) {
      base_prog_data = &shader_data.prog_data.cs.base;
      base_prog_key = &shader_data.key.cs.base;
   } else if (ms == MESA_SHADER_FRAGMENT) {
      base_prog_data = &shader_data.prog_data.fs.base;
      base_prog_key = &shader_data.key.fs.base;
   } else if (s2i_stage_is_rt(ms)) {
      /* All six RT stages compile through brw_compile_bs; brw_bs_prog_data nests the stage base
       * directly (like fragment, unlike the VUE stages). */
      base_prog_data = &shader_data.prog_data.bs.base;
      base_prog_key = &shader_data.key.bs.base;
   } else if (ms == MESA_SHADER_TASK || ms == MESA_SHADER_MESH) {
      /* Task/mesh prog_data nest brw_cs_prog_data (they dispatch like compute), so the stage base is
       * two levels down. Unlike compute, brw_compile_task/mesh run brw_nir_lower_cs_intrinsics
       * themselves, so we must not do it here. A mesh shader compiled on its own gets tue_map = NULL,
       * which brw explicitly supports (task and mesh need not be compiled together). */
      base_prog_data = ms == MESA_SHADER_TASK ? &shader_data.prog_data.task.base.base : &shader_data.prog_data.mesh.base.base;
      base_prog_key = ms == MESA_SHADER_TASK ? &shader_data.key.task.base : &shader_data.key.mesh.base;
   } else {
      /* VUE stages (vertex / geometry / tess-ctrl / tess-eval): the prog_data nests
       * brw_vue_prog_data -> brw_stage_prog_data; the union members overlap at offset 0, so the vs
       * view addresses the shared base. brw_compile_* casts to the real per-stage prog_data/key. */
      base_prog_data = &shader_data.prog_data.vs.base.base;
      base_prog_key = &shader_data.key.vs.base;
   }

   /* The lowering reads the stage off prog_data, and runs before brw_compile would set it. */
   base_prog_data->stage = ms;

   /* Assigned outside the lowering, so it stays ours. Zero is UNKNOWN, which is neither DIRECT nor
    * INDIRECT, and almost every site tests for one of those two. */
   shader_data.bind_map.layout_type =
      layouts.set_count > 0 ? layouts.sets[0]->type
                            : ANV_PIPELINE_DESCRIPTOR_SET_LAYOUT_TYPE_DIRECT;

   /* Mesh, task and the RT stages address every resource bindlessly, so ANV gives them no tables at
    * all; the sized ones use ANV's own binding table bound. */
   if (!brw_shader_stage_requires_bindless_resources(ms)) {
      shader_data.bind_map.surface_to_descriptor =
         rzalloc_array(session.mem_ctx, struct anv_pipeline_binding, MAX_BINDING_TABLE_SIZE);
      shader_data.bind_map.sampler_to_descriptor =
         rzalloc_array(session.mem_ctx, struct anv_pipeline_binding, MAX_BINDING_TABLE_SIZE);
   }

   /* The layout apply writes an entry per embedded sampler it places, so the table has to exist for
    * every stage, exactly as ANV sizes it from the layouts. Never zero length: rzalloc_array(0)
    * gives a pointer the pass would still be indexing into. */
   uint32_t embedded_samplers = 0;

   for (uint32_t set = 0; set < layouts.set_count; set++) {
      if (layouts.sets[set])
         embedded_samplers += layouts.sets[set]->embedded_sampler_count;
   }

   shader_data.bind_map.embedded_sampler_to_binding =
      rzalloc_array(session.mem_ctx, struct anv_pipeline_embedded_sampler_binding,
                    MAX2(embedded_samplers, 1));

   /* ANV's lowering run rather than mirrored, so the descriptor and push-constant placement is the
    * driver's own. */
   anv_shader_lower_nir(device, session.mem_ctx, NULL /* graphics pipeline state */, &shader_data);

   /* Seeded rather than left NULL, so every append stays on the session context: ralloc's append
    * allocates against a NULL context when the string is NULL. */
   struct s2i_intel_log_sink sink = { .notes = ralloc_strdup(session.mem_ctx, "") };

   /* Use brw's own params union: every stage's struct starts with the shared .base and the tails
    * differ, so one stage's struct is not a safe superset for another (fragment's max_polygons sits
    * where mesh keeps a wa_18019110168 function pointer). Zero it, fill .base, then only the fields
    * of the stage in hand; what stays zero reads as not supplied. */
   union brw_any_compile_params params;
   memset(&params, 0, sizeof(params));
   params.base.mem_ctx = session.mem_ctx;
   params.base.log_data = &sink;
   params.base.nir = nir;
   params.base.key = base_prog_key;
   params.base.prog_data = base_prog_data;
   /* brw fills the per-variant numbers only into params->stats; prog_data->program_size counts every
    * variant plus the constant data, which is not the shader's size. ANV passes the same array. */
   params.base.stats = shader_data.stats;

   /* ANV's finish: subgroup fixup, shader workarounds, the RT entry lowering and shader-call split,
    * and the per-stage compile params (the TCS key copy, TES and mesh and fragment inputs, the RT
    * resume shaders). The NULL is the intersection shader's any-hit, which becomes caller data when
    * the RT pipeline API carries hit groups. */
   anv_shader_offline_finish(device, &shader_data, NULL, &params, session.mem_ctx);

   /* Refuse by name anything no pass claimed, instead of letting brw abort on an intrinsic it
    * has no case for. A descriptor-heap module reaches here this way. */

   const char *unlowered = s2i_intel_first_unlowered(nir);

   if (unlowered) {

      if (message) {
         char buf[192];
         snprintf(buf, sizeof(buf), "%s is not lowered by this backend yet, so the module cannot be "
                                    "compiled device-free", unlowered);
         *message = strdup(buf);
      }

      result = S2I_UNSUPPORTED_CAP;
      goto done;
   }

   /* Capture the EU disassembly brw prints to stderr under INTEL_DEBUG. We set the stage's disasm bit
    * in the intel_debug global (NOT the NIR bit, so no NIR noise) and redirect stderr to a temp file
    * just around brw_compile, then read it back. This mutates process-global state (the intel_debug
    * bitset + fd 2), so captures are serialized on s2i_intel_capture_mtx; a real brw disasm API would
    * remove the global mutation and the mutex with it. */
   const uint64_t dbg_flag = intel_debug_flag_for_shader_stage(ms);
   const bool dbg_was_set = BITSET_TEST(intel_debug, dbg_flag);
   int saved_stderr = -1;
   bool capture_locked = false;

   if (isa_text && !getenv("S2I_NO_CAPTURE")) {
      simple_mtx_lock(&s2i_intel_capture_mtx);
      capture_locked = true;

      if (!dbg_was_set)
         BITSET_SET(intel_debug, dbg_flag);
      cap = tmpfile();
      if (cap) {
         fflush(stderr);
         saved_stderr = dup(2);
         dup2(fileno(cap), 2);
      }
   }

   const unsigned *program = brw_compile(compiler, &params.base);

   if (isa_text) {
      if (saved_stderr >= 0) {
         fflush(stderr);
         dup2(saved_stderr, 2);
         close(saved_stderr);
      }
      if (!dbg_was_set)
         BITSET_CLEAR(intel_debug, dbg_flag);
   }

   if (capture_locked)
      simple_mtx_unlock(&s2i_intel_capture_mtx);

   /* Handed over before the failure check: what brw remarked on explains a failed compile as often
    * as a successful one. */
   if (info && sink.notes && sink.notes[0])
      info->notes = strdup(sink.notes);

   if (!program) {
      if (message)
         *message = strdup(params.base.error_str ? params.base.error_str : "brw_compile failed");
      result = S2I_COMPILE_FAILED;
      goto done;
   }

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      /* The widest variant brw emitted. Variants are appended narrowest first and each fills the
       * next stats slot in order, so the last filled slot is the widest. Nothing picks it: a
       * fragment program can carry several at once for the hardware to choose between, so these
       * numbers describe one variant rather than the whole program. */
      const struct genisa_stats *widest = NULL;

      for (unsigned i = 0; i < ARRAY_SIZE(shader_data.stats); i++) {
         if (shader_data.stats[i].dispatch_width != 0)
            widest = &shader_data.stats[i];
      }

      /* brw only fills grf_registers from Xe3 on (brw_to_binary.cpp), so that one always comes off
       * prog_data; the rest describe the variant and are only in the stats. */
      stats->grf_used = base_prog_data->grf_used;

      if (widest) {
         stats->program_size = widest->code_size;
         stats->scratch_size = widest->scratch_memory_size;
         stats->shared_size = widest->workgroup_memory_size;
         stats->simd_width = widest->dispatch_width;

         /* Only the variant carries these, so a stage brw reported no variant for keeps them 0. */
         stats->instrs = widest->instrs;
         stats->cycles = widest->cycles;
         stats->spills = widest->spills;
         stats->fills = widest->fills;
         stats->sends = widest->sends;
         stats->loops = widest->loops;
         stats->max_live_registers = widest->max_live_registers;
      } else {
         stats->program_size = base_prog_data->program_size;
         stats->scratch_size = base_prog_data->total_scratch;
         stats->shared_size = base_prog_data->total_shared;
      }

      /* Dispatch width. Compute, task and mesh all report a valid-SIMD-variant mask (bit0=8, bit1=16,
       * bit2=32) in the brw_cs_prog_data they share; the ray tracing stages carry a plain width in
       * brw_bs_prog_data instead. The fixed-function graphics stages have no single dispatch width, so
       * they keep 0. */
      if (stats->simd_width != 0) {
         /* brw already reported it */
      } else if (ms == MESA_SHADER_COMPUTE || ms == MESA_SHADER_TASK || ms == MESA_SHADER_MESH) {
         const unsigned m = ms == MESA_SHADER_COMPUTE ? shader_data.prog_data.cs.prog_mask
                          : ms == MESA_SHADER_TASK    ? shader_data.prog_data.task.base.prog_mask
                                                      : shader_data.prog_data.mesh.base.prog_mask;
         stats->simd_width = (m & 4) ? 32 : (m & 2) ? 16 : (m & 1) ? 8 : 0;
      } else if (s2i_stage_is_rt(ms)) {
         stats->simd_width = shader_data.prog_data.bs.simd_size;
      }

      /* Ray tracing runs on a per-ray stack rather than scratch, accumulated across the main shader
       * and its resume shaders, so scratch alone makes an RT shader look free. */
      if (s2i_stage_is_rt(ms))
         stats->stack_size = shader_data.prog_data.bs.max_stack_size;
   }

   if (cap) {
      fseek(cap, 0, SEEK_END);
      long sz = ftell(cap);
      fseek(cap, 0, SEEK_SET);
      if (isa_text && sz > 0) {
         char *buf = (char *)malloc(sz + 1);
         if (buf) {
            size_t rd = fread(buf, 1, sz, cap);
            buf[rd] = '\0';
            /* brw prints the NIR listing before the "Native code" disassembly under the same stage
             * debug flag; keep only the machine code (from the first "Native code" header). */
            const char *asm_start = strstr(buf, "Native code");
            *isa_text = strdup(asm_start ? asm_start : buf);
            free(buf);
         }
      }
      fclose(cap);
      cap = NULL;
   }

   /* The one way out, success included. Nothing below is set until the step that produces it ran. */
done:

   if (cap)
      fclose(cap);

   if (layouts_built)
      s2i_intel_anv_free_layouts(&layouts);

   s2i_session_close(&session);
   return result;
}
