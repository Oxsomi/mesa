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
#include "compiler/glsl_types.h"          /* glsl_type_singleton_init_or_ref/decref (no device inits it) */
#include "compiler/nir/nir.h"             /* nir_shader, address formats, gather_info */
#include "compiler/nir/nir_builder.h"     /* nir builder for the buffer-descriptor lowering */
#include "compiler/spirv/nir_spirv.h"     /* spirv_to_nir + options */
#include "compiler/spirv/spirv_info.h"    /* struct spirv_capabilities */
#include "brw_compiler.h"                  /* brw_compiler_create, brw_compile, prog_data/key */
#include "brw_nir.h"                       /* brw_preprocess_nir, brw_nir_lower_cs_intrinsics */

struct s2i_intel_target_desc {
   int pci_id;
   const char *name;
};

static const struct s2i_intel_target_desc s2i_intel_targets[S2I_INTEL_TARGET_COUNT] = {
   [S2I_INTEL_GEN9_SKL]   = { 0x1912, "Gen9 Skylake (SKL GT2)" },
   [S2I_INTEL_GEN11_ICL]  = { 0x8a52, "Gen11 Ice Lake (ICL GT2)" },
   [S2I_INTEL_GEN12_TGL]  = { 0x9a49, "Xe-LP Tiger Lake (TGL GT2)" },
   [S2I_INTEL_XE_HPG_DG2] = { 0x56a0, "Xe-HPG DG2 (Arc A770)" },
   [S2I_INTEL_XE_MTL]     = { 0x7d40, "Xe-LPG Meteor Lake (MTL)" },
   [S2I_INTEL_XE2_LNL]    = { 0x64a0, "Xe2 Lunar Lake (LNL)" },
};

static const mesa_shader_stage s2i_intel_mesa_stage[S2I_INTEL_STAGE_COUNT] = {
   [S2I_INTEL_STAGE_VERTEX]   = MESA_SHADER_VERTEX,
   [S2I_INTEL_STAGE_PIXEL]    = MESA_SHADER_FRAGMENT,
   [S2I_INTEL_STAGE_COMPUTE]  = MESA_SHADER_COMPUTE,
   [S2I_INTEL_STAGE_GEOMETRY] = MESA_SHADER_GEOMETRY,
   [S2I_INTEL_STAGE_HULL]     = MESA_SHADER_TESS_CTRL,
   [S2I_INTEL_STAGE_DOMAIN]   = MESA_SHADER_TESS_EVAL,
};

/* brw calls compiler->shader_{debug,perf}_log during codegen with no NULL check; the driver normally
 * installs these. Without them a plain compile (no INTEL_DEBUG) calls a NULL pointer and crashes. */
static void
s2i_intel_log_noop(void *data, unsigned *id, const char *fmt, ...)
{
   (void)data;
   (void)id;
   (void)fmt;
}

const char *
s2i_intel_target_name(s2i_intel_target target)
{
   if (target < 0 || target >= S2I_INTEL_TARGET_COUNT)
      return "(invalid)";
   return s2i_intel_targets[target].name;
}

/* Device-free buffer-descriptor lowering (UBO/SSBO only). A real driver resolves resources to
 * hardware surfaces (ANV: anv_nir_apply_pipeline_layout + binding tables / bindless + resource_intel
 * annotations, device-coupled). We instead use STATELESS 64-bit global addressing: each
 * vulkan_resource_index becomes a synthetic per-binding base address and nir_lower_explicit_io then
 * turns the buffer access into ordinary global (A64) loads/stores - no surface, no binding table. The
 * addresses are placeholders; the emitted memory instructions (the ISA we care about) are real.
 * Images/samplers still need surface handles and are rejected before here. */
static bool
s2i_intel_lower_buffer_desc(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      unsigned set = nir_intrinsic_desc_set(intr);
      unsigned binding = nir_intrinsic_binding(intr);
      nir_def *base = nir_imm_int64(b, 0x100000000ull + ((uint64_t)set * 64 + binding) * 0x10000ull);
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(b, intr->src[0].ssa, 0x1000)));
      nir_def_replace(&intr->def, addr);
      return true;
   }
   case nir_intrinsic_vulkan_resource_reindex: {
      nir_def *addr = nir_iadd(b, intr->src[0].ssa,
                               nir_u2u64(b, nir_imul_imm(b, intr->src[1].ssa, 0x1000)));
      nir_def_replace(&intr->def, addr);
      return true;
   }
   case nir_intrinsic_load_vulkan_descriptor:
      nir_def_replace(&intr->def, intr->src[0].ssa);
      return true;
   case nir_intrinsic_get_ssbo_size:
      /* Stateless A64 carries no buffer size; give OpArrayLength a synthetic one (placeholder). */
      nir_def_replace(&intr->def, nir_imm_int(b, 0x10000));
      return true;
   case nir_intrinsic_load_base_workgroup_id:
      /* brw does not emit this directly (a driver lowers it); no dispatch base offline, so it is 0. */
      nir_def_replace(&intr->def, nir_imm_ivec3(b, 0, 0, 0));
      return true;
   default:
      return false;
   }
}

/* Device-free STORAGE-IMAGE lowering. brw wants image_load/store/atomic/size with the surface index
 * in src[0] (get_nir_image_intrinsic_image reads it as an immediate), not the image_deref_* form
 * spirv_to_nir emits. A real driver (ANV: anv_nir_apply_pipeline_layout -> nir_rewrite_image_intrinsic)
 * resolves the deref to a binding-table index or a bindless surface handle from the pipeline layout.
 * We synthesize a constant binding-table index (set * 64 + binding, plus any array index) the same way
 * we synthesize buffer addresses: the index is a placeholder, the emitted image message is real.
 * brw_nir_lower_storage_image must run first: it matches image_deref_* (the format side) before we
 * rename them to image_*. Sampled textures (tex instructions) are a separate path, still rejected. */
static bool
s2i_intel_lower_image_desc(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_sparse_load:
      break;
   default:
      return false;
   }

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   uint32_t bti = var->data.descriptor_set * 64 + var->data.binding;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *handle = nir_imm_int(b, bti);
   /* An image array indexes off the base binding; fold the array index into the surface index. */
   if (deref->deref_type == nir_deref_type_array)
      handle = nir_iadd(b, handle, deref->arr.index.ssa);
   nir_rewrite_image_intrinsic(intr, handle, nir_image_intrinsic_type_default);
   return true;
}

/* Device-free SAMPLED-TEXTURE lowering, the tex-instruction analogue of the storage-image pass above.
 * brw wants a texture/sampler_deref replaced by a texture/sampler_offset (a binding-table index added
 * to tex->texture_index) or a *_handle (bindless). ANV builds these from the pipeline layout in
 * lower_tex_deref; we synthesize the same constant binding-table index (set * 64 + binding + array).
 * A combined image sampler resolves both derefs to the same binding; separate ones use their own. */
static bool
s2i_intel_lower_tex_deref(nir_builder *b, nir_tex_instr *tex, nir_tex_src_type deref_type)
{
   int idx = nir_tex_instr_src_index(tex, deref_type);
   if (idx < 0)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[idx].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   uint32_t bti = var->data.descriptor_set * 64 + var->data.binding;

   nir_def *index = nir_imm_int(b, bti);
   if (deref->deref_type == nir_deref_type_array)
      index = nir_iadd(b, index, deref->arr.index.ssa);

   nir_src_rewrite(&tex->src[idx].src, index);
   tex->src[idx].src_type = deref_type == nir_tex_src_texture_deref ? nir_tex_src_texture_offset
                                                                    : nir_tex_src_sampler_offset;
   return true;
}

static bool
s2i_intel_lower_tex(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   bool progress = false;
   progress |= s2i_intel_lower_tex_deref(b, tex, nir_tex_src_texture_deref);
   progress |= s2i_intel_lower_tex_deref(b, tex, nir_tex_src_sampler_deref);
   if (progress) {
      /* The full index now rides in the offset source; the base indices are folded in. */
      tex->texture_index = 0;
      tex->sampler_index = 0;
   }
   return progress;
}

s2i_intel_result
s2i_compile_intel(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_intel_stage stage,
                  s2i_intel_target target, char **isa_text, s2i_intel_stats *stats, char **message)
{
   if (isa_text)
      *isa_text = NULL;
   if (message)
      *message = NULL;

   if (!spirv || spirv_words < 5 || spirv[0] != 0x07230203u)
      return S2I_INTEL_BAD_SPIRV;
   if (!entry || target < 0 || target >= S2I_INTEL_TARGET_COUNT || stage < 0 ||
       stage >= S2I_INTEL_STAGE_COUNT)
      return S2I_INTEL_BAD_SPIRV;

   /* All of compute + vertex + fragment + geometry + tessellation control/eval are wired. The graphics
    * stages run their own input lowering inside brw_compile_*; TES computes its input VUE map itself
    * when we leave it NULL. (Mesh/task are not in the s2i_intel_stage enum yet.) */

   /* Initialize the INTEL_DEBUG / INTEL_SIMD globals a driver would set at startup; without this the
    * SIMD-width selection reads every width as disabled and nothing compiles. call_once-guarded. */
   process_intel_debug_variable();

   void *mem_ctx = ralloc_context(NULL);

   /* No device did this for us; the glsl type system uses a singleton linear allocator. */
   glsl_type_singleton_init_or_ref();

   /* 1. device-free devinfo from a PCI id, then the brw compiler from just that. */
   struct intel_device_info devinfo;
   if (!intel_get_device_info_from_pci_id(s2i_intel_targets[target].pci_id, &devinfo)) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      if (message)
         *message = strdup("intel_get_device_info_from_pci_id failed");
      return S2I_INTEL_BAD_DEVICE;
   }
   struct brw_compiler *compiler = brw_compiler_create(mem_ctx, &devinfo);
   compiler->shader_debug_log = s2i_intel_log_noop;
   compiler->shader_perf_log = s2i_intel_log_noop;

   const mesa_shader_stage ms = s2i_intel_mesa_stage[stage];

   /* 2. SPIR-V -> NIR. Broad capability set so vtn never rejects a feature the module declares; a
    * feature the HW cannot do still fails later in brw. No debug callback (device would own it). */
   struct spirv_capabilities caps;
   memset(&caps, 0xff, sizeof(caps));

   struct spirv_to_nir_options spirv_opts;
   memset(&spirv_opts, 0, sizeof(spirv_opts));
   spirv_opts.capabilities = &caps;
   /* 64-bit global addressing for UBO/SSBO: the buffer-descriptor lowering turns each resource into a
    * synthetic address and nir_lower_explicit_io then emits stateless A64 global loads/stores. */
   spirv_opts.ubo_addr_format = nir_address_format_64bit_global;
   spirv_opts.ssbo_addr_format = nir_address_format_64bit_global;
   spirv_opts.phys_ssbo_addr_format = nir_address_format_64bit_global;
   spirv_opts.push_const_addr_format = nir_address_format_logical;
   spirv_opts.shared_addr_format = nir_address_format_32bit_offset;

   const nir_shader_compiler_options *nir_options = &compiler->nir_options[ms];
   nir_shader *nir =
      spirv_to_nir(spirv, spirv_words, NULL, ms, entry, &spirv_opts, nir_options);
   if (!nir) {
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      if (message)
         *message = strdup("spirv_to_nir failed (bad entrypoint or unsupported module)");
      return S2I_INTEL_NO_ENTRYPOINT;
   }
   ralloc_steal(mem_ctx, nir);
   nir->info.stage = ms;

   /* Descriptors are all lowered device-free with synthetic indices: buffers (UBO/SSBO) via stateless
    * A64 global addresses, storage images and sampled textures via synthetic binding-table indices. */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Lower UBO/SSBO descriptors to synthetic global addresses, then to stateless A64 global access,
    * then clean up so brw sees well-formed load/store_global intrinsics. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, s2i_intel_lower_buffer_desc, nir_metadata_none, NULL);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_dce);

   /* Storage images: brw_nir_lower_storage_image does the format side on the deref form (typed-load
    * emulation etc), then we rename image_deref_* -> image_* with a synthetic binding-table index. */
   if (nir->info.num_images) {
      NIR_PASS(_, nir, brw_nir_lower_storage_image, compiler,
               &(struct brw_nir_lower_storage_image_opts){
                  .lower_loads = true,
                  .lower_stores_64bit = true,
               });
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, s2i_intel_lower_image_desc, nir_metadata_none, NULL);
   }

   /* 3. brw preprocessing (common); the stage-specific input lowering runs inside brw_compile_* for
    * graphics, while compute needs brw_nir_lower_cs_intrinsics done here first (brw asserts it). */
   struct brw_nir_compiler_opts opts;
   memset(&opts, 0, sizeof(opts));
   brw_preprocess_nir(compiler, nir, &opts);

   /* Sampled textures: rename tex texture/sampler_deref -> texture/sampler_offset (synthetic binding
    * table index). ANV lowers these after brw_preprocess (derefs still present here); a no-op when the
    * shader has no tex instructions. Storage images were already renamed before preprocess. */
   NIR_PASS(_, nir, nir_shader_instructions_pass, s2i_intel_lower_tex, nir_metadata_none, NULL);

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Per-stage prog_data + key. brw_compile dispatches by nir->info.stage; brw_compile_cs/vs params
    * are just { base }, so a bare brw_compile_params is a valid argument for both. */
   union {
      struct brw_cs_prog_data cs;
      struct brw_vs_prog_data vs;
      struct brw_fs_prog_data fs;
      struct brw_gs_prog_data gs;
      struct brw_tcs_prog_data tcs;
      struct brw_tes_prog_data tes;
   } prog_data;
   memset(&prog_data, 0, sizeof(prog_data));

   union {
      struct brw_cs_prog_key cs;
      struct brw_vs_prog_key vs;
      struct brw_fs_prog_key fs;
      struct brw_gs_prog_key gs;
      struct brw_tcs_prog_key tcs;
      struct brw_tes_prog_key tes;
   } prog_key;
   memset(&prog_key, 0, sizeof(prog_key));

   struct brw_stage_prog_data *base_prog_data;
   struct brw_base_prog_key *base_prog_key;

   if (ms == MESA_SHADER_COMPUTE) {
      brw_nir_lower_cs_intrinsics(nir, &devinfo, &prog_data.cs);
      /* brw_nir_lower_cs_intrinsics introduces load_base_workgroup_id; lower it now (blorp does the
       * same right after this call). Re-running the pass only matches that intrinsic here. */
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, s2i_intel_lower_buffer_desc, nir_metadata_none, NULL);
      base_prog_data = &prog_data.cs.base;
      base_prog_key = &prog_key.cs.base;
   } else if (ms == MESA_SHADER_FRAGMENT) {
      base_prog_data = &prog_data.fs.base;
      base_prog_key = &prog_key.fs.base;
   } else {
      /* VUE stages (vertex / geometry / tess-ctrl / tess-eval): the prog_data nests
       * brw_vue_prog_data -> brw_stage_prog_data; the union members overlap at offset 0, so the vs
       * view addresses the shared base. brw_compile_* casts to the real per-stage prog_data/key. */
      base_prog_data = &prog_data.vs.base.base;
      base_prog_key = &prog_key.vs.base;
   }

   /* 4. compile to EU ISA. brw_compile_fs_params is a superset (fragment adds vue_map/max_polygons);
    * its .base sits at offset 0, so &fs_params.base is a valid brw_compile_params for every stage and
    * brw_compile only touches the fragment fields when the stage actually is fragment. */
   struct brw_compile_fs_params fs_params;
   memset(&fs_params, 0, sizeof(fs_params));
   fs_params.max_polygons = 1; /* one polygon per dispatch; only consulted for fragment */
   fs_params.base.mem_ctx = mem_ctx;
   fs_params.base.nir = nir;
   fs_params.base.key = base_prog_key;
   fs_params.base.prog_data = base_prog_data;

   /* Capture the EU disassembly brw prints to stderr under INTEL_DEBUG. We set the stage's disasm bit
    * in the intel_debug global (NOT the NIR bit, so no NIR noise) and redirect stderr to a temp file
    * just around brw_compile, then read it back. This mutates process-global state (the intel_debug
    * bitset + fd 2), so it is not thread-safe; fine for the CLI, worth a real brw disasm API later. */
   const uint64_t dbg_flag = intel_debug_flag_for_shader_stage(ms);
   const bool dbg_was_set = BITSET_TEST(intel_debug, dbg_flag);
   FILE *cap = NULL;
   int saved_stderr = -1;
   if (isa_text && !getenv("S2I_NO_CAPTURE")) {
      if (!dbg_was_set)
         BITSET_SET(intel_debug, dbg_flag);
      cap = tmpfile();
      if (cap) {
         fflush(stderr);
         saved_stderr = dup(2);
         dup2(fileno(cap), 2);
      }
   }

   const unsigned *program = brw_compile(compiler, &fs_params.base);

   if (isa_text) {
      if (saved_stderr >= 0) {
         fflush(stderr);
         dup2(saved_stderr, 2);
         close(saved_stderr);
      }
      if (!dbg_was_set)
         BITSET_CLEAR(intel_debug, dbg_flag);
   }

   if (!program) {
      if (message)
         *message = strdup(fs_params.base.error_str ? fs_params.base.error_str : "brw_compile failed");
      if (cap)
         fclose(cap);
      glsl_type_singleton_decref();
      ralloc_free(mem_ctx);
      return S2I_INTEL_COMPILE_FAILED;
   }

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->grf_used = base_prog_data->grf_used;
      stats->program_size = base_prog_data->program_size;
      stats->scratch_size = base_prog_data->total_scratch;
      stats->shared_size = base_prog_data->total_shared;
      /* compute dispatch width from the valid-SIMD-variant mask (bit0=8, bit1=16, bit2=32) */
      if (ms == MESA_SHADER_COMPUTE) {
         unsigned m = prog_data.cs.prog_mask;
         stats->simd_width = (m & 4) ? 32 : (m & 2) ? 16 : (m & 1) ? 8 : 0;
      }
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
   }

   glsl_type_singleton_decref();
   ralloc_free(mem_ctx);
   return S2I_INTEL_OK;
}
