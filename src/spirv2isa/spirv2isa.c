/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * spirv2isa: the one public entry point, dispatching to whichever backend owns the target's vendor.
 *
 * Each backend is compiled in independently (S2I_HAVE_AMD / S2I_HAVE_INTEL / S2I_HAVE_NVIDIA, set
 * by meson from the spirv2isa-backends list option), so a library built with a subset still answers
 * for the rest: it reports a missing vendor as absent and refuses its targets, rather than failing
 * to link.
 */
#include "spirv2isa.h"

#include <stdlib.h>
#include <string.h>

#ifdef S2I_HAVE_AMD
#include "../amd/spirv2isa/spirv2isa_amd.h"
#endif

#ifdef S2I_HAVE_INTEL
#include "../intel/spirv2isa/spirv2isa_intel.h"
#endif
#ifdef S2I_HAVE_NVIDIA
#include "../nouveau/spirv2isa/spirv2isa_nvk.h"
#endif

/*
 * Every flagship target of every backend, in vendor then architecture order, each with the short
 * token it is named by. A vendor that isn't built stays in this table: it is what the target names
 * and the "built without that backend" refusal read.
 * The ids live here rather than in a backend because they are the cross-vendor vocabulary: one flat
 * namespace a caller types, stores and keys golden files by, so no two of them may collide.
 * Extended-tier tokens come from the built backend that owns them; each vendor shapes its own (AMD
 * family names, Intel PCI ids, NVIDIA die names), which keeps that namespace collision free too.
 */
static const struct {
   s2i_target target;
   const char *id;
} s2i_all_targets[] = {
   { S2I_TARGET_GFX8_POLARIS10, "gfx803"  },
   { S2I_TARGET_GFX9_VEGA10,    "gfx900"  },
   { S2I_TARGET_GFX10_NAVI10,   "gfx1010" },
   { S2I_TARGET_GFX10_3_NAVI21, "gfx1030" },
   { S2I_TARGET_GFX11_NAVI31,   "gfx1100" },
   { S2I_TARGET_GFX12_GFX1201,  "gfx1201" },
   { S2I_TARGET_GEN9_SKL,       "skl"     },
   { S2I_TARGET_GEN11_ICL,      "icl"     },
   { S2I_TARGET_GEN12_TGL,      "tgl"     },
   { S2I_TARGET_XE_HPG_DG2,     "dg2"     },
   { S2I_TARGET_XE_MTL,         "mtl"     },
   { S2I_TARGET_XE2_LNL,        "lnl"     },
   { S2I_TARGET_XE2_BMG,        "bmg"     },

   { S2I_TARGET_GM204, "gm204" },
   { S2I_TARGET_GP102, "gp102" },
   { S2I_TARGET_TU102, "tu102" },
   { S2I_TARGET_GA102, "ga102" },
   { S2I_TARGET_AD102, "ad102" },
   { S2I_TARGET_GB202, "gb202" },
};

/*
 * The extended tier: a built backend owns its full target list (flagships first, then every other
 * device it can model); these adapters let the vendor-agnostic lookups below reach it. Both give the
 * empty answer for a vendor that is not built, and the token adapter also returns "" for flagship
 * indices, whose tokens live in s2i_all_targets above.
 */
static int
s2i_vendor_target_count(s2i_vendor vendor)
{
   switch (vendor) {

#ifdef S2I_HAVE_AMD
   case S2I_VENDOR_AMD: return s2i_amd_target_count();
#endif

#ifdef S2I_HAVE_INTEL
   case S2I_VENDOR_INTEL: return s2i_intel_target_count();
#endif

#ifdef S2I_HAVE_NVIDIA
   case S2I_VENDOR_NVIDIA: return s2i_nvk_target_count();
#endif

   default: return 0;
   }
}

static const char *
s2i_vendor_target_token(s2i_vendor vendor, int target_index)
{
   switch (vendor) {

#ifdef S2I_HAVE_AMD
   case S2I_VENDOR_AMD: return s2i_amd_target_token(target_index);
#endif

#ifdef S2I_HAVE_INTEL
   case S2I_VENDOR_INTEL: return s2i_intel_target_token(target_index);
#endif

#ifdef S2I_HAVE_NVIDIA
   case S2I_VENDOR_NVIDIA: return s2i_nvk_target_token(target_index);
#endif

   default: return "";
   }
}


static const char *s2i_stage_ids[S2I_STAGE_COUNT] = {
   [S2I_STAGE_VERTEX]       = "vs",
   [S2I_STAGE_PIXEL]        = "ps",
   [S2I_STAGE_COMPUTE]      = "cs",
   [S2I_STAGE_HULL]         = "hs",
   [S2I_STAGE_DOMAIN]       = "ds",
   [S2I_STAGE_GEOMETRY]     = "gs",
   [S2I_STAGE_TASK]         = "task",
   [S2I_STAGE_MESH]         = "mesh",
   [S2I_STAGE_RAYGEN]       = "raygen",
   [S2I_STAGE_CALLABLE]     = "callable",
   [S2I_STAGE_MISS]         = "miss",
   [S2I_STAGE_CLOSEST_HIT]  = "chit",
   [S2I_STAGE_ANY_HIT]      = "ahit",
   [S2I_STAGE_INTERSECTION] = "isect",
};

static int
s2i_id_equals(const char *a, const char *b)
{
   for (; *a && *b; ++a, ++b) {

      const char ca = (*a >= 'A' && *a <= 'Z') ? (char)(*a + ('a' - 'A')) : *a;
      const char cb = (*b >= 'A' && *b <= 'Z') ? (char)(*b + ('a' - 'A')) : *b;

      if (ca != cb)
         return 0;
   }

   return *a == *b;
}

static char *
s2i_message(const char *text)
{
   const size_t len = strlen(text) + 1;
   char *copy = (char *)malloc(len);

   if (copy)
      memcpy(copy, text, len);

   return copy;
}

/* A target is servable when its vendor exists, its index is one of that vendor's, and the backend
 * that owns it was built in. Anything else is S2I_BAD_TARGET with a message saying which it was. */
static s2i_result
s2i_check_target(s2i_target target, char **message)
{
   const s2i_vendor vendor = S2I_TARGET_VENDOR_OF(target);

   switch (vendor) {

#ifdef S2I_HAVE_AMD
   case S2I_VENDOR_AMD:
      if (S2I_TARGET_INDEX_OF(target) >= 0 && S2I_TARGET_INDEX_OF(target) < s2i_amd_target_count())
         return S2I_OK;
      break;
#endif

#ifdef S2I_HAVE_INTEL
   case S2I_VENDOR_INTEL:
      if (S2I_TARGET_INDEX_OF(target) >= 0 && S2I_TARGET_INDEX_OF(target) < s2i_intel_target_count())
         return S2I_OK;
      break;
#endif

#ifdef S2I_HAVE_NVIDIA
   case S2I_VENDOR_NVIDIA:
      if (S2I_TARGET_INDEX_OF(target) >= 0 && S2I_TARGET_INDEX_OF(target) < s2i_nvk_target_count())
         return S2I_OK;
      break;
#endif

   default:
      break;
   }

   if (message) {
      if (vendor > 0 && vendor < S2I_VENDOR_COUNT && !s2i_has_vendor(vendor))
         *message = s2i_message("spirv2isa was built without this target's backend");
      else
         *message = s2i_message("no such spirv2isa target");
   }

   return S2I_BAD_TARGET;
}

/* The three facts every backend measures, lifted out of its own stats so a caller can read them
 * without knowing the vendor. */
static void
s2i_stats_fill_shared(s2i_stats *stats)
{
   switch (stats->vendor) {

   case S2I_VENDOR_AMD:
      stats->code_size = stats->amd.code_size;
      stats->scratch_size = stats->amd.scratch_size;
      stats->shared_size = stats->amd.lds_size;
      break;

   case S2I_VENDOR_INTEL:
      stats->code_size = stats->intel.program_size;
      stats->scratch_size = stats->intel.scratch_size;
      stats->shared_size = stats->intel.shared_size;
      break;

   case S2I_VENDOR_NVIDIA:
      stats->code_size = stats->nvidia.code_size;
      stats->scratch_size = stats->nvidia.slm_size;
      stats->shared_size = stats->nvidia.smem_size;
      break;

   default:
      break;
   }
}

s2i_result
s2i_compile(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage,
            const s2i_pipeline *pipeline, char **isa_text, s2i_stats *stats, s2i_info *info,
            char **message)
{
   if (!pipeline)
      return S2I_BAD_TARGET;

   const s2i_target target = pipeline->target;
   const s2i_result targetResult = s2i_check_target(target, message);

   if (targetResult != S2I_OK)
      return targetResult;

   s2i_result result = S2I_BAD_TARGET;

#if !defined(S2I_HAVE_AMD) && !defined(S2I_HAVE_INTEL) && !defined(S2I_HAVE_NVIDIA)
   (void)spirv; (void)spirv_words; (void)entry; (void)stage; (void)pipeline; (void)isa_text;
   (void)info;
#endif

   /* The backend fills its own half of the stats; the shared half is derived from it afterwards, so
    * a backend never has to know the shared fields exist. */

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->vendor = S2I_TARGET_VENDOR_OF(target);
   }

   switch (S2I_TARGET_VENDOR_OF(target)) {

#ifdef S2I_HAVE_AMD
   case S2I_VENDOR_AMD:
      result = s2i_amd_compile(spirv, spirv_words, entry, stage, S2I_TARGET_INDEX_OF(target),
                               pipeline, isa_text,
                               stats ? &stats->amd : NULL, info, message);
      break;
#endif

#ifdef S2I_HAVE_INTEL
   case S2I_VENDOR_INTEL:

      /* Intel reports no RT mode of its own, so those fields keep their defaults rather than reading
       * as a monolithic compile that never happened; the backend fills in the rest. */

      if (info)
         memset(info, 0, sizeof(*info));

      result = s2i_intel_compile(spirv, spirv_words, entry, stage, S2I_TARGET_INDEX_OF(target),
                                 pipeline, isa_text,
                                 stats ? &stats->intel : NULL, info, message);
      break;
#endif

#ifdef S2I_HAVE_NVIDIA
   case S2I_VENDOR_NVIDIA:

      /* The NVIDIA backend fills no compile facts yet, so they read as zeroed rather than stale. */

      if (info)
         memset(info, 0, sizeof(*info));

      result = s2i_nvk_compile(spirv, spirv_words, entry, stage, S2I_TARGET_INDEX_OF(target),
                               pipeline, isa_text,
                               stats ? &stats->nvidia : NULL, info, message);
      break;
#endif

   default:
      break;
   }

   if (result == S2I_OK && stats)
      s2i_stats_fill_shared(stats);

   return result;
}

s2i_result
s2i_compile_rt_pipeline(const s2i_rt_shader *shaders, size_t shader_count, size_t entry_index,
                        int compile_traversal, const s2i_pipeline *pipeline, char **isa_text,
                        s2i_stats *stats, char **message)
{
   if (!pipeline)
      return S2I_BAD_TARGET;

   const s2i_target target = pipeline->target;
   const s2i_result targetResult = s2i_check_target(target, message);

   if (targetResult != S2I_OK)
      return targetResult;

   if (stats) {
      memset(stats, 0, sizeof(*stats));
      stats->vendor = S2I_TARGET_VENDOR_OF(target);
   }

#ifndef S2I_HAVE_AMD
   (void)shaders; (void)shader_count; (void)entry_index; (void)compile_traversal; (void)pipeline;
   (void)isa_text;
#endif

#ifdef S2I_HAVE_AMD
   if (S2I_TARGET_VENDOR_OF(target) == S2I_VENDOR_AMD) {

      const s2i_result result = s2i_amd_compile_rt_pipeline(
         shaders, shader_count, entry_index, compile_traversal, S2I_TARGET_INDEX_OF(target),
         pipeline, isa_text, stats ? &stats->amd : NULL, message);

      if (result == S2I_OK && stats)
         s2i_stats_fill_shared(stats);

      return result;
   }
#endif

   /* The target is real, the operation is not wired for it: only RADV's traversal + inlining model is
    * implemented so far. Intel compiles RT stages one at a time through s2i_compile, so its message
    * points there; NVIDIA refuses RT stages entirely, so its message must not. */

   if (message) {
      *message = S2I_TARGET_VENDOR_OF(target) == S2I_VENDOR_NVIDIA ?
         s2i_message("ray tracing is not wired on the NVIDIA backend yet") :
         s2i_message(
            "this backend has no whole-pipeline ray tracing compile yet; compile each shader with "
            "s2i_compile instead");
   }

   return S2I_UNSUPPORTED_CAP;
}

char *
s2i_unsupported_caps(const uint32_t *caps_used, size_t caps_count, s2i_target target)
{
#ifdef S2I_HAVE_AMD
   if (S2I_TARGET_VENDOR_OF(target) == S2I_VENDOR_AMD &&
       S2I_TARGET_INDEX_OF(target) >= 0 && S2I_TARGET_INDEX_OF(target) < s2i_amd_target_count())
      return s2i_amd_unsupported_caps(caps_used, caps_count, S2I_TARGET_INDEX_OF(target));
#else
   (void)caps_used;
   (void)caps_count;
#endif

   /* A backend without its own capability matrix reports nothing unsupported here; s2i_compile's
    * declared-capability gate is the authoritative check either way. */

   (void)target;
   return NULL;
}

const char *
s2i_target_name(s2i_target target)
{
   switch (S2I_TARGET_VENDOR_OF(target)) {

#ifdef S2I_HAVE_AMD
   case S2I_VENDOR_AMD:
      return s2i_amd_target_name(S2I_TARGET_INDEX_OF(target));
#endif

#ifdef S2I_HAVE_INTEL
   case S2I_VENDOR_INTEL:
      return s2i_intel_target_name(S2I_TARGET_INDEX_OF(target));
#endif

#ifdef S2I_HAVE_NVIDIA
   case S2I_VENDOR_NVIDIA:
      return s2i_nvk_target_name(S2I_TARGET_INDEX_OF(target));
#endif

   default:
      break;
   }

   return "";
}

size_t
s2i_targets(s2i_target *targets, size_t capacity)
{
   const size_t static_count = sizeof(s2i_all_targets) / sizeof(s2i_all_targets[0]);
   size_t n = 0;

   for (int vendor = 1; vendor < S2I_VENDOR_COUNT; ++vendor) {

      const int built_count = s2i_vendor_target_count((s2i_vendor)vendor);

      if (built_count > 0) {
         for (int i = 0; i < built_count; ++i, ++n)
            if (targets && n < capacity)
               targets[n] = S2I_TARGET(vendor, i);
         continue;
      }

      for (size_t i = 0; i < static_count; ++i)
         if ((int)S2I_TARGET_VENDOR_OF(s2i_all_targets[i].target) == vendor) {
            if (targets && n < capacity)
               targets[n] = s2i_all_targets[i].target;
            ++n;
         }
   }

   return n;
}

int
s2i_target_is_flagship(s2i_target target)
{
   const int index = S2I_TARGET_INDEX_OF(target);

   switch (S2I_TARGET_VENDOR_OF(target)) {
   case S2I_VENDOR_AMD:    return index >= 0 && index < (int)S2I_TARGET_AMD_COUNT;
   case S2I_VENDOR_INTEL:  return index >= 0 && index < (int)S2I_TARGET_INTEL_COUNT;
   case S2I_VENDOR_NVIDIA: return index >= 0 && index < (int)S2I_TARGET_NVIDIA_COUNT;
   default:                return 0;
   }
}

const char *
s2i_target_id(s2i_target target)
{
   const size_t count = sizeof(s2i_all_targets) / sizeof(s2i_all_targets[0]);

   for (size_t i = 0; i < count; ++i)
      if (s2i_all_targets[i].target == target)
         return s2i_all_targets[i].id;

   {
      const char *token = s2i_vendor_target_token(S2I_TARGET_VENDOR_OF(target), S2I_TARGET_INDEX_OF(target));

      if (token[0])
         return token;
   }

   return "";
}

s2i_target
s2i_target_from_id(const char *id)
{
   const size_t count = sizeof(s2i_all_targets) / sizeof(s2i_all_targets[0]);

   if (!id)
      return 0;

   for (size_t i = 0; i < count; ++i)
      if (s2i_id_equals(id, s2i_all_targets[i].id))
         return s2i_all_targets[i].target;

   for (int vendor = 1; vendor < S2I_VENDOR_COUNT; ++vendor) {

      const int built_count = s2i_vendor_target_count((s2i_vendor)vendor);

      for (int i = 0; i < built_count; ++i) {

         const char *token = s2i_vendor_target_token((s2i_vendor)vendor, i);

         if (token[0] && s2i_id_equals(id, token))
            return S2I_TARGET(vendor, i);
      }
   }

   return 0;
}

const char *
s2i_stage_id(s2i_stage stage)
{
   if (stage < 0 || stage >= S2I_STAGE_COUNT || !s2i_stage_ids[stage])
      return "";

   return s2i_stage_ids[stage];
}

int
s2i_stage_from_id(const char *id)
{
   if (!id)
      return -1;

   for (int i = 0; i < S2I_STAGE_COUNT; ++i)
      if (s2i_stage_ids[i] && s2i_id_equals(id, s2i_stage_ids[i]))
         return i;

   return -1;
}

const char *
s2i_vendor_name(s2i_vendor vendor)
{
   switch (vendor) {
   case S2I_VENDOR_AMD:   return "AMD";
   case S2I_VENDOR_INTEL: return "Intel";
   case S2I_VENDOR_NVIDIA: return "NVIDIA";
   default:               return "";
   }
}

int
s2i_has_vendor(s2i_vendor vendor)
{
   switch (vendor) {

#ifdef S2I_HAVE_AMD
   case S2I_VENDOR_AMD:   return 1;
#endif

#ifdef S2I_HAVE_INTEL
   case S2I_VENDOR_INTEL: return 1;
#endif

#ifdef S2I_HAVE_NVIDIA
   case S2I_VENDOR_NVIDIA: return 1;
#endif

   default:               return 0;
   }
}
