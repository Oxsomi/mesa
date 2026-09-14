/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * The setup every backend does before it can build a vendor's world, shared, INTERNAL.
 *
 * A compile runs in three phases: this generic setup, the vendor's device creation, the vendor's
 * compile. Only the middle is vendor shaped, so the outer two live here.
 *
 * Backends open a session at the top and close it at ONE label, so every failure path leaves the
 * same way; whatever a vendor owns beyond it is released at that label first.
 */
#ifndef SPIRV2ISA_SESSION_H
#define SPIRV2ISA_SESSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "compiler/glsl_types.h"
#include "util/ralloc.h"

#include "spirv2isa.h"

/* The arguments no backend can do anything with. The target index is not among them: only the
 * dispatcher knows which backends were built, so each backend checks its own count beside this. */
static inline bool
s2i_module_args_ok(const uint32_t *spirv, size_t spirv_words, const char *entry, s2i_stage stage)
{
   if (!spirv || spirv_words < 5 || spirv[0] != 0x07230203u)
      return false;

   return entry && stage >= 0 && stage < S2I_STAGE_COUNT;
}

/* mem_ctx parents every allocation a backend makes for this compile, so closing the session is the
 * whole teardown of the generic half. */
struct s2i_session {
   void *mem_ctx;
};

static inline bool
s2i_session_open(struct s2i_session *session)
{
   session->mem_ctx = ralloc_context(NULL);

   if (!session->mem_ctx)
      return false;

   /* No device did this for us, and the glsl type system is a refcounted process singleton. */
   glsl_type_singleton_init_or_ref();
   return true;
}

static inline void
s2i_session_close(struct s2i_session *session)
{
   glsl_type_singleton_decref();
   ralloc_free(session->mem_ctx);
   session->mem_ctx = NULL;
}

#endif /* SPIRV2ISA_SESSION_H */
