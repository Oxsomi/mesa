/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * The declared-capability gate, shared by every backend, INTERNAL.
 *
 * vtn only warns on a capability the target lacks and parses on, because a driver can rely on the
 * validation layers having rejected the module first. Nothing vets a module here, so every declared
 * capability is checked against the driver's own tables and a miss is refused by name, rather than
 * warned about and then aborted deep inside the backend.
 */
#ifndef SPIRV2ISA_CAPS_H
#define SPIRV2ISA_CAPS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"

/* The refusal a caller sees and frees, or NULL when the module declares nothing the target lacks.
 * `malformed` says the module could not be walked at all, which is S2I_BAD_SPIRV to the caller
 * rather than S2I_UNSUPPORTED_CAP. */
static inline char *
s2i_gate_declared_capabilities(const uint32_t *spirv, size_t spirv_words,
                               const struct spirv_capabilities *target_caps,
                               const char *target_name,
                               bool *malformed)
{
   *malformed = false;

   for (size_t w = 5; w + 1 < spirv_words && (spirv[w] & 0xffffu) == SpvOpCapability;
        w += spirv[w] >> 16) {

      /* A zero word count would spin forever; nothing else vets the module before this runs. */
      if ((spirv[w] >> 16) == 0) {
         *malformed = true;
         return strdup("malformed SPIR-V: an instruction declares a word count of 0");
      }

      const SpvCapability cap = (SpvCapability)spirv[w + 1];

      if (spirv_capabilities_get(target_caps, cap))
         continue;

      char buf[160];

      snprintf(buf, sizeof(buf), "the module declares SPIR-V capability %u (%s), which %s does not "
               "support", (unsigned)cap, spirv_capability_to_string(cap), target_name);
      return strdup(buf);
   }

   return NULL;
}

#endif /* SPIRV2ISA_CAPS_H */
