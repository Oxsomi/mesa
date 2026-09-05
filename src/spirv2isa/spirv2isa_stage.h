/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * The stage vocabulary every backend shares, INTERNAL.
 *
 * s2i_stage is the caller's enum; mesa_shader_stage is what every Mesa compiler dispatches on. The
 * mapping between them is a property of those two enums and not of any vendor, and the two enums have
 * drifted apart once already, so there is exactly one table.
 */
#ifndef SPIRV2ISA_STAGE_H
#define SPIRV2ISA_STAGE_H

#include <stdbool.h>

#include "compiler/shader_enums.h"

#include "spirv2isa.h"

/* The caller's stage as Mesa names it. Never derived from the SPIR-V: the caller says which stage it
 * compiled, because a module can carry several entrypoints. */
static inline mesa_shader_stage
s2i_stage_to_mesa(s2i_stage stage)
{
   static const mesa_shader_stage to_mesa[S2I_STAGE_COUNT] = {
      [S2I_STAGE_VERTEX]       = MESA_SHADER_VERTEX,
      [S2I_STAGE_PIXEL]        = MESA_SHADER_FRAGMENT,
      [S2I_STAGE_COMPUTE]      = MESA_SHADER_COMPUTE,
      [S2I_STAGE_GEOMETRY]     = MESA_SHADER_GEOMETRY,
      [S2I_STAGE_HULL]         = MESA_SHADER_TESS_CTRL,
      [S2I_STAGE_DOMAIN]       = MESA_SHADER_TESS_EVAL,
      [S2I_STAGE_TASK]         = MESA_SHADER_TASK,
      [S2I_STAGE_MESH]         = MESA_SHADER_MESH,
      [S2I_STAGE_RAYGEN]       = MESA_SHADER_RAYGEN,
      [S2I_STAGE_CALLABLE]     = MESA_SHADER_CALLABLE,
      [S2I_STAGE_MISS]         = MESA_SHADER_MISS,
      [S2I_STAGE_CLOSEST_HIT]  = MESA_SHADER_CLOSEST_HIT,
      [S2I_STAGE_ANY_HIT]      = MESA_SHADER_ANY_HIT,
      [S2I_STAGE_INTERSECTION] = MESA_SHADER_INTERSECTION,
   };

   return to_mesa[stage];
}

/* True for the ray tracing pipeline stages. They are entered from an RT dispatcher and can suspend at
 * a traceRay or callable call, so every backend compiles them as call-shaped shaders rather than as a
 * fixed-function stage, and every backend needs the same answer to which stages those are. */
static inline bool
s2i_stage_is_rt(mesa_shader_stage ms)
{
   switch (ms) {
   case MESA_SHADER_RAYGEN:
   case MESA_SHADER_ANY_HIT:
   case MESA_SHADER_CLOSEST_HIT:
   case MESA_SHADER_MISS:
   case MESA_SHADER_INTERSECTION:
   case MESA_SHADER_CALLABLE:
      return true;
   default:
      return false;
   }
}

#endif /* SPIRV2ISA_STAGE_H */
