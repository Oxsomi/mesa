# Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
#
# spirv2isa (Intel): additive standalone SPIR-V -> Intel EU ISA via the brw compiler.
# Built ONLY when -Dspirv2isa=true and 'intel' is in -Dspirv2isa-backends. Unlike the AMD backend
# (folded into RADV), brw is a real library with a public compile API, so we just link
# libintel_compiler (brw) + its deps and drive it.
#
# The CLI is not here: one binary drives every backend, so it is built next to the dispatcher
# (src/spirv2isa) against the unified API.
#
# Wire-up (upstream edits, kept minimal for clean merges):
#   src/meson.build          : enter src/intel when with_spirv2isa_intel (or with_any_intel)
#   src/intel/meson.build    : if with_spirv2isa_intel / subdir('spirv2isa') / endif  (after 'compiler')

spirv2isa_intel_deps = [
  idep_anv_nir_lowering,     # ANV's own descriptor / driver-value / multiview lowering, linked not copied
  idep_intel_compiler_brw,   # brw_compile / brw_compiler_create (+ isl, nir, vtn, mesautil, gen)
  idep_intel_dev,            # intel_get_device_info_from_pci_id
  idep_nir,
  idep_vtn,                  # spirv_to_nir
  idep_mesautil,
]

spirv2isa_intel_incs = [
  include_directories('.'),
  include_directories('../../spirv2isa'),   # the public spirv2isa.h
  include_directories('../compiler/brw'),   # brw_compiler.h, brw_nir.h
  inc_include, inc_src, inc_intel, inc_intel_compiler,
]

libspirv2isa_intel = static_library(
  'spirv2isa_intel',
  files('spirv2isa_intel.c', 'spirv2isa_intel_anv.c'),
  include_directories : spirv2isa_intel_incs,
  c_args : ['-DANV_SHADER_LOWER_NIR_ONLY', '-DANV_PHYSICAL_DEVICE_FEATURES_ONLY'],
  dependencies : spirv2isa_intel_deps,
  build_by_default : false,
)

# Bundle the Intel backend the way the AMD one is bundled, so the dispatcher links either or both
# without knowing how each was built.

idep_spirv2isa_intel = declare_dependency(
  link_with : libspirv2isa_intel,
  include_directories : [ include_directories('.'), include_directories('../../spirv2isa') ],
  dependencies : spirv2isa_intel_deps + [ dep_thread, dep_dl, dep_m ],
)
