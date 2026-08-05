# Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
#
# spirv2isa (Intel): additive standalone SPIR-V -> Intel EU ISA via the brw compiler.
# Built ONLY when -Dspirv2isa=true. Unlike the AMD backend (folded into RADV), brw is a real library
# with a public compile API, so we just link libintel_compiler (brw) + its deps and drive it.
#
# Wire-up (upstream edits, kept minimal for clean merges):
#   src/meson.build          : enter src/intel when -Dspirv2isa=true (or with_any_intel)
#   src/intel/meson.build    : if get_option('spirv2isa') / subdir('spirv2isa') / endif  (after 'compiler')

spirv2isa_intel_deps = [
  idep_intel_compiler_brw,   # brw_compile / brw_compiler_create (+ isl, nir, vtn, mesautil, gen)
  idep_intel_dev,            # intel_get_device_info_from_pci_id
  idep_nir,
  idep_vtn,                  # spirv_to_nir
  idep_mesautil,
]

spirv2isa_intel_incs = [
  include_directories('.'),
  include_directories('../compiler/brw'),   # brw_compiler.h, brw_nir.h
  inc_include, inc_src, inc_intel, inc_intel_compiler,
]

libspirv2isa_intel = static_library(
  'spirv2isa_intel',
  files('spirv2isa_intel.c'),
  include_directories : spirv2isa_intel_incs,
  dependencies : spirv2isa_intel_deps,
  build_by_default : false,
)

executable(
  'spirv2isa-intel-cli',
  files('spirv2isa_intel_cli.c'),
  include_directories : [ include_directories('.') ],
  link_with : libspirv2isa_intel,
  dependencies : spirv2isa_intel_deps + [ dep_thread, dep_dl, dep_m ],
  build_by_default : true,
  install : false,
)

# Unified multi-vendor CLI: links BOTH backends (AMD via idep_spirv2isa_amd from src/amd/spirv2isa,
# processed before src/intel, and Intel via libspirv2isa_intel) into one binary that dispatches by a
# backend argument. Only buildable when the AMD backend is also present (idep_spirv2isa_amd exists).
if is_variable('idep_spirv2isa_amd')
  executable(
    'spirv2isa-cli-all',
    files('spirv2isa_unified_cli.c'),
    include_directories : [ include_directories('.') ],
    link_with : libspirv2isa_intel,
    dependencies : spirv2isa_intel_deps + [ idep_spirv2isa_amd, dep_thread, dep_dl, dep_m ],
    build_by_default : true,
    install : false,
  )
endif
