# Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
#
# The one library and the one CLI: the dispatcher here plus whichever backends were
# enabled. Each backend is optional, so a build with a single vendor still links: the dispatcher
# compiles the other vendor's arm out (S2I_HAVE_AMD / S2I_HAVE_INTEL) and reports it as absent
# instead of failing to resolve its symbols.
#
# Built ONLY when -Dspirv2isa=true, and included from src/meson.build after the vendor subdirs, so
# the backends' dependencies (idep_spirv2isa_amd / idep_spirv2isa_intel) already exist.

spirv2isa_args = []
spirv2isa_deps = []

if with_spirv2isa_amd
  spirv2isa_args += '-DS2I_HAVE_AMD'
  spirv2isa_deps += idep_spirv2isa_amd
endif

if with_spirv2isa_intel
  spirv2isa_args += '-DS2I_HAVE_INTEL'
  spirv2isa_deps += idep_spirv2isa_intel
endif

if with_spirv2isa_nvidia
  spirv2isa_args += '-DS2I_HAVE_NVIDIA'
  spirv2isa_deps += idep_spirv2isa_nvidia
endif

libspirv2isa = static_library(
  'spirv2isa',
  files('spirv2isa.c'),
  include_directories : [ include_directories('.'), inc_include, inc_src ],
  c_args : spirv2isa_args,
  dependencies : spirv2isa_deps,
  build_by_default : true,
  install : false,
)

# What OxC3 links: the dispatcher, its backends and the public header's directory.

idep_spirv2isa = declare_dependency(
  link_with : libspirv2isa,
  include_directories : include_directories('.'),
  dependencies : spirv2isa_deps,
)

executable(
  'spirv2isa-cli',
  files('spirv2isa_cli.c'),
  include_directories : [ include_directories('.'), inc_include, inc_src ],
  dependencies : [ idep_spirv2isa ],
  build_by_default : true,
  install : true,
)
