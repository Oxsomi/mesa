# Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
#
# spirv2isa: additive standalone SPIR-V -> AMD ISA (ACO), no VkDevice.
# Built ONLY when -Dspirv2isa=true, so it never affects a normal Mesa/OxC3 build.
# Wire-up (the only edits to upstream files, kept to two lines so upstream merges stay clean):
#   meson.options : option('spirv2isa', type : 'boolean', value : false,
#                          description : 'Build the standalone SPIR-V->AMD ISA library (OxC3)')
#   src/amd/meson.build (end) : if get_option('spirv2isa')
#                                 subdir('spirv2isa')
#                               endif

libspirv2isa = static_library(
  'spirv2isa',
  files('spirv2isa.c'),
  include_directories : [
    inc_include, inc_src, inc_util, inc_amd, inc_amd_common, inc_compiler,
  ],
  dependencies : [
    idep_nir, idep_vtn, idep_aco, idep_mesautil,
    idep_amd_generated_headers,
    # TODO(iterate): radv_postprocess_nir / radv_declare_shader_args / radv_nir_lower_descriptors live in
    # the RADV build (libvulkan_radeon). To reuse them statically, either add these files to the RADV
    # meson target, or expose a small 'radv_compile' static_library. Resolve during the first WSL build.
  ],
  gnu_symbol_visibility : 'hidden',
  build_by_default : true,
)

idep_spirv2isa = declare_dependency(
  link_with : libspirv2isa,
  include_directories : include_directories('.'),
)

# Tiny CLI for iterating/testing: spirv2isa-cli <target> <shader.spv> [entry]
executable(
  'spirv2isa-cli',
  files('spirv2isa_cli.c'),
  dependencies : [idep_spirv2isa],
  build_by_default : true,
  install : false,
)
