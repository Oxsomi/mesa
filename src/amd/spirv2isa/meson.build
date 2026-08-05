# Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
#
# spirv2isa: additive standalone SPIR-V -> AMD ISA (ACO), no VkDevice.
# Built ONLY when -Dspirv2isa=true. spirv2isa.c is folded into RADV's own compilation (see
# src/amd/vulkan/meson.build: appended to libradv_files) so it can call the internal device-free
# radv_* compile functions. Here we only build a test CLI, linking RADV's compiled objects
# statically via extract_all_objects(); that pulls in our folded object + radv_* without needing
# to export anything from the (hidden-visibility) libvulkan_radeon.so.
#
# Wire-up (upstream edits, kept minimal for clean merges):
#   meson.options                    : option('spirv2isa', ...)
#   src/amd/meson.build (end)        : if with_amd_vk and get_option('spirv2isa') / subdir('spirv2isa') / endif
#   src/amd/vulkan/meson.build       : if get_option('spirv2isa') / libradv_files += files('../spirv2isa/spirv2isa.c') / endif

# Bundle the AMD backend (the folded RADV objects + its link deps + the header dir) as a dependency,
# so both the AMD CLI here and the unified multi-vendor CLI (src/intel/spirv2isa) can link it.
# NOTE: no radv_deps / idep_radv_tracepoints - those inject generated tracepoint *sources* that need
# RADV's internal include context; their compiled objects already come via extract_all_objects.
idep_spirv2isa_amd = declare_dependency(
  objects : libvulkan_radeon.extract_all_objects(recursive : false),
  include_directories : [ include_directories('.'), inc_include, inc_src, inc_amd, inc_amd_common, inc_util ],
  link_with : [ libamd_common, libamd_common_llvm, libamdgpu_addrlib ],
  dependencies : [
    dep_llvm, dep_libdrm_amdgpu, dep_thread, dep_elf, dep_dl, dep_m,
    dep_valgrind, idep_aco, idep_mesautil, idep_nir,
    idep_vulkan_util, idep_vulkan_wsi, idep_vulkan_runtime,
    idep_amd_generated_headers, idep_xmlconfig,
  ],
)

executable(
  'spirv2isa-cli',
  files('spirv2isa_cli.c'),
  dependencies : [ idep_spirv2isa_amd ],
  build_by_default : true,
  install : false,
)
