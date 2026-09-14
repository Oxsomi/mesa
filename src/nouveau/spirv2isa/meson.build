# Copyright 2026 Oxsomi / Nielsbishere - SPDX-License-Identifier: MIT
#
# spirv2isa (NVIDIA): additive standalone SPIR-V -> SASS via NVK's lowering and NAK. Built ONLY when
# -Dspirv2isa=true and 'nvidia' is in -Dspirv2isa-backends. NAK is a real library with a public
# compile API (like Intel's brw, unlike the RADV fold), and NVK's layout and lowering code is reused
# through guarded rebuilds, so nothing driver-shaped is copied.

libspirv2isa_nvk = static_library(
  'spirv2isa_nvk',
  files('spirv2isa_nvk.c'),
  include_directories : [
    include_directories('.'),
    include_directories('../../spirv2isa'),
    include_directories('../vulkan'),
    include_directories('../headers'),
    include_directories('../headers/nvidia/classes'),
    inc_include, inc_src,
  ],
  c_args : ['-DNVK_SHADER_LOWER_NIR_ONLY', '-DNVK_PHYSICAL_DEVICE_FEATURES_ONLY'],
  dependencies : [
    idep_nvk_nir_lowering,
    idep_nak,
    idep_nir,
    idep_vtn,
    idep_mesautil,
    idep_vulkan_wsi_headers,
  ],
  build_by_default : false,
)

idep_spirv2isa_nvidia = declare_dependency(
  link_with : libspirv2isa_nvk,
  include_directories : [ include_directories('.'), include_directories('../../spirv2isa') ],
  dependencies : [ idep_nvk_nir_lowering, idep_nak, idep_nir, idep_vtn, idep_mesautil,
                   dep_thread, dep_dl, dep_m ],
)
