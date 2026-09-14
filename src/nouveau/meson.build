# Copyright © 2021 Ilia Mirkin
# SPDX-License-Identifier: MIT

subdir('headers')
subdir('drm')
subdir('winsys')
if with_nouveau_vk or with_spirv2isa_nvidia
  subdir('rust')
  subdir('nil')
  subdir('compiler')
endif
if with_tools.contains('drm-shim')
  subdir('drm-shim')
endif

if with_nouveau_vk or with_spirv2isa_nvidia
  subdir('cubin')
  subdir('mme')
  subdir('vulkan')
endif

if with_spirv2isa_nvidia
  subdir('spirv2isa')
endif
