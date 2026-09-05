/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Stand-in for the POSIX header of the same name, for Windows builds of spirv2isa.
 *
 * The Intel common headers include <sys/ioctl.h> to talk to the i915/xe kernel driver. spirv2isa is
 * device-free and never opens a device, so the declaration only has to exist for the include to
 * resolve; nothing here is ever called, and a build that did call it would fail to link.
 */
#ifndef SPIRV2ISA_SHIM_SYS_IOCTL_H
#define SPIRV2ISA_SHIM_SYS_IOCTL_H

int ioctl(int fd, unsigned long request, ...);

#endif /* SPIRV2ISA_SHIM_SYS_IOCTL_H */
