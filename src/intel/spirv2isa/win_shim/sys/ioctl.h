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

/* The Linux ioctl-number encoding, which the vendored kernel UAPI headers (include/drm-uapi) build
 * their request constants with. Only the arithmetic is needed; nothing here is ever issued. */
#define _IOC(dir, type, nr, size)    (((dir) << 30) | ((size) << 16) | ((type) << 8) | (nr))
#define _IO(type, nr)         _IOC(0u, (type), (nr), 0u)
#define _IOR(type, nr, size)  _IOC(2u, (type), (nr), (unsigned)sizeof(size))
#define _IOW(type, nr, size)  _IOC(1u, (type), (nr), (unsigned)sizeof(size))
#define _IOWR(type, nr, size) _IOC(3u, (type), (nr), (unsigned)sizeof(size))

#endif /* SPIRV2ISA_SHIM_SYS_IOCTL_H */
