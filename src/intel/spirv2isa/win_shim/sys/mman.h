/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Stand-in for the POSIX header of the same name, for Windows builds of spirv2isa.
 *
 * Reached through the Intel device headers, which map kernel-driver memory. spirv2isa maps nothing:
 * the constants exist so the code that names them compiles, and the functions are declared but never
 * called, so a build that did call one would fail to link rather than map anything.
 */
#ifndef SPIRV2ISA_SHIM_SYS_MMAN_H
#define SPIRV2ISA_SHIM_SYS_MMAN_H

#include <stddef.h>

#define MAP_FAILED    ((void *)-1)
#define PROT_READ     1
#define PROT_WRITE    2
#define MAP_SHARED    1
#define MAP_PRIVATE   2
#define MAP_FIXED     16
#define MAP_ANONYMOUS 32

void *mmap(void *addr, size_t len, int prot, int flags, int fd, long off);
int munmap(void *addr, size_t len);

#endif /* SPIRV2ISA_SHIM_SYS_MMAN_H */
