/*
 * Copyright 2026 Oxsomi / Nielsbishere
 * SPDX-License-Identifier: MIT
 *
 * Stand-in for libdrm's header of the same name, for Windows builds of spirv2isa.
 *
 * Reached through ANV's physical-device code, whose feature tables spirv2isa compiles. Everything
 * that would call into libdrm sits behind guards in that build, so only the names have to exist.
 */
#ifndef SPIRV2ISA_SHIM_XF86DRM_H
#define SPIRV2ISA_SHIM_XF86DRM_H

#include <stdint.h>

#define DRM_NODE_PRIMARY 0
#define DRM_NODE_RENDER  2
#define DRM_BUS_PCI      0

typedef struct _drmPciBusInfo {
   uint16_t domain;
   uint8_t bus, dev, func;
} drmPciBusInfoRec, *drmPciBusInfoPtr;

typedef struct _drmPciDeviceInfo {
   uint16_t vendor_id, device_id, subvendor_id, subdevice_id;
   uint8_t revision_id;
} drmPciDeviceInfoRec, *drmPciDeviceInfoPtr;

typedef struct _drmDevice {
   char **nodes;
   int available_nodes;
   int bustype;
   union { drmPciBusInfoPtr pci; } businfo;
   union { drmPciDeviceInfoPtr pci; } deviceinfo;
} drmDevice, *drmDevicePtr;

int drmGetDevice2(int fd, uint32_t flags, drmDevicePtr *device);
int drmGetDevices2(uint32_t flags, drmDevicePtr devices[], int max_devices);
void drmFreeDevice(drmDevicePtr *device);
void drmFreeDevices(drmDevicePtr devices[], int count);
char *drmGetDeviceNameFromFd2(int fd);
int drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle);

#endif /* SPIRV2ISA_SHIM_XF86DRM_H */
