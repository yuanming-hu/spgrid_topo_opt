//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __FLAGS_H__
#define __FLAGS_H__

#include <cstdint>

constexpr uint32_t CLEAR              = 0x00000;
constexpr uint32_t ACTIVE_NODE        = 0x00010;
constexpr uint32_t DIRICHLET_NODE_X   = 0x00020;
constexpr uint32_t DIRICHLET_NODE_Y   = 0x00040;
constexpr uint32_t DIRICHLET_NODE_Z   = 0x00080;
constexpr uint32_t DIRICHLET_NODE     = DIRICHLET_NODE_X | DIRICHLET_NODE_Y | DIRICHLET_NODE_Z;
constexpr uint32_t ACTIVE_CELL        = 0x00100;
constexpr uint32_t DIRICHLET_CELL_X   = 0x00200;
constexpr uint32_t DIRICHLET_CELL_Y   = 0x00400;
constexpr uint32_t DIRICHLET_CELL_Z   = 0x00800;
constexpr uint32_t DIRICHLET_CELL     = DIRICHLET_CELL_X | DIRICHLET_CELL_Y | DIRICHLET_CELL_Z;
constexpr uint32_t BOUNDARY_NODE      = 0x01000; //for boundary smoothing
constexpr uint32_t BOUNDARY_NODE_TEMP = 0x02000; //for boundary smoothing
constexpr uint32_t INSIGNIFICANT_CELL = 0x10000; //for visualization smoothing
constexpr uint32_t DONE               = 0x00004; //for fast marching

#endif
