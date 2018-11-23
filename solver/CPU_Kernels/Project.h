//#####################################################################
// Copyright (c) 2012-2013, Eftychios Sifakis, Sean Bauer
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Project_h__
#define __Project_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "../Flags.h"
#include <vector>
struct Project
{
    template<typename Array_Type>
    Project(Array_Type array,std::vector<uint64_t> dirichlet)
    {
        #pragma omp parallel for
        for (int i = 0; i < dirichlet.size(); i++) array(dirichlet[i]) = 0;
    }
};
#endif
