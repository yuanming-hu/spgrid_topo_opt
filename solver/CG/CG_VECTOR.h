//#####################################################################
// Copyright 2017, Haixiang Liu
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_VECTOR
//#####################################################################
#ifndef __CG_VECTOR__
#define __CG_VECTOR__

#include "KRYLOV_VECTOR_BASE.h"
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>

using namespace SPGrid;

template<class T_STRUCT,class T,int d,int log2_page>
class CG_VECTOR:public KRYLOV_VECTOR_BASE<T>
{
    using Base          = KRYLOV_VECTOR_BASE<T>;
    using SPG_Allocator = SPGrid_Allocator<T_STRUCT,d,log2_page>;
    using Mask_type     = typename SPG_Allocator::Array_mask<T>;
    enum{ELEMENTS_PER_BLOCK=Mask_type::elements_per_block};

public:

    bool using_float;
    T T_STRUCT::* field[d];
    SPG_Allocator& allocator;
    const SPGrid_Page_Map<log2_page>& page_map;

    CG_VECTOR(T T_STRUCT::* field_input[d],SPG_Allocator& allocator_input,
              const SPGrid_Page_Map<log2_page>& page_map_input)
        :allocator(allocator_input),page_map(page_map_input),using_float(false)
    {for(int v=0;v<d;++v) field[v]=field_input[v];}
    
    static const CG_VECTOR& Cg_Vector(const Base& base)
    {return dynamic_cast<const CG_VECTOR&>(base);}

    static CG_VECTOR& Cg_Vector(Base& base)
    {return dynamic_cast<CG_VECTOR&>(base);}

    void Clear();
//#####################################################################
    Base& operator+=(const Base& bv);
    Base& operator-=(const Base& bv);
    Base& operator*=(const T a);
    void Copy(const T c,const Base& bv);
    void Copy(const T c1,const Base& bv1,const Base& bv2);
    int Raw_Size() const;
    T& Raw_Get(int i);
//#####################################################################
};
#endif
