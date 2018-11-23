//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SYSTEM
//#####################################################################
#ifndef __CG_SYSTEM__
#define __CG_SYSTEM__

#include "KRYLOV_SYSTEM_BASE.h"
#include "CG_VECTOR.h"
#include "../CPU_Kernels/SPGrid_Linear_Elasticity.h"
#include "../MG/Multigrid_Hierarchy.h"
using namespace SPGrid;

template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width>
class CG_SYSTEM:public KRYLOV_SYSTEM_BASE<T>
{
    using Base              = KRYLOV_SYSTEM_BASE<T>;
    using Vector_Base       = KRYLOV_VECTOR_BASE<T>;
    using SPG_Allocator     = SPGrid_Allocator<T_STRUCT,d,log2_page>;
    using Page_Map          = SPGrid_Page_Map<log2_page>;
    using T_MASK            = typename SPG_Allocator::Array_mask<T>;
    using Linear_Elasticity = SPGrid_Linear_Elasticity<T,T_STRUCT,log2_page,simd_width>;
    using MG_Type           = Multigrid_Hierarchy<T_STRUCT,T,T_FLAG,log2_page,d,simd_width>;
    enum{ELEMENTS_PER_BLOCK=T_MASK::elements_per_block};
    T dx;
    T T_STRUCT::* mu_field,T_STRUCT::* lambda_field;
public:
    MG_Type* mg;    
    mutable T T_STRUCT::* diagonal_channels[d*2];
    mutable T T_STRUCT::* mg_r_channels[d];
    T* s_bk[d];
    std::array<std::vector<uint64_t>,d> dirichlet;
    int interior_smoothing_iterations;
    int boundary_smoothing_iterations;
    CG_SYSTEM(SPG_Allocator& allocator,Page_Map& page_map,
              T T_STRUCT::* mu_field,T T_STRUCT::* lambda_field,
              T dx, std::vector<uint64_t> dirichlet_cells[d]);
    CG_SYSTEM(SPG_Allocator& allocator,Page_Map& page_map,
              T T_STRUCT::* mu_field_input,T T_STRUCT::* lambda_field_input,
              T dx_input, std::vector<uint64_t> dirichlet_cells[d],
              int mg_levels,int interior_smoothing_iterations_input,int boundary_smoothing_iterations_input);
    ~CG_SYSTEM(){if(mg) delete mg;for(int v=0;v<d;++v) if(s_bk[v]) delete s_bk[v];}
    //###################inherited#########################################
    void Multiply(const Vector_Base& v,Vector_Base& result) const;
    void Multiply(const Vector_Base& v,Vector_Base& result,T T_STRUCT::* mu_field,T T_STRUCT::* lambda_field) const;
    void Residual(const Vector_Base& v,const Vector_Base& rhs,Vector_Base& result,T T_STRUCT::* mu_field,T T_STRUCT::* lambda_field) const;
    double Inner_Product(const Vector_Base& x,const Vector_Base& y) const;
    T Convergence_Norm(const Vector_Base& x) const;
    void Project(Vector_Base& x) const;
    void Set_Boundary_Conditions(Vector_Base& x) const;
    void Project_Nullspace(Vector_Base& x) const;
protected:
    void Apply_Preconditioner(const Vector_Base& r, Vector_Base& z) const;
//#####################################################################
};
#endif
