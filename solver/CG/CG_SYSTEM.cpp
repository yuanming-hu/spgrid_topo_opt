//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SYSTEM
//#####################################################################
#include <math.h>
#include "CG_SYSTEM.h"
#include "CG_VECTOR.h"
#include "../CPU_Kernels/Flaging.h"
#include "../CPU_Kernels/Validate_Blocks.h"
#include "../CPU_Kernels/Project_Null_Space.h"
#include "../Elasticity_Solver_Data.h"

using namespace SPGrid;

// All empty: 0.76s

//#####################################################################
// Constructor
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width>
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
CG_SYSTEM(SPG_Allocator& allocator,Page_Map& page_map,
          T T_STRUCT::* mu_field_input,T T_STRUCT::* lambda_field_input,
          T dx_input, std::vector<uint64_t> dirichlet_cells[d])
    :Base(false,false),mu_field(mu_field_input),lambda_field(lambda_field_input),dx(dx_input),mg(NULL)
{
    // Compilation: 0.15s
    Flaging<T_STRUCT,T_FLAG,T,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags,mu_field);
    Flaging<T_STRUCT,T_FLAG,T,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags,dirichlet_cells);
    Flag_Nodes<T_STRUCT,T_FLAG,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags);
    Create_Dirichlet_List<T_STRUCT,T_FLAG,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags,dirichlet);
    for(int v=0;v<d;++v) s_bk[v]=NULL;
}
//#####################################################################
// Constructor
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width>
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
CG_SYSTEM(SPG_Allocator& allocator,Page_Map& page_map,
          T T_STRUCT::* mu_field_input,T T_STRUCT::* lambda_field_input,
          T dx_input, std::vector<uint64_t> dirichlet_cells[d],
          int mg_levels,int interior_smoothing_iterations_input,int boundary_smoothing_iterations_input)
    :Base(true,false),mu_field(mu_field_input),lambda_field(lambda_field_input),dx(dx_input),
     interior_smoothing_iterations(interior_smoothing_iterations_input),boundary_smoothing_iterations(boundary_smoothing_iterations_input)
{
    // Compilation: 2.55s
    Flaging<T_STRUCT,T_FLAG,T,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags,mu_field);
    Flaging<T_STRUCT,T_FLAG,T,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags,dirichlet_cells);
    mg = new MG_Type(allocator,page_map,mg_levels,dx);
    mg->Flag_Hierarchy(&T_STRUCT::flags);
    mg->Create_Dirichlet_Lists(&T_STRUCT::flags);
    Create_Dirichlet_List<T_STRUCT,T_FLAG,log2_page,d>(allocator,page_map.Get_Blocks(),&T_STRUCT::flags,dirichlet);

    {Validate_Blocks<T_STRUCT,log2_page,d> validator;
        if(!validator.Validate_With_Density(allocator,page_map,mu_field)){
            std::cerr<<"(CG_SYSTEM.cpp, Validate_With_Density) Block activation check has faild...Aborting..."<<std::endl;
            abort();
            return;}
        if(!validator.Validate_With_Flag(allocator,page_map,&T_STRUCT::flags)){
            std::cerr<<"(CG_SYSTEM.cpp, Validate_With_Flag) Block activation check has faild...Aborting..."<<std::endl;
            abort();
            return;}}
    if(!mg->Check_Padding()){
        std::cerr << "(CG_SYSTEM.cpp, Check_Padding) Padding violated..." << std::endl;
        abort();
        return;}

    {Validate_Blocks<T_STRUCT,log2_page,d> validator;
        for(int level = 1; level < mg_levels;++level){
            if(!validator.Validate_With_Flag(*(mg->allocators[level]),*(mg->page_maps[level]),&T_STRUCT::flags)){
            std::cerr<<"(CG_SYSTEM.cpp, Validate_With_Flag, level " << level << ") Block activation check has faild...Aborting..."<<std::endl;
            abort();
            return;}}}

    for(int v=0;v<d;++v) s_bk[v]=new T[page_map.Get_Blocks().second*uint64_t(ELEMENTS_PER_BLOCK)];
    
    mg_r_channels[0] = &T_STRUCT::ch6;
    mg_r_channels[1] = &T_STRUCT::ch7;
    mg_r_channels[2] = &T_STRUCT::ch8;
    mg->Build_Matrices(diagonal_channels,&T_STRUCT::flags,mu_field,lambda_field);
    //mg->Invert_Diagonal_Matrix(diagonal_channels,&T_STRUCT::flags);
    //mg->Transpose_Diagonal_Channels(diagonal_channels);
    mg->Factorize_Bottom_Level();
}
//#####################################################################
// Function Multiply
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Multiply(const Vector_Base& v,Vector_Base& result) const
{
    // Compilation: 5.85s
    auto v_field      = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v).field;
    auto result_field = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).field;
    auto blocks       = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).page_map.Get_Blocks();
    auto& allocator   = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).allocator;
    Linear_Elasticity::Multiply(allocator,blocks,v_field,result_field,mu_field,lambda_field,dx);
}
//#####################################################################
// Function Multiply
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Multiply(const Vector_Base& v,Vector_Base& result,T T_STRUCT::* mu_field,T T_STRUCT::* lambda_field) const
{
    // Compilation: 5.85s
    auto v_field      = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v).field;
    auto result_field = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).field;
    auto blocks       = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).page_map.Get_Blocks();
    auto& allocator   = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).allocator;
    Linear_Elasticity::Multiply(allocator,blocks,v_field,result_field,mu_field,lambda_field,dx);
}
//#####################################################################
// Function Residual
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Residual(const Vector_Base& v,const Vector_Base& rhs,Vector_Base& result,T T_STRUCT::* mu_field,T T_STRUCT::* lambda_field) const
{
    // Compilation: 5.48s
    auto v_field      = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v).field;
    auto rhs_field    = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(rhs).field;
    auto result_field = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).field;
    auto blocks       = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).page_map.Get_Blocks();
    auto& allocator   = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(result).allocator;
    Linear_Elasticity::Residual(allocator,blocks,v_field,rhs_field,result_field,mu_field,lambda_field,dx);
    Project(result);
}
//#####################################################################
// Function Inner_Product
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> double
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Inner_Product(const Vector_Base& v1,const Vector_Base& v2) const
{
    // Compilation: 0.04s
    auto& allocator1 = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v1).allocator;
    auto& allocator2 = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v2).allocator;
    auto blocks_pair = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v1).page_map.Get_Blocks();
    // Take dot-product of hierarchy, use doubles for temporaries
    double sum = 0;
    for(int v = 0;v < d;++v){
        auto data1 = allocator1.Get_Const_Array(CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v1).field[v]);
        auto data2 = allocator2.Get_Const_Array(CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(v2).field[v]);
        #pragma omp parallel for reduction(+:sum)
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset = blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e, offset += sizeof(T)){
                sum += data1(offset) * data2(offset);}}}
    return sum;
}
//#####################################################################
// Function Convergence_Norm
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> T
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Convergence_Norm(const Vector_Base& x) const
{
    // Compilation: 0.02s
    auto& allocator = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).allocator;
    auto blocks_pair = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).page_map.Get_Blocks();
    // Take dot-product of hierarchy, use doubles for temporaries
    T max = 0;
    for(int v = 0;v < d;++v){
        auto data = allocator.Get_Const_Array(CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).field[v]);
        #pragma omp parallel for reduction(max: max)
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e, offset += sizeof(T)){
                max = (fabs(data(offset)) > max) ? fabs(data(offset)) : max;}}}
    return max;
}
//#####################################################################
// Function Project
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Project(Vector_Base& x) const
{
    // Compilation: 0.01s
    auto& allocator = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).allocator;
    for (int v = 0; v < d; v++)
    {
        auto v_array = allocator.Get_Array(CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).field[v]);
        #pragma omp parallel for
        for (int i = 0; i < dirichlet[v].size(); i++)
            v_array(dirichlet[v][i]) = 0;
    }
}
//#####################################################################
// Function Set_Boundary_Conditions
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Set_Boundary_Conditions(Vector_Base& x) const
{
    Project(x);
}
//#####################################################################
// Function Project_Nullspace
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Project_Nullspace(Vector_Base& x) const
{
    auto x_fields   = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).field;
    auto blocks     = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).page_map.Get_Blocks();
    auto& allocator = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(x).allocator;
    Project_Null_Space<T_STRUCT,T,T_FLAG,d,log2_page>(allocator,blocks,x_fields,&T_STRUCT::flags,ACTIVE_NODE);
} 
//#####################################################################
// Function Apply_Preconditioner
//#####################################################################
template<class T_STRUCT,class T,class T_FLAG,int d,int log2_page,int simd_width> void
CG_SYSTEM<T_STRUCT,T,T_FLAG,d,log2_page,simd_width>::
Apply_Preconditioner(const Vector_Base& r, Vector_Base& z) const
{
    // Compilation: 213.58s
    auto& allocator   = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(r).allocator;
    auto& page_map    = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(r).page_map;
    if(mg){
        for(int v = 0;v < d;++v){
            auto data = allocator.Get_Const_Array(mg_r_channels[v]);
            auto blocks = page_map.Get_Blocks();
            #pragma omp parallel for
            for(int b = 0;b < blocks.second;b++){
                uint64_t array_offset = b * uint64_t(ELEMENTS_PER_BLOCK);
                uint64_t data_offset  = blocks.first[b];
                for(int i = 0; i < ELEMENTS_PER_BLOCK; i++,array_offset++,data_offset+=sizeof(T)){
                    s_bk[v][array_offset] = data(data_offset);}}}

        T T_STRUCT::* z_channels[d];
        T T_STRUCT::* r_channels[d];
        for(int v = 0;v < d;++v){
            z_channels[v] = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(z).field[v];
            r_channels[v] = CG_VECTOR<T_STRUCT,T,d,log2_page>::Cg_Vector(r).field[v];}
        mg->Clear_Data(0,z_channels);
        mg->Run_VCycle(z_channels,r_channels,mg_r_channels,
                       diagonal_channels,mu_field,lambda_field,
                       &T_STRUCT::flags,interior_smoothing_iterations,
                       boundary_smoothing_iterations);

        for(int v = 0;v < d;++v){
            auto data = allocator.Get_Array(mg_r_channels[v]);
            auto blocks = page_map.Get_Blocks();
            #pragma omp parallel for
            for(int b = 0;b < blocks.second;b++){
                uint64_t array_offset = b * uint64_t(ELEMENTS_PER_BLOCK);
                uint64_t data_offset  = blocks.first[b];
                for(int i = 0; i < ELEMENTS_PER_BLOCK; i++,array_offset++,data_offset+=sizeof(T)){
                    data(data_offset) = s_bk[v][array_offset];}}}

    }else{
        z=r;}
}
//#####################################################################
#ifndef USING_DOUBLE
#ifndef USING_AVX512
template class CG_SYSTEM<Elasticity_Solver_Data<float,uint32_t>,float,uint32_t,3,14,16>;
#else
template class CG_SYSTEM<Elasticity_Solver_Data<float,uint32_t>,float,uint32_t,3,14,8>;
#endif
#else
#ifndef USING_AVX512
template class CG_SYSTEM<Elasticity_Solver_Data<double,uint64_t>,double,uint64_t,3,14,4>;
//template class CG_SYSTEM<Elasticity_Solver_Data<double,uint64_t>,double,uint64_t,3,15,4>;
#else
template class CG_SYSTEM<Elasticity_Solver_Data<double,uint64_t>,double,uint64_t,3,14,8>;
//template class CG_SYSTEM<Elasticity_Solver_Data<double,uint64_t>,double,uint64_t,3,15,8>;
#endif
#endif
