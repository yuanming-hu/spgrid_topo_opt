//#####################################################################
// Copyright 2015, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_VECTOR
//#####################################################################
#include "CG_VECTOR.h"
#include "../Elasticity_Solver_Data.h"

using namespace SPGrid;
//#####################################################################
// operator+=
//#####################################################################
template<class T_STRUCT, class T, int d,int log2_page> KRYLOV_VECTOR_BASE<T>& CG_VECTOR<T_STRUCT,T,d,log2_page>::
operator+=(const Base& bv)
{
    for(int v = 0;v < d;++v){
        auto d1 = allocator.Get_Array(field[v]);
        auto d2 = Cg_Vector(bv).allocator.Get_Const_Array(Cg_Vector(bv).field[v]);
        auto blocks_pair = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset = blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e,offset += sizeof(T)){
                if(using_float){
                    float tmp=(float)d1(offset)+(float)d2(offset);
                    d1(offset)=tmp;
                }else
                    d1(offset)+=d2(offset);}}}
    return *this;
}
//#####################################################################
// operator-=
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> KRYLOV_VECTOR_BASE<T>& CG_VECTOR<T_STRUCT,T,d,log2_page>::
operator-=(const Base& bv)
{
    for(int v = 0;v < d;++v){
        auto d1 = allocator.Get_Array(field[v]);
        auto d2 = Cg_Vector(bv).allocator.Get_Const_Array(Cg_Vector(bv).field[v]);
        auto blocks_pair = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                if(using_float){
                    float tmp=(float)d1(offset)-(float)d2(offset);
                    d1(offset)=tmp;
                }else
                    d1(offset)-=d2(offset);}}}
    return *this;
}
//#####################################################################
// operator*=
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> KRYLOV_VECTOR_BASE<T>& CG_VECTOR<T_STRUCT,T,d,log2_page>::
operator*=(const T a)
{
    for(int v = 0;v < d;++v){
        auto d = allocator.Get_Array(field[v]);
        auto blocks_pair = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset = blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e, offset += sizeof(T)){
                if(using_float){
                    float tmp=(float)d(offset)*(float)a;
                    d(offset)=tmp;
                }else
                    d(offset)*=a;}}}
    return *this;
}
//#####################################################################
// Function Copy
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> void CG_VECTOR<T_STRUCT,T,d,log2_page>::
Copy(const T c,const Base& bv)
{
    for(int v = 0;v < d;++v){
        auto d1 = allocator.Get_Array(field[v]);
        auto d2 = Cg_Vector(bv).allocator.Get_Const_Array(Cg_Vector(bv).field[v]);
        auto blocks_pair = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset = blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e, offset += sizeof(T)){
                if(using_float){
                    float tmp = (float)c * (float)d2(offset);
                    d1(offset) = tmp;
                }else
                    d1(offset) = c * d2(offset);}}}
}
//#####################################################################
// Function Copy
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> void CG_VECTOR<T_STRUCT,T,d,log2_page>::
Copy(const T c1,const Base& bv1,const Base& bv2)
{
    for(int v = 0;v < d;++v){
        auto d1 = allocator.Get_Array(field[v]);
        auto d2 = Cg_Vector(bv1).allocator.Get_Const_Array(Cg_Vector(bv1).field[v]);
        auto d3 = Cg_Vector(bv2).allocator.Get_Const_Array(Cg_Vector(bv2).field[v]);
        auto blocks_pair = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset = blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e, offset += sizeof(T)){
                if(using_float){
                    float tmp = (float)c1 * (float)d2(offset) + (float)d3(offset);
                    //std::cout<<tmp<<std::endl;
                    d1(offset) = tmp;
                }else
                    d1(offset) = c1 * d2(offset) + d3(offset);}}}
}
//#####################################################################
// Function Raw_Size
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> int CG_VECTOR<T_STRUCT,T,d,log2_page>::
Raw_Size() const
{

}
//#####################################################################
// Function Raw_Get
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> T& CG_VECTOR<T_STRUCT,T,d,log2_page>::
Raw_Get(int i)
{

}
//#####################################################################
// Function Clear
//#####################################################################
template<class T_STRUCT, class T, int d, int log2_page> void CG_VECTOR<T_STRUCT,T,d,log2_page>::
Clear()
{
    for(int v = 0;v < d;++v){
        auto d = allocator.Get_Array(field[v]);
        auto blocks_pair = page_map.Get_Blocks();
        #pragma omp parallel for
        for(int b = 0;b < blocks_pair.second;++b){
            unsigned long offset = blocks_pair.first[b];
            for(int e = 0;e < ELEMENTS_PER_BLOCK;++e, offset += sizeof(T)){
                d(offset) = 0;}}}
}
//#####################################################################
template class CG_VECTOR<Elasticity_Solver_Data<float,uint32_t>,float,3,12>;
template class CG_VECTOR<Elasticity_Solver_Data<float,uint32_t>,float,3,13>;
template class CG_VECTOR<Elasticity_Solver_Data<float,uint32_t>,float,3,14>;
template class CG_VECTOR<Elasticity_Solver_Data<float,uint32_t>,float,3,15>;

template class CG_VECTOR<Elasticity_Solver_Data<double,uint64_t>,double,3,12>;
template class CG_VECTOR<Elasticity_Solver_Data<double,uint64_t>,double,3,13>;
template class CG_VECTOR<Elasticity_Solver_Data<double,uint64_t>,double,3,14>;
template class CG_VECTOR<Elasticity_Solver_Data<double,uint64_t>,double,3,15>;
