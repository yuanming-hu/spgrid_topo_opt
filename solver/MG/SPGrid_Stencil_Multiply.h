//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Stencil_Multiply_h__
#define __SPGrid_Stencil_Multiply_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "../Flags.h"

using namespace SPGrid;

template<typename Stencil_Struct_Type,typename Struct_Type,typename T,int log2_page>
class SPGrid_Stencil_Multiply
{
    static constexpr int d = 3;
    using SPG_Allocator            = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using SPG_Page_Map             = SPGrid_Page_Map<log2_page>;
    using Array_Type               = typename SPG_Allocator::Array_type<const T>;
    using Mask_Type                = typename SPG_Allocator::Array_mask<const T>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;

    using SPG_Stencil_Allocator    = SPGrid_Allocator<Stencil_Struct_Type,d>;
    using SPG_Stencil_Page_Map     = SPGrid_Page_Map<>;
    using Stencil_Array_Type       = typename SPG_Stencil_Allocator::Array_type<>;
    using Stencil_Mask_Type        = typename SPG_Stencil_Allocator::Array_mask<>;

public:
    SPGrid_Stencil_Multiply(Stencil_Array_Type stencil_array,
                            SPG_Allocator& allocator,const std::pair<const uint64_t*,unsigned> blocks,
                            T Struct_Type::* u[d],T Struct_Type::* result[d])
    {
        uint64_t neighbor_offsets[3][3][3];
        for(int i = 0;i < 3;++i)
        for(int j = 0;j < 3;++j)
        for(int k = 0;k < 3;++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i-1,j-1,k-1);
        
        auto U0_array = allocator.Get_Const_Array(u[0]);
        auto U1_array = allocator.Get_Const_Array(u[1]);
        auto U2_array = allocator.Get_Const_Array(u[2]);

        auto F0_array = allocator.Get_Array(result[0]);
        auto F1_array = allocator.Get_Array(result[1]);
        auto F2_array = allocator.Get_Array(result[2]);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                const auto& node_index = Mask_Type::LinearToCoord(offset);
                const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(node_index);
                T stencil[3][3][3][3][3];// x,y,z,v(f),w(u);
                T* flat_stencil = &stencil[0][0][0][0][0];
                for(int i = 0;i < 243;++i){
                    flat_stencil[i] = stencil_array(stencil_offset).data[i];}
                T f[d] = {0,0,0};
                for(int i = 0;i < 3;++i)
                for(int j = 0;j < 3;++j)
                for(int k = 0;k < 3;++k){
                    auto neighbor_offset = Mask_Type::Packed_Add(offset,neighbor_offsets[i][j][k]);
                    const T u[d] = {U0_array(neighbor_offset),U1_array(neighbor_offset),U2_array(neighbor_offset)};
                    for(int v = 0;v < 3;++v)
                    for(int w = 0;w < 3;++w){
                        f[v] += u[w] * stencil[i][j][k][v][w];}}
                F0_array(offset) = f[0];
                F1_array(offset) = f[1];
                F2_array(offset) = f[2];}}
    }
};

template<typename Stencil_Struct_Type,typename Struct_Type,typename T,int log2_page>
class SPGrid_Stencil_Colored_Multiply
{
    static constexpr int d = 3;
    using SPG_Allocator            = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using SPG_Page_Map             = SPGrid_Page_Map<log2_page>;
    using Array_Type               = typename SPG_Allocator::Array_type<const T>;
    using Mask_Type                = typename SPG_Allocator::Array_mask<const T>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;

    using SPG_Stencil_Allocator    = SPGrid_Allocator<Stencil_Struct_Type,d>;
    using SPG_Stencil_Page_Map     = SPGrid_Page_Map<>;
    using Stencil_Array_Type       = typename SPG_Stencil_Allocator::Array_type<>;
    using Stencil_Mask_Type        = typename SPG_Stencil_Allocator::Array_mask<>;
    static constexpr int block_xsize = 1<<Mask_Type::block_xbits;
    static constexpr int block_ysize = 1<<Mask_Type::block_ybits;
    static constexpr int block_zsize = 1<<Mask_Type::block_zbits;
    
public:
    SPGrid_Stencil_Colored_Multiply(Stencil_Array_Type stencil_array,                            
                                    SPG_Allocator& allocator,const std::pair<const uint64_t*,unsigned> blocks,
                                    T Struct_Type::* u[d],T Struct_Type::* result[d], int color)
    {
        uint64_t neighbor_offsets[3][3][3];
        for(int i = 0;i < 3;++i)
        for(int j = 0;j < 3;++j)
        for(int k = 0;k < 3;++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i-1,j-1,k-1);
        
        auto U0_array = allocator.Get_Const_Array(u[0]);
        auto U1_array = allocator.Get_Const_Array(u[1]);
        auto U2_array = allocator.Get_Const_Array(u[2]);

        auto F0_array = allocator.Get_Array(result[0]);
        auto F1_array = allocator.Get_Array(result[1]);
        auto F2_array = allocator.Get_Array(result[2]);

        const int x_parity = (color >> 2) & 0x1;
        const int y_parity = (color >> 1) & 0x1;
        const int z_parity = color & 0x1;

        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            const auto block_offset = blocks.first[b];
            const auto block_index = Mask_Type::LinearToCoord(block_offset);
            for(int i = x_parity;i < block_xsize;i += 2)
            for(int j = y_parity;j < block_ysize;j += 2)
            for(int k = z_parity;k < block_zsize;k += 2){
                const auto offset = block_offset + sizeof(T) * ((block_zsize*block_ysize) * i + block_zsize * j + k);                
                const std::array<int,d> node_index{block_index[0]+i,block_index[1]+j,block_index[2]+k};
                const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(node_index);
                T stencil[3][3][3][3][3];// x,y,z,v(f),w(u);
                T* flat_stencil = &stencil[0][0][0][0][0];
                for(int i = 0;i < 243;++i){
                    flat_stencil[i] = stencil_array(stencil_offset).data[i];}
                T f[d] = {0,0,0};
                for(int i = 0;i < 3;++i)
                for(int j = 0;j < 3;++j)
                for(int k = 0;k < 3;++k){
                    auto neighbor_offset = Mask_Type::Packed_Add(offset,neighbor_offsets[i][j][k]);
                    const T u[d] = {U0_array(neighbor_offset),U1_array(neighbor_offset),U2_array(neighbor_offset)};
                    for(int v = 0;v < 3;++v)
                    for(int w = 0;w < 3;++w){
                        f[v] += u[w] * stencil[i][j][k][v][w];}}
                F0_array(offset) = f[0];
                F1_array(offset) = f[1];
                F2_array(offset) = f[2];}}
    }
};

template<typename Stencil_Struct_Type,typename Struct_Type,typename T,typename T_FLAG,int log2_page>
class SPGrid_Stencil_Colored_Smooth
{
    static constexpr int d = 3;
    using SPG_Allocator            = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using SPG_Page_Map             = SPGrid_Page_Map<log2_page>;
    using Array_Type               = typename SPG_Allocator::Array_type<const T>;
    using Mask_Type                = typename SPG_Allocator::Array_mask<const T>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;

    using SPG_Stencil_Allocator    = SPGrid_Allocator<Stencil_Struct_Type,d>;
    using SPG_Stencil_Page_Map     = SPGrid_Page_Map<>;
    using Stencil_Array_Type       = typename SPG_Stencil_Allocator::Array_type<>;
    using Stencil_Mask_Type        = typename SPG_Stencil_Allocator::Array_mask<>;
    static constexpr int block_xsize = 1<<Mask_Type::block_xbits;
    static constexpr int block_ysize = 1<<Mask_Type::block_ybits;
    static constexpr int block_zsize = 1<<Mask_Type::block_zbits;
    
public:
    SPGrid_Stencil_Colored_Smooth(Stencil_Array_Type stencil_array,                            
                                  SPG_Allocator& allocator,const std::pair<const uint64_t*,unsigned> blocks,
                                  T Struct_Type::* u[d],T Struct_Type::* b[d], T_FLAG Struct_Type::* flag,
                                  const int color,const bool boundary_smoothing=false)
    {
        uint64_t neighbor_offsets[3][3][3];
        for(int i = 0;i < 3;++i)
        for(int j = 0;j < 3;++j)
        for(int k = 0;k < 3;++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i-1,j-1,k-1);
        
        auto U0_array = allocator.Get_Array(u[0]);
        auto U1_array = allocator.Get_Array(u[1]);
        auto U2_array = allocator.Get_Array(u[2]);

        auto F0_array = allocator.Get_Const_Array(b[0]);
        auto F1_array = allocator.Get_Const_Array(b[1]);
        auto F2_array = allocator.Get_Const_Array(b[2]);

        auto Flag_array = allocator.Get_Const_Array(flag);

        const int x_parity = (color >> 2) & 0x1;
        const int y_parity = (color >> 1) & 0x1;
        const int z_parity = color & 0x1;

        static_assert(sizeof(T_FLAG)==sizeof(T),"Currently only supports sizeof(T_FLAG)==sizeof(T)");

        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            const auto block_offset = blocks.first[b];
            const auto block_index = Mask_Type::LinearToCoord(block_offset);
            for(int i = x_parity;i < block_xsize;i += 2)
            for(int j = y_parity;j < block_ysize;j += 2)
            for(int k = z_parity;k < block_zsize;k += 2){
                const auto offset = block_offset + sizeof(T) * ((block_zsize*block_ysize) * i + block_zsize * j + k);                
                if(Flag_array(offset)&ACTIVE_NODE){
                    if(boundary_smoothing&&(!(Flag_array(offset)&BOUNDARY_NODE))) continue;
                    const std::array<int,d> node_index{block_index[0]+i,block_index[1]+j,block_index[2]+k};
                    const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(node_index);
                    T stencil[3][3][3][3][3];// x,y,z,v(f),w(u);
                    T* flat_stencil = &stencil[0][0][0][0][0];
                    for(int i = 0;i < 243;++i){
                        flat_stencil[i] = stencil_array(stencil_offset).data[i];}
                    T r[d] = {F0_array(offset),F1_array(offset),F2_array(offset)};
                    T diag[6] = {
                        stencil[1][1][1][0][0],
                        stencil[1][1][1][1][1],
                        stencil[1][1][1][2][2],
                        stencil[1][1][1][0][1],
                        stencil[1][1][1][0][2],
                        stencil[1][1][1][1][2]};
                    
                    //invert the diagonal
                    if(Flag_array(offset) & DIRICHLET_NODE_X){
                        diag[0] = 1.0;
                        diag[3] = 0;
                        diag[4] = 0;}
                    if(Flag_array(offset) & DIRICHLET_NODE_Y){
                        diag[1] = 1.0;
                        diag[3] = 0;
                        diag[5] = 0;}
                    if(Flag_array(offset) & DIRICHLET_NODE_Z){
                        diag[2] = 1.0;
                        diag[4] = 0;
                        diag[5] = 0;}
                    const T Minor_11 = diag[1] * diag[2] - diag[5] * diag[5];
                    const T Minor_12 = diag[5] * diag[4] - diag[2] * diag[3];
                    const T Minor_13 = diag[3] * diag[5] - diag[4] * diag[1];
                    const T Minor_22 = diag[2] * diag[0] - diag[4] * diag[4];
                    const T Minor_23 = diag[4] * diag[3] - diag[0] * diag[5];
                    const T Minor_33 = diag[0] * diag[1] - diag[3] * diag[3];
                    const T determinant = diag[0] * Minor_11 + diag[3] * Minor_12 + diag[4] * Minor_13;
                    const T one_over_determinant = T(1.0) / determinant;
                    diag[0] = Minor_11 * one_over_determinant;
                    diag[1] = Minor_22 * one_over_determinant;
                    diag[2] = Minor_33 * one_over_determinant;
                    diag[3] = Minor_12 * one_over_determinant;
                    diag[4] = Minor_13 * one_over_determinant;
                    diag[5] = Minor_23 * one_over_determinant;
                    if(Flag_array(offset) & DIRICHLET_NODE_X){
                        diag[0] = 0;
                        diag[3] = 0;
                        diag[4] = 0;}
                    if(Flag_array(offset) & DIRICHLET_NODE_Y){
                        diag[1] = 0;
                        diag[3] = 0;
                        diag[5] = 0;}
                    if(Flag_array(offset) & DIRICHLET_NODE_Z){
                        diag[2] = 0;
                        diag[4] = 0;
                        diag[5] = 0;}
                    
                    for(int i = 0;i < 3;++i)
                    for(int j = 0;j < 3;++j)
                    for(int k = 0;k < 3;++k){
                        auto neighbor_offset = Mask_Type::Packed_Add(offset,neighbor_offsets[i][j][k]);
                        const T u[d] = {U0_array(neighbor_offset),U1_array(neighbor_offset),U2_array(neighbor_offset)};
                        for(int v = 0;v < 3;++v)
                        for(int w = 0;w < 3;++w){
                            r[v] -= u[w] * stencil[i][j][k][v][w];}}

                    T delta[d]={0,0,0};
                    
                    delta[0] += diag[0] * r[0];
                    delta[0] += diag[3] * r[1];
                    delta[0] += diag[4] * r[2];

                    delta[1] += diag[3] * r[0];
                    delta[1] += diag[1] * r[1];
                    delta[1] += diag[5] * r[2];

                    delta[2] += diag[4] * r[0];
                    delta[2] += diag[5] * r[1];
                    delta[2] += diag[2] * r[2];

                    U0_array(offset)+=delta[0];
                    U1_array(offset)+=delta[1];
                    U2_array(offset)+=delta[2];}}}
    }
};
#endif
