//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Delete_Dirichlet_Equations_h__
#define __Delete_Dirichlet_Equations_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "Elasticity_Matrix.h"

using namespace SPGrid;

template<typename Stencil_Struct_Type,typename T>
class Delete_Dirichlet_Equations
{
    static constexpr int d = 3;
    using SPG_Stencil_Allocator    = SPGrid_Allocator<Stencil_Struct_Type,d>;
    using SPG_Stencil_Page_Map     = SPGrid_Page_Map<>;
    using Stencil_Array_Type       = typename SPG_Stencil_Allocator::Array_type<>;
    using Stencil_Mask_Type        = typename SPG_Stencil_Allocator::Array_mask<>;
    static constexpr int elements_per_block = Stencil_Mask_Type::elements_per_block;
public:
    Delete_Dirichlet_Equations(SPG_Stencil_Allocator& stencil_allocator,const std::pair<const uint64_t*,unsigned> blocks)
    {
        using Spoke_Stencil_Type = T (&)[3][3][3][3][3];
        auto matrix = stencil_allocator.Get_Array();
        uint64_t neighbor_offsets[3][3][3];
        for(int i = 0;i < 3;++i)
        for(int j = 0;j < 3;++j)
        for(int k = 0;k < 3;++k)
            neighbor_offsets[i][j][k] = Stencil_Mask_Type::Linear_Offset(i-1,j-1,k-1);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset+=Stencil_Mask_Type::field_size){
                Spoke_Stencil_Type stencil = reinterpret_cast<Spoke_Stencil_Type>(matrix(offset).data[0]);
                for(int i = 0;i < 3;++i)
                for(int j = 0;j < 3;++j)
                for(int k = 0;k < 3;++k)
                    if(matrix(offset).flags & Elasticity_Matrix_Flags::Spoke_Active_Uncentered[i][j][k]){
                        uint32_t neighbor_flag = matrix(Stencil_Mask_Type::Packed_Add(neighbor_offsets[i][j][k],offset)).flags;
                        for(int v = 0;v < d;++v)
                        for(int w = 0;w < d;++w)
                            if(matrix(offset).flags & Elasticity_Matrix_Flags::Coordinate_Dirichlet[v]){
                                if(v == w && i == 1 && j == 1 && k == 1){
                                    stencil[i][j][k][v][w] = 1.0f;}
                                else stencil[i][j][k][v][w] = 0.0f;}
                            else if(neighbor_flag & Elasticity_Matrix_Flags::Coordinate_Dirichlet[w]){
                                stencil[i][j][k][v][w] = 0.0f;}}}}
    }
};
#endif
