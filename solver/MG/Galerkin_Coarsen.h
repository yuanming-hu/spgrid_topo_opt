//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Galerkin_Coarsen_h__
#define __Galerkin_Coarsen_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "Elasticity_Matrix.h"
#include "Linear_Elasticity_SIMD_Cell_Multiply.h"
#include "../Flags.h"

using namespace SPGrid;

template<typename Stencil_Struct_Type,typename Struct_Type,typename T,typename T_FLAG,int log2_page,int simd_width>
class Galerkin_Coarsen
{
    static constexpr int d = 3;
    using SPG_Allocator            = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using SPG_Page_Map             = SPGrid_Page_Map<log2_page>;
    using Array_Type               = typename SPG_Allocator::Array_type<const T>;
    using Mask_Type                = typename SPG_Allocator::Array_mask<const T>;
    using Flag_Array_Type          = typename SPG_Allocator::Array_type<const T_FLAG>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;

    using SPG_Stencil_Allocator    = SPGrid_Allocator<Stencil_Struct_Type,d>;
    using SPG_Stencil_Page_Map     = SPGrid_Page_Map<>;
    using Stencil_Array_Type       = typename SPG_Stencil_Allocator::Array_type<>;
    using Stencil_Mask_Type        = typename SPG_Stencil_Allocator::Array_mask<>;

    static constexpr int ncells    = (simd_width / 8 > 1) ? simd_width / 8 : 1;
    static constexpr int nSIMDs    = ncells * 8 / simd_width;
    using SIMD_Operations          = SIMD_Operations<T,simd_width>; 
    using SIMD_Utilities           = SPGrid_SIMD_Utilities<T,simd_width>;
    using SIMD_Type                = typename SIMD_type<T,simd_width>::type;
    using Cell_Multiply            = Linear_Elasticity_SIMD_Cell_Operators<T,simd_width,ncells>;
    
public:
    Galerkin_Coarsen(Stencil_Array_Type& stencil_array,SPG_Stencil_Page_Map& stencil_page_map,//stencil data
                     const std::pair<const uint64_t*,unsigned> blocks,Flag_Array_Type& flag_array,//coarse level data
                     const SPG_Page_Map& page_map,Array_Type& mu_array,Array_Type& lambda_array,//finest level data
                     const T dx,const int depth)
    {
        static_assert(sizeof(T_FLAG) == sizeof(T),"T and T_FLAG should have the same number of bytes.");        
        uint64_t cell_offsets[8];
        for(int cell = 0;cell < 8;++cell){
            cell_offsets[cell] = Mask_Type::Linear_Offset(((cell>>2)&0x1)-1,((cell>>1)&0x1)-1,(cell&0x1)-1);}
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                if(flag_array(offset) & ACTIVE_NODE){
                    T stencil[3][3][3][3][3]={};// x,y,z,v(f),w(u);
                    T* flat_stencil = &stencil[0][0][0][0][0];
                    const auto& node_index = Mask_Type::LinearToCoord(offset);
                    const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(node_index);
                    stencil_page_map.Set_Page(stencil_offset);
                    for(int cell = 0;cell < 8;cell+=ncells){
                        uint64_t c_offset[ncells];
                        for(int c = 0;c < ncells;++c)
                            c_offset[c] = Mask_Type::Packed_Add(cell_offsets[cell+c],offset);
                        alignas(64) T u[ncells][8] = {};
                        for(int c = 0;c < ncells;++c) u[c][7 - cell - c] = 1.0f;                        
                        SIMD_Type vU[nSIMDs];
                        for(int l = 0;l < nSIMDs;++l) vU[l] = SIMD_Operations::load(&u[0][0]+l*simd_width);
                        SIMD_Type vF[3][3][nSIMDs];
                        Cell_Multiply::SPGrid_Galerkin_Multiply(vU,vF,mu_array,lambda_array,page_map,c_offset,depth);
                        alignas(64) T array[3][3][ncells][8];
                        for(int v = 0;v < 3;++v)
                        for(int w = 0;w < 3;++w)
                        for(int l = 0;l < nSIMDs;++l)
                            SIMD_Operations::store(&array[v][w][0][0]+l*simd_width,vF[v][w][l]);
                        int cell_parity[ncells][3];
                        for(int c = 0;c < ncells;++c){
                            cell_parity[c][0] = (((cell+c)>>2)&0x1);
                            cell_parity[c][1] = (((cell+c)>>1)&0x1);
                            cell_parity[c][2] = (( cell+c    )&0x1);}//{0,1}
                        for(int c = 0;c < ncells;++c)
                        for(int node = 0;node < 8;++node){
                            const int node_parity[3] = {((node>>2)&0x1),((node>>1)&0x1),(node&0x1)}; //{0,1}
                            bool spoke_active = false;
                            const int stencil_index[3] = {cell_parity[c][0]+node_parity[0],
                                                          cell_parity[c][1]+node_parity[1],
                                                          cell_parity[c][2]+node_parity[2]};
                            for(int v = 0;v < 3;++v){
                            for(int w = 0;w < 3;++w){
                                if(array[w][v][c][node] != 0) spoke_active = true;
                                stencil[stencil_index[0]][stencil_index[1]][stencil_index[2]]
                                    [v][w] += array[w][v][c][node];}}
                            if(spoke_active) {
                                stencil_array(stencil_offset).flags |= 
                                    Elasticity_Matrix_Flags::Spoke_Active_Uncentered[stencil_index[0]][stencil_index[1]][stencil_index[2]];}}}
                    for(int v = 0;v < 3;++v) 
                        if(flag_array(offset) & T_FLAG(DIRICHLET_NODE_X << v)){
                            stencil_array(stencil_offset).flags |= Elasticity_Matrix_Flags::Coordinate_Dirichlet[v];}                            
                    for(int i = 0;i < 243;++i){
                        stencil_array(stencil_offset).data[i] = flat_stencil[i]*dx;}}}}
        stencil_page_map.Update_Block_Offsets();
    }
};


#endif
