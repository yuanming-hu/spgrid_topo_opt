//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Restriction_h__
#define __Restriction_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <vector>
template<typename Struct_Type,typename T,int log2_page,int d> struct Restriction;
template<typename Struct_Type,typename T,int log2_page>
struct Restriction<Struct_Type,T,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Mask_Type       = typename Allocator_Type::Array_mask<T>;
    enum {
        block_xsize = 1<<Mask_Type::block_xbits,
        block_ysize = 1<<Mask_Type::block_ybits,
        block_zsize = 1<<Mask_Type::block_zbits
    };
    Restriction(Allocator_Type& coarse_allocator,const Allocator_Type& fine_allocator,std::pair<const uint64_t*,unsigned> coarse_blocks,
                T Struct_Type::* coarse_channel,T Struct_Type::* fine_channel)
    {
        auto fine_array = fine_allocator.Get_Const_Array(fine_channel);
        auto coarse_array = coarse_allocator.Get_Array(coarse_channel);
        constexpr T weights[3] = {0.5,1.0,0.5};
        uint64_t neighbor_offsets[3][3][3];
        for(int i = 0;i < 3; ++i)
        for(int j = 0;j < 3; ++j)
        for(int k = 0;k < 3; ++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i-1,j-1,k-1);
        
        #pragma omp parallel for
        for(int b = 0;b < coarse_blocks.second;++b){
            auto offset = coarse_blocks.first[b];
            for(int i = 0; i < block_xsize;++i){
            for(int j = 0; j < block_ysize;++j){
            for(int k = 0; k < block_zsize;++k,offset += sizeof(T)){
                T value = 0;
                auto upsampled_offset = Mask_Type::template UpsampleOffset<block_xsize,block_ysize,block_zsize>(offset);
                for(int ii = 0;ii < 3; ++ii){
                const T weight_i = weights[ii];
                for(int jj = 0;jj < 3; ++jj){
                const T weight_j = weights[jj];
                for(int kk = 0;kk < 3; ++kk){
                    const T weight_k = weights[kk];
                    const T weight = weight_i * weight_j * weight_k;
                    value += weight * fine_array(Mask_Type::Packed_Add(upsampled_offset,neighbor_offsets[ii][jj][kk]));}}}
                coarse_array(offset) = value;}}}
        }        
    }
    Restriction(Allocator_Type& coarse_allocator,const Allocator_Type& fine_allocator,std::pair<const uint64_t*,unsigned> coarse_blocks,
                T Struct_Type::* coarse_channel[d],T Struct_Type::* fine_channel[d])
    {
        auto fine_x_array = fine_allocator.Get_Const_Array(fine_channel[0]);
        auto fine_y_array = fine_allocator.Get_Const_Array(fine_channel[1]);
        auto fine_z_array = fine_allocator.Get_Const_Array(fine_channel[2]);
        auto coarse_x_array = coarse_allocator.Get_Array(coarse_channel[0]);
        auto coarse_y_array = coarse_allocator.Get_Array(coarse_channel[1]);
        auto coarse_z_array = coarse_allocator.Get_Array(coarse_channel[2]);
        constexpr T weights[3] = {0.5,1.0,0.5};
        uint64_t neighbor_offsets[3][3][3];
        for(int i = 0;i < 3; ++i)
        for(int j = 0;j < 3; ++j)
        for(int k = 0;k < 3; ++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i-1,j-1,k-1);
        
        #pragma omp parallel for
        for(int b = 0;b < coarse_blocks.second;++b){
            auto offset = coarse_blocks.first[b];
            for(int i = 0; i < block_xsize;++i){
            for(int j = 0; j < block_ysize;++j){
            for(int k = 0; k < block_zsize;++k,offset += sizeof(T)){
                T value_x = 0;
                T value_y = 0;
                T value_z = 0;
                auto upsampled_offset = Mask_Type::template UpsampleOffset<block_xsize,block_ysize,block_zsize>(offset);
                for(int ii = 0;ii < 3; ++ii){
                const T weight_i = weights[ii];
                for(int jj = 0;jj < 3; ++jj){
                const T weight_j = weights[jj];
                for(int kk = 0;kk < 3; ++kk){
                    const T weight_k = weights[kk];
                    const T weight = weight_i * weight_j * weight_k;
                    value_x += weight * fine_x_array(Mask_Type::Packed_Add(upsampled_offset,neighbor_offsets[ii][jj][kk]));
                    value_y += weight * fine_y_array(Mask_Type::Packed_Add(upsampled_offset,neighbor_offsets[ii][jj][kk]));
                    value_z += weight * fine_z_array(Mask_Type::Packed_Add(upsampled_offset,neighbor_offsets[ii][jj][kk]));}}}
                coarse_x_array(offset) = value_x;
                coarse_y_array(offset) = value_y;
                coarse_z_array(offset) = value_z;}}}
        }        
    }
};
#endif
