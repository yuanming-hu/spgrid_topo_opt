//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Prolongation_h__
#define __Prolongation_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <vector>
template<typename Struct_Type,typename T,int log2_page,int d> struct Prolongation;
template<typename Struct_Type,typename T,int log2_page>
struct Prolongation<Struct_Type,T,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Mask_Type       = typename Allocator_Type::Array_mask<T>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    enum {
        block_xsize = 1<<Mask_Type::block_xbits,
        block_ysize = 1<<Mask_Type::block_ybits,
        block_zsize = 1<<Mask_Type::block_zbits
    };
    
    Prolongation(Allocator_Type& fine_allocator,const Allocator_Type& coarse_allocator,std::pair<const uint64_t*,unsigned> fine_blocks,
                 T Struct_Type::* fine_channel,T Struct_Type::* coarse_channel)
    {
        auto fine_array = fine_allocator.Get_Array(fine_channel);
        auto coarse_array = coarse_allocator.Get_Const_Array(coarse_channel);
        constexpr T weights_unaligned[2] = {0.5,0.5};
        constexpr T weights_aligned[2] = {1,0};
        uint64_t neighbor_offsets[2][2][2];
        for(int i = 0;i < 2; ++i)
        for(int j = 0;j < 2; ++j)
        for(int k = 0;k < 2; ++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i,j,k);
        
        #pragma omp parallel for
        for(int b = 0;b < fine_blocks.second;++b){
            auto offset = fine_blocks.first[b];
            for(int i = 0; i < block_xsize;++i){
            for(int j = 0; j < block_ysize;++j){
            for(int k = 0; k < block_zsize;++k,offset += sizeof(T)){
                T value = fine_array(offset); // Prolongation is an accumulation operation
                auto downsampled_offset = Mask_Type::template DownsampleOffset<block_xsize,block_ysize,block_zsize>(offset);
                for(int ii = 0;ii < 2; ++ii){
                const T weight_i = (i & 0x1) ? weights_unaligned[ii] : weights_aligned[ii];
                for(int jj = 0;jj < 2; ++jj){
                const T weight_j = (j & 0x1) ? weights_unaligned[jj] : weights_aligned[jj];
                for(int kk = 0;kk < 2; ++kk){
                    const T weight_k = (k & 0x1) ? weights_unaligned[kk] : weights_aligned[kk];
                    const T weight = weight_i * weight_j * weight_k;
                    if(weight > 0)
                    value += weight * coarse_array(Mask_Type::Packed_Add(downsampled_offset,neighbor_offsets[ii][jj][kk]));}}}
                fine_array(offset) = value;}}}
        }
    }
    Prolongation(Allocator_Type& fine_allocator,const Allocator_Type& coarse_allocator,std::pair<const uint64_t*,unsigned> fine_blocks,
                 T Struct_Type::* fine_channel[d],T Struct_Type::* coarse_channel[d])
    {
        auto fine_x_array = fine_allocator.Get_Array(fine_channel[0]);
        auto fine_y_array = fine_allocator.Get_Array(fine_channel[1]);
        auto fine_z_array = fine_allocator.Get_Array(fine_channel[2]);
        auto coarse_x_array = coarse_allocator.Get_Const_Array(coarse_channel[0]);
        auto coarse_y_array = coarse_allocator.Get_Const_Array(coarse_channel[1]);
        auto coarse_z_array = coarse_allocator.Get_Const_Array(coarse_channel[2]);
        constexpr T weights_unaligned[2] = {0.5,0.5};
        constexpr T weights_aligned[2] = {1,0};
        uint64_t neighbor_offsets[2][2][2];
        for(int i = 0;i < 2; ++i)
        for(int j = 0;j < 2; ++j)
        for(int k = 0;k < 2; ++k)
            neighbor_offsets[i][j][k] = Mask_Type::Linear_Offset(i,j,k);
        
        #pragma omp parallel for
        for(int b = 0;b < fine_blocks.second;++b){
            auto offset = fine_blocks.first[b];
            for(int i = 0; i < block_xsize;++i){
            for(int j = 0; j < block_ysize;++j){
            for(int k = 0; k < block_zsize;++k,offset += sizeof(T)){
                T value_x = fine_x_array(offset); // Prolongation is an accumulation operation
                T value_y = fine_y_array(offset); // Prolongation is an accumulation operation
                T value_z = fine_z_array(offset); // Prolongation is an accumulation operation
                auto downsampled_offset = Mask_Type::template DownsampleOffset<block_xsize,block_ysize,block_zsize>(offset);
                for(int ii = 0;ii < 2; ++ii){
                const T weight_i = (i & 0x1) ? weights_unaligned[ii] : weights_aligned[ii];
                for(int jj = 0;jj < 2; ++jj){
                const T weight_j = (j & 0x1) ? weights_unaligned[jj] : weights_aligned[jj];
                for(int kk = 0;kk < 2; ++kk){
                    const T weight_k = (k & 0x1) ? weights_unaligned[kk] : weights_aligned[kk];
                    const T weight = weight_i * weight_j * weight_k;
                    if(weight > 0){
                        value_x += weight * coarse_x_array(Mask_Type::Packed_Add(downsampled_offset,neighbor_offsets[ii][jj][kk]));
                        value_y += weight * coarse_y_array(Mask_Type::Packed_Add(downsampled_offset,neighbor_offsets[ii][jj][kk]));
                        value_z += weight * coarse_z_array(Mask_Type::Packed_Add(downsampled_offset,neighbor_offsets[ii][jj][kk]));}}}}
                fine_x_array(offset) = value_x;
                fine_y_array(offset) = value_y;
                fine_z_array(offset) = value_z;}}}
        }
    }
};
#endif
