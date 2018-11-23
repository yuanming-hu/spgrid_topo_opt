//#####################################################################
// Copyright (c) 2018, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Transpose_h__
#define __SPGrid_Transpose_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <iostream>
namespace SPGrid{
template<typename Struct_Type,typename T,int log2_page,int d> struct Transpose;
template<typename Struct_Type,typename T,int log2_page>
struct Transpose<Struct_Type,T,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type          = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Data_Mask_Type          = typename Allocator_Type::Array_mask<T>;
    using Page_Map                = SPGrid_Page_Map<log2_page>;
    enum {
        block_xsize = 1<<Data_Mask_Type::block_xbits,
        block_ysize = 1<<Data_Mask_Type::block_ybits,
        block_zsize = 1<<Data_Mask_Type::block_zbits
    };
    using Block_Array             = T(&) [block_xsize][block_ysize][block_zsize];
    static constexpr int elements_per_block = Data_Mask_Type::elements_per_block;
    static void TransposeToColored(Allocator_Type& allocator,const std::pair<const uint64_t*,unsigned> blocks,T Struct_Type::* data_channel)
    {
        auto data = allocator.Get_Array(data_channel);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            T buffer[elements_per_block];
            Block_Array casted_buffer = reinterpret_cast<Block_Array>(buffer[0]);
            auto data_offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,data_offset+=sizeof(T))
                buffer[e] = data(data_offset);
            data_offset = blocks.first[b];
            for(int ci = 0;ci < 2;++ci)
            for(int cj = 0;cj < 2;++cj)
            for(int ck = 0;ck < 2;++ck)
                for(int i = 0;i < block_xsize/2;++i)
                for(int j = 0;j < block_ysize/2;++j)
                for(int k = 0;k < block_zsize/2;++k,data_offset+=sizeof(T)){
                    data(data_offset)=casted_buffer[i*2+ci][j*2+cj][k*2+ck];}
        }
    }
    static void TransposeToFlat(Allocator_Type& allocator,const std::pair<const uint64_t*,unsigned> blocks,T Struct_Type::* data_channel)
    {
        auto data = allocator.Get_Array(data_channel);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            T buffer[elements_per_block];
            Block_Array casted_buffer = reinterpret_cast<Block_Array>(buffer[0]);
            auto data_offset = blocks.first[b];
            for(int ci = 0;ci < 2;++ci)
            for(int cj = 0;cj < 2;++cj)
            for(int ck = 0;ck < 2;++ck)
                for(int i = 0;i < block_xsize/2;++i)
                for(int j = 0;j < block_ysize/2;++j)
                for(int k = 0;k < block_zsize/2;++k,data_offset+=sizeof(T)){
                    casted_buffer[i*2+ci][j*2+cj][k*2+ck]=data(data_offset);}
            data_offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,data_offset+=sizeof(T))
                data(data_offset)=buffer[e];
        }
    }

};
}
#endif
