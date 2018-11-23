//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Clear_h__
#define __Clear_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <vector>
using namespace SPGrid;
template<typename Struct_Type,typename T,int log2_page,int d>
struct Clear
{
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Mask_Type       = typename Allocator_Type::Array_mask<T>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;

    Clear(Allocator_Type& allocator,const std::pair<const uint64_t*,unsigned> blocks,T Struct_Type::* data_channel)
    {
        auto data = allocator.Get_Array(data_channel);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T))
                data(offset) = 0;}
    }
};
#endif
