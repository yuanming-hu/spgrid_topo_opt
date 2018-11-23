//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Project_Null_Space_h__
#define __Project_Null_Space_h__

namespace SPGrid{
template<class Struct_Type,typename T,typename T_FLAG,int d,int log2_page>
struct Project_Null_Space
{
    using Allocator_Type   = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Mask_Type       = typename Allocator_Type::Array_mask<T>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;

    Project_Null_Space(Allocator_Type& allocator, const std::pair<const unsigned long*,unsigned> blocks,
                       T Struct_Type::* const u_fields[d], T Struct_Type::* const d_fields[d])
    {
        for(int v = 0;v < d;++v){
            auto U_array=(allocator.Get_Array(u_fields[v]));
            auto D_array=(allocator.Get_Array(d_fields[v]));
            #pragma omp parallel for
            for(int b = 0;b < blocks.second;b++){
                auto offset=blocks.first[b];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                    if(D_array(offset) == 0) U_array(offset) = 0;}}}
    }
    Project_Null_Space(Allocator_Type& allocator, const std::pair<const unsigned long*,unsigned> blocks,
                       T Struct_Type::* const u_fields[d], T_FLAG Struct_Type::* const flag_field, T_FLAG active_flag)
    {
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            auto offset=blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                if(!(allocator.Get_Const_Array(flag_field)(offset) & active_flag))
                    for(int v = 0;v < d;++v) allocator.Get_Array(u_fields[v])(offset) = 0;}}
    }

};
}

#endif
