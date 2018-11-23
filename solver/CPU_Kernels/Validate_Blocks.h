//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Validate_Blocks_h__
#define __Validate_Blocks_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "../Flags.h"
#include <vector>

template<typename Struct_Type,int log2_page,int d> struct Validate_Blocks;

template<typename Struct_Type,int log2_page>
struct Validate_Blocks<Struct_Type,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    template<typename T_FLAG>
    bool Validate_With_Flag(const Allocator_Type& allocator,const Page_Map& page_map,
                            T_FLAG Struct_Type::* flags_channel)
    {
        using Flag_Mask_Type  = typename Allocator_Type::Array_mask<T_FLAG>;
        static constexpr int elements_per_block = Flag_Mask_Type::elements_per_block;
        auto flags = allocator.Get_Const_Array(flags_channel);
        //#pragma omp parallel for
        auto blocks = page_map.Get_Blocks();
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                if(flags(offset) & ACTIVE_CELL){
                    if(!(page_map.Test_Page(Flag_Mask_Type::Packed_Offset<0,0,1>(offset)) &&
                         page_map.Test_Page(Flag_Mask_Type::Packed_Offset<0,1,0>(offset)) &&
                         page_map.Test_Page(Flag_Mask_Type::Packed_Offset<0,1,1>(offset)) &&
                         page_map.Test_Page(Flag_Mask_Type::Packed_Offset<1,0,0>(offset)) &&
                         page_map.Test_Page(Flag_Mask_Type::Packed_Offset<1,0,1>(offset)) &&
                         page_map.Test_Page(Flag_Mask_Type::Packed_Offset<1,1,0>(offset)) &&
                         page_map.Test_Page(Flag_Mask_Type::Packed_Offset<1,1,1>(offset)))){
                        auto coord = Flag_Mask_Type::LinearToCoord(offset);
                        std::cout << "Problematic coord (" << coord[0] << ", " << coord[1] << ", "  << coord[2] << ")" << std::endl;
                        return false;}}}}
        return true;
    }
    template<typename T>
    bool Validate_With_Density(const Allocator_Type& allocator,const Page_Map& page_map,
                               T Struct_Type::* density_channel)
    {
        using Data_Mask_Type  = typename Allocator_Type::Array_mask<T>;
        static constexpr int elements_per_block = Data_Mask_Type::elements_per_block;
        auto density = allocator.Get_Const_Array(density_channel);
        //#pragma omp parallel for
        auto blocks = page_map.Get_Blocks();
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                if(density(offset) > 0){
                    if(!(page_map.Test_Page(Data_Mask_Type::Packed_Offset<0,0,1>(offset)) &&
                         page_map.Test_Page(Data_Mask_Type::Packed_Offset<0,1,0>(offset)) &&
                         page_map.Test_Page(Data_Mask_Type::Packed_Offset<0,1,1>(offset)) &&
                         page_map.Test_Page(Data_Mask_Type::Packed_Offset<1,0,0>(offset)) &&
                         page_map.Test_Page(Data_Mask_Type::Packed_Offset<1,0,1>(offset)) &&
                         page_map.Test_Page(Data_Mask_Type::Packed_Offset<1,1,0>(offset)) &&
                         page_map.Test_Page(Data_Mask_Type::Packed_Offset<1,1,1>(offset)))){
                        auto coord = Data_Mask_Type::LinearToCoord(offset);
                        std::cout << "Problematic coord (" << coord[0] << ", " << coord[1] << ", "  << coord[2] << ")" << std::endl;
                        return false;}}}}
        return true;
    }
};
#endif
