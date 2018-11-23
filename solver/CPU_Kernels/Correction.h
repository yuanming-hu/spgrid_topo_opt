#ifndef __CORRECTION_H__
#define __CORRECTION_H__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <SPGrid/Tools/SPGrid_SIMD_Utilities.h>
#include "Stiffness_Matrices.h"

namespace SPGrid{
template<typename T,typename Struct_Type,int log2_page>
struct Correction
{
    static constexpr int d = 3;
    using Allocator_Type   = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Array_Type       = typename Allocator_Type::Array_type<T>;
    using Const_Array_Type = typename Allocator_Type::Array_type<const T>;
    using Mask_Type        = typename Allocator_Type::Array_mask<T>;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;
    static constexpr int block_xsize = 1<<Mask_Type::block_xbits;
    static constexpr int block_ysize = 1<<Mask_Type::block_ybits;
    static constexpr int block_zsize = 1<<Mask_Type::block_zbits;

    static void Run(Allocator_Type& allocator, const std::pair<const unsigned long*,unsigned> blocks,
                    T Struct_Type::* const u_fields[d], const T Struct_Type::* const r_fields[d],
                    const T Struct_Type::* const d_fields[6])
    {
        constexpr int matrix_entry[3][3] =
            {0, 3, 4,
             3, 1, 5,
             4, 5, 2};
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset+=sizeof(T)){
                T r[3];
                for(int v = 0;v < d;++v)
                    r[v] = allocator.Get_Array(r_fields[v])(offset);
                T u[3] = {};
                for(int v = 0;v < d;++v)
                for(int w = 0;w < d;++w)
                    u[v] += allocator.Get_Array(d_fields[matrix_entry[v][w]])(offset) * r[w];
                for(int v = 0;v < d;++v)
                    allocator.Get_Array(u_fields[v])(offset) += u[v] * omega;}}
    }
    template<typename T_FLAG>
    static void Run_Boundary(Allocator_Type& allocator, const std::pair<const unsigned long*,unsigned> blocks,
                             T Struct_Type::* const u_fields[d], const T Struct_Type::* const r_fields[d],
                             const T Struct_Type::* const d_fields[d*2], const T_FLAG Struct_Type::* const flags_field,
                             T_FLAG inclusive_flag, T_FLAG exclusive_flag, T omega)
    {
        constexpr int matrix_entry[3][3] =
            {0, 3, 4,
             3, 1, 5,
             4, 5, 2};
        auto flag = allocator.Get_Const_Array(flags_field);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            auto offset = blocks.first[b];
            auto flag_offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset+=sizeof(T),flag_offset+=sizeof(T_FLAG)){
                if((flag(flag_offset) & inclusive_flag) && !(flag(flag_offset) & exclusive_flag)){
                    T r[3];
                    for(int v = 0;v < d;++v)
                        r[v] = allocator.Get_Array(r_fields[v])(offset);
                    T u[3] = {};
                    for(int v = 0;v < d;++v)
                        for(int w = 0;w < d;++w)
                            u[v] += allocator.Get_Array(d_fields[matrix_entry[v][w]])(offset) * r[w];
                    for(int v = 0;v < d;++v)
                        allocator.Get_Array(u_fields[v])(offset) += u[v] * omega;}}}
    }
    template<typename T_FLAG>
    static void Run_Colored_Boundary(Allocator_Type& allocator, const std::pair<const unsigned long*,unsigned> blocks,
                                     T Struct_Type::* const u_fields[d], const T Struct_Type::* const r_fields[d],
                                     const T Struct_Type::* const d_fields[d*2], const T_FLAG Struct_Type::* const flags_field,
                                     T_FLAG inclusive_flag, T_FLAG exclusive_flag, int color)
    {
        constexpr int matrix_entry[3][3] =
            {0, 3, 4,
             3, 1, 5,
             4, 5, 2};
        auto flag = allocator.Get_Const_Array(flags_field);
        int x_parity = (color >> 2) & 0x1;
        int y_parity = (color >> 1) & 0x1;
        int z_parity = color & 0x1;
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            auto block_offset = blocks.first[b];
            for(int i = x_parity;i < block_xsize;i += 2)
            for(int j = y_parity;j < block_ysize;j += 2)
            for(int k = z_parity;k < block_zsize;k += 2){
                const int index_in_block = block_zsize * block_ysize * i + block_zsize * j + k;
                const auto offset = block_offset + sizeof(T) * index_in_block;
                const auto flag_offset = block_offset + sizeof(T_FLAG) * index_in_block;
                if((flag(flag_offset) & inclusive_flag) && !(flag(flag_offset) & exclusive_flag)){
                    T r[3];
                    for(int v = 0;v < d;++v)
                        r[v] = allocator.Get_Array(r_fields[v])(offset);
                    T u[3] = {};
                    for(int v = 0;v < d;++v)
                        for(int w = 0;w < d;++w)
                            u[v] += allocator.Get_Array(d_fields[matrix_entry[v][w]])(offset) * r[w];
                    for(int v = 0;v < d;++v)
                        allocator.Get_Array(u_fields[v])(offset) += u[v];}}}
    }
    static void Run_Colored(Allocator_Type& allocator, const std::pair<const unsigned long*,unsigned> blocks,
                            T Struct_Type::* const u_fields[d], const T Struct_Type::* const r_fields[d],
                            const T Struct_Type::* const d_fields[d],int color)
    {
        constexpr int matrix_entry[3][3] =
            {0, 3, 4,
             3, 1, 5,
             4, 5, 2};
        int x_parity = (color >> 2) & 0x1;
        int y_parity = (color >> 1) & 0x1;
        int z_parity = color & 0x1;
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;b++){
            const auto block_offset = blocks.first[b];
            for(int i = x_parity;i < block_xsize;i += 2)
            for(int j = y_parity;j < block_ysize;j += 2)
            for(int k = z_parity;k < block_zsize;k += 2){
                const auto offset = block_offset + sizeof(T) * (block_zsize * block_ysize * i + block_zsize * j + k);
                T r[3];
                for(int v = 0;v < d;++v)
                    r[v] = allocator.Get_Array(r_fields[v])(offset);
                T u[3] = {};
                for(int v = 0;v < d;++v)
                for(int w = 0;w < d;++w)
                    u[v] += allocator.Get_Array(d_fields[matrix_entry[v][w]])(offset) * r[w];
                for(int v = 0;v < d;++v)
                    allocator.Get_Array(u_fields[v])(offset) += u[v];}}
    }
};
}
#endif
