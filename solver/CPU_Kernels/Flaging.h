//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Flaging_h__
#define __Flaging_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "../Flags.h"
#include <vector>

using namespace SPGrid;

template<typename Struct_Type,typename T_FLAG,typename T,int log2_page,int d> struct Flaging;

template<typename Struct_Type,typename T_FLAG,typename T,int log2_page>
struct Flaging<Struct_Type,T_FLAG,T,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Data_Mask_Type  = typename Allocator_Type::Array_mask<T>;
    using Flag_Mask_Type  = typename Allocator_Type::Array_mask<T_FLAG>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    enum {
        block_xsize = 1<<Data_Mask_Type::block_xbits,
        block_ysize = 1<<Data_Mask_Type::block_ybits,
        block_zsize = 1<<Data_Mask_Type::block_zbits
    };
    static constexpr int elements_per_block = Flag_Mask_Type::elements_per_block;

    Flaging(Allocator_Type& allocator,const std::pair<const uint64_t*,unsigned> blocks,
            T_FLAG Struct_Type::* flags_channel, T Struct_Type::* density_channel)
    {
        auto density = allocator.Get_Const_Array(density_channel);
        auto flags   = allocator.Get_Array(flags_channel);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto flag_offset = blocks.first[b];
            auto data_offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;
                ++e,flag_offset += sizeof(T_FLAG),data_offset += sizeof(T)){
                if(density(data_offset) != 0){
                    flags(flag_offset) |= ACTIVE_CELL;}}}
    }
    Flaging(Allocator_Type& allocator,const std::pair<const uint64_t*,unsigned> blocks,
            T_FLAG Struct_Type::* flags_channel, std::vector<uint64_t> dirichlet_cells[d])
    {
        auto flags = allocator.Get_Array(flags_channel);
        for(int v = 0;v < d;++v){
            #pragma omp parallel for
            for (int i = 0; i < dirichlet_cells[v].size(); i++) flags(dirichlet_cells[v][i]) |= (DIRICHLET_CELL_X << v);}
    }
};

template<typename Struct_Type,typename T_FLAG,int log2_page,int d> struct Flag_Nodes;
template<typename Struct_Type,typename T_FLAG,int log2_page>
struct Flag_Nodes<Struct_Type,T_FLAG,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Flag_Mask_Type  = typename Allocator_Type::Array_mask<T_FLAG>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    static constexpr int elements_per_block = Flag_Mask_Type::elements_per_block;

    Flag_Nodes(Allocator_Type& allocator,std::pair<const uint64_t*,unsigned> blocks,T_FLAG Struct_Type::* flags_channel)
    {
        auto flags = allocator.Get_Array(flags_channel);
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                auto node_flag = (flags(offset) & (ACTIVE_CELL | DIRICHLET_CELL_X |DIRICHLET_CELL_Y | DIRICHLET_CELL_Z)) >> 4;
                flags(offset) |= node_flag;
                (flags.operator()<0,0,1>(offset)) |= node_flag;
                (flags.operator()<0,1,0>(offset)) |= node_flag;
                (flags.operator()<0,1,1>(offset)) |= node_flag;
                (flags.operator()<1,0,0>(offset)) |= node_flag;
                (flags.operator()<1,0,1>(offset)) |= node_flag;
                (flags.operator()<1,1,0>(offset)) |= node_flag;
                (flags.operator()<1,1,1>(offset)) |= node_flag;}}
    }
};

template<typename Struct_Type,typename T_FLAG,int log2_page,int d> struct Flag_Boundary_Nodes;
template<typename Struct_Type,typename T_FLAG,int log2_page>
struct Flag_Boundary_Nodes<Struct_Type,T_FLAG,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Flag_Mask_Type  = typename Allocator_Type::Array_mask<T_FLAG>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    static constexpr int elements_per_block = Flag_Mask_Type::elements_per_block;

    Flag_Boundary_Nodes(Allocator_Type& allocator,std::pair<const uint64_t*,unsigned> blocks,
                        T_FLAG Struct_Type::* flags_channel, const int band = 3)
    {
        uint64_t neighbor_offsets[27];
        for(int i = -1;i <= 1;++i)
        for(int j = -1;j <= 1;++j)
        for(int k = -1;k <= 1;++k)
            neighbor_offsets[i * 9 + j * 3 + k + 13] = Flag_Mask_Type::Linear_Offset(i,j,k);
        auto flags = allocator.Get_Array(flags_channel);
        for(int pass = 0;pass < band;++pass){
            #pragma omp parallel for
            for(int b = 0;b < blocks.second;++b){
                auto offset = blocks.first[b];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                    if(flags(offset) & DIRICHLET_NODE) continue;
                    if(flags(offset) & ACTIVE_NODE){
                        for(int neighbor = 0;neighbor < 27;++neighbor){
                            const auto neighbor_offset = Flag_Mask_Type::Packed_Add(offset,neighbor_offsets[neighbor]);
                            if(flags(neighbor_offset) & (DIRICHLET_NODE | BOUNDARY_NODE)){
                                flags(offset) |= BOUNDARY_NODE_TEMP;
                                break;}}}}}
            //NOW COMMIT THE FLAGS.
            #pragma omp parallel for
            for(int b = 0;b < blocks.second;++b){
                auto offset = blocks.first[b];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                    if(flags(offset) & BOUNDARY_NODE_TEMP){
                        flags(offset) -= BOUNDARY_NODE_TEMP;                        
                        flags(offset) |= BOUNDARY_NODE;}}}}
    }
};

template<typename Struct_Type,typename T_FLAG,int log2_page,int d> struct Get_Boundary_Blocks;
template<typename Struct_Type,typename T_FLAG,int log2_page>
struct Get_Boundary_Blocks<Struct_Type,T_FLAG,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Flag_Mask_Type  = typename Allocator_Type::Array_mask<T_FLAG>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    static constexpr int elements_per_block = Flag_Mask_Type::elements_per_block;

    Get_Boundary_Blocks(Allocator_Type& allocator,std::pair<const uint64_t*,unsigned> blocks,
                        std::vector<uint64_t>& boundary_blocks,T_FLAG Struct_Type::* flags_channel)
    {
        boundary_blocks.resize(0);
        auto flags = allocator.Get_Const_Array(flags_channel);
        for(int b = 0;b < blocks.second;++b){
            const auto block_offset = blocks.first[b];
            auto offset = block_offset;
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                if(flags(offset) & BOUNDARY_NODE) {
                    boundary_blocks.push_back(block_offset);
                    break;}}}
    }
};

template<typename Struct_Type,typename T_FLAG,int log2_page,int d> struct Create_Dirichlet_List;
template<typename Struct_Type,typename T_FLAG,int log2_page>
struct Create_Dirichlet_List<Struct_Type,T_FLAG,log2_page,3>
{
    static constexpr int d = 3;
    using Allocator_Type  = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Flag_Mask_Type  = typename Allocator_Type::Array_mask<T_FLAG>;
    using Page_Map        = SPGrid_Page_Map<log2_page>;
    static constexpr int elements_per_block = Flag_Mask_Type::elements_per_block;
    
    Create_Dirichlet_List(const Allocator_Type& allocator,std::pair<const uint64_t*,unsigned> blocks,
                          const T_FLAG Struct_Type::* flags_channel,std::array<std::vector<uint64_t>,d>& dirichlet_nodes)
    {
        for(int v = 0;v < d;++v) dirichlet_nodes[v].clear();
        auto flags = allocator.Get_Array(flags_channel);
        const uint32_t DIRICHLET_NODE_FLAGS[d] = {DIRICHLET_NODE_X,DIRICHLET_NODE_Y,DIRICHLET_NODE_Z};
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                for(int v = 0;v < d;++v){
                    if(flags(offset) & DIRICHLET_NODE_FLAGS[v]){
                        dirichlet_nodes[v].push_back(offset);}}}}
    }
};

#endif
