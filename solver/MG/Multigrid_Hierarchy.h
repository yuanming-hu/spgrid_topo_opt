//#####################################################################
// Copyright (c) 2017, Haixiang Liu.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Multigrid_Hierarchy_h__
#define __Multigrid_Hierarchy_h__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include "Prolongation.h"
#include "Restriction.h"
#include "Elasticity_Matrix.h"
#include "Galerkin_Coarsen.h"
#include "SPGrid_Stencil_Multiply.h"
#include "Delete_Dirichlet_Equations.h"
#include "../Flags.h"
#include "../CPU_Kernels/Flaging.h"
#include "../CPU_Kernels/Correction.h"
#include "../CPU_Kernels/Clear.h"
#include "../CPU_Kernels/Project.h"
#include "../CPU_Kernels/Project_Null_Space.h"
#include "../CPU_Kernels/SPGrid_Linear_Elasticity.h"
#include "../CPU_Kernels/SPGrid_Gauss_Seidel.h"
#include "../CPU_Kernels/SPGrid_Transpose.h"
#include <vector>
#include <chrono>

using namespace SPGrid;

template<typename Struct_Type,typename T,typename T_FLAG,int log2_page,int d,int simd_width> struct Multigrid_Hierarchy;

template<typename Struct_Type,typename T,typename T_FLAG,int log2_page,int simd_width>
struct Multigrid_Hierarchy<Struct_Type,T,T_FLAG,log2_page,3,simd_width>
{
    static constexpr int d  = 3;
    using Allocator_Type    = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Mask_Type         = typename Allocator_Type::Array_mask<T>;
    using Page_Map_Type     = SPGrid_Page_Map<log2_page>;
    using Linear_Elasticity = SPGrid_Linear_Elasticity<T,Struct_Type,log2_page,simd_width>;
    using Restrict_Helper   = Restriction<Struct_Type,T,log2_page,d>;
    using Prolongate_Helper = Prolongation<Struct_Type,T,log2_page,d>;
    using Correction        = Correction<T,Struct_Type,log2_page>;
    using Clear_Helper      = Clear<Struct_Type,T,log2_page,d>;
    enum {
        block_xsize = 1<<Mask_Type::block_xbits,
        block_ysize = 1<<Mask_Type::block_ybits,
        block_zsize = 1<<Mask_Type::block_zbits
    };
    static constexpr int elements_per_block = Mask_Type::elements_per_block;
    using T_STENCIL_FLAG           = T_FLAG;
    using Stencil_Struct_Type      = Elasticity_Matrix_Data<T>;
    using SPG_Stencil_Allocator    = SPGrid_Allocator<Stencil_Struct_Type,d>;
    using SPG_Stencil_Page_Map     = SPGrid_Page_Map<>;
    using Stencil_Array_Type       = typename SPG_Stencil_Allocator::Array_type<>;
    using Stencil_Mask_Type        = typename SPG_Stencil_Allocator::Array_mask<>;

    std::vector<Allocator_Type*> allocators;
    std::vector<Page_Map_Type*> page_maps;
    std::vector<std::array<ucoord_t,d>> level_sizes;
    std::vector<std::array<std::vector<uint64_t>,d>> dirichlet_lists;
    std::vector<Elasticity_Matrix<T>*> elasticity_matrices;
    std::vector<std::vector<uint64_t>> boundary_blocks;
    int explicit_level;
    const T dx;
    static void Invert_Matrix(const T A[6], T A_inv[6])
    {
        constexpr int matrix_entry[3][3] =
            {0, 3, 4,
             3, 1, 5,
             4, 5, 2};
#if 0
        A_inv[0] = 1.0/A[0];
        A_inv[1] = 1.0/A[1];
        A_inv[2] = 1.0/A[2];
        A_inv[3] = 0;
        A_inv[4] = 0;
        A_inv[5] = 0;
        return;
#endif   
        const T Minor_11 = A[1] * A[2] - A[5] * A[5];
        const T Minor_12 = A[5] * A[4] - A[2] * A[3];
        const T Minor_13 = A[3] * A[5] - A[4] * A[1];
        const T Minor_22 = A[2] * A[0] - A[4] * A[4];
        const T Minor_23 = A[4] * A[3] - A[0] * A[5];
        const T Minor_33 = A[0] * A[1] - A[3] * A[3];
        const T determinant = A[0] * Minor_11 + A[3] * Minor_12 + A[4] * Minor_13;
        if(determinant == 0){
            A_inv[0] = 0;
            A_inv[1] = 0;
            A_inv[2] = 0;
            A_inv[3] = 0;
            A_inv[4] = 0;
            A_inv[5] = 0;
            return;}
        const T one_over_determinant = T(1.0) / determinant;
        A_inv[0] = Minor_11 * one_over_determinant;
        A_inv[1] = Minor_22 * one_over_determinant;
        A_inv[2] = Minor_33 * one_over_determinant;
        A_inv[3] = Minor_12 * one_over_determinant;
        A_inv[4] = Minor_13 * one_over_determinant;
        A_inv[5] = Minor_23 * one_over_determinant;
    }
    int Levels()
    {
        return allocators.size();
    }
    Allocator_Type& Allocator(int level)
    {
        return *allocators[level];
    }
    const Allocator_Type& Allocator(int level) const
    {
        return *allocators[level];
    }
    Page_Map_Type& Page_Map(int level)
    {
        return *page_maps[level];
    }
    const Page_Map_Type& Page_Map(int level) const
    {
        return *page_maps[level];
    }
    Multigrid_Hierarchy(Allocator_Type& top_allocator,Page_Map_Type& top_page_map,int nlevels,const T dx_input,const int explicit_level_input=2)
        :dx(dx_input),explicit_level(explicit_level_input)
    {
        constexpr int block_size[d] = {block_xsize,block_ysize,block_zsize};
        dirichlet_lists.resize(nlevels);
        allocators.resize(nlevels);
        page_maps.resize(nlevels);
        level_sizes.resize(nlevels);
        elasticity_matrices.resize(nlevels);
        boundary_blocks.resize(nlevels);
        allocators[0] = &top_allocator;
        page_maps[0] = &top_page_map;
        level_sizes[0] = std::array<ucoord_t,d>{top_allocator.xsize,top_allocator.ysize,top_allocator.zsize};
        for(int level = 1;level < nlevels;++level){
            for(int v = 0;v < d;++v){
                const int padding = 2 * block_size[v];
                const int fine_level_size = level_sizes[level-1][v] + 1 - padding;
                level_sizes[level][v] = padding + (fine_level_size / 2) + ((fine_level_size % 2) ? 1 : 0);}
            allocators[level] = new Allocator_Type(level_sizes[level]);
            page_maps[level]  = new Page_Map_Type(*allocators[level]);}
        for(int i = 0;i < nlevels;++i){
            elasticity_matrices[i] = NULL;}
    }
    bool Check_Padding()
    {
        const auto top_size = std::array<ucoord_t,d>{allocators[0]->xsize,allocators[0]->ysize,allocators[0]->zsize};
        const auto blocks = page_maps[0] -> Get_Blocks();
        for(int b = 0;b < blocks.second;++b){
            const auto block_offset = blocks.first[b];
            const auto coord = Mask_Type::LinearToCoord(block_offset);
            if(coord[0] < block_xsize || coord[0] >= top_size[0] - 2 * block_xsize ||
               coord[1] < block_ysize || coord[1] >= top_size[1] - 2 * block_ysize ||
               coord[2] < block_zsize || coord[2] >= top_size[2] - 2 * block_zsize){
                return false;}}
        return true;
    }
    void Flag_Hierarchy(T_FLAG Struct_Type::* flags)
    {
        // This function simply assumes the cells on the top level is marked ACTIVE/DIRICHLET.
        const T_FLAG DIRICHLET_CELL_FLAGS[d] = {DIRICHLET_CELL_X,DIRICHLET_CELL_Y,DIRICHLET_CELL_Z};
        const T_FLAG DIRICHLET_NODE_FLAGS[d] = {DIRICHLET_NODE_X,DIRICHLET_NODE_Y,DIRICHLET_NODE_Z};
        using Flag_Mask_Type = typename Allocator_Type::Array_mask<T_FLAG>;
        const int nlevels    = allocators.size();
        for(int i = 1;i < nlevels;++i){
            const auto fine_blocks = page_maps [i-1] -> Get_Blocks();
            auto fine_flags   = allocators[i-1] -> Get_Const_Array(flags);
            auto coarse_flags = allocators[i]   -> Get_Array(flags);
            //NOTE:: NO PARALLEL! WRITE HARZARD!
            for(int b = 0;b < fine_blocks.second;++b){
                auto offset = fine_blocks.first[b];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T_FLAG)){
                    const uint64_t downsampled_offset =
                        Flag_Mask_Type::template DownsampleOffset<block_xsize,block_ysize,block_zsize>(offset);
                    if(fine_flags(offset) & ACTIVE_CELL){
                        coarse_flags(downsampled_offset) |= ACTIVE_CELL;
                        page_maps[i] -> Set_Page(downsampled_offset);
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<0,0,1>(downsampled_offset));
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<0,1,0>(downsampled_offset));
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<0,1,1>(downsampled_offset));
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,0,0>(downsampled_offset));
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,0,1>(downsampled_offset));
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,1,0>(downsampled_offset));
                        page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,1,1>(downsampled_offset));}
                    for(int v = 0;v < d;++v){
                        if(fine_flags(offset) & DIRICHLET_CELL_FLAGS[v]){
                            coarse_flags(downsampled_offset) |= DIRICHLET_CELL_FLAGS[v];
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<0,0,1>(downsampled_offset));
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<0,1,0>(downsampled_offset));
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<0,1,1>(downsampled_offset));
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,0,0>(downsampled_offset));
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,0,1>(downsampled_offset));
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,1,0>(downsampled_offset));
                            page_maps[i] -> Set_Page(Flag_Mask_Type::Packed_Offset<1,1,1>(downsampled_offset));}}}}
            page_maps[i] -> Update_Block_Offsets();}

        for(int i = 0;i < nlevels;++i){
            const auto blocks = page_maps [i] -> Get_Blocks();
            Flag_Nodes<Struct_Type,T_FLAG,log2_page,d>(*allocators[i],blocks,flags);}

        for(int i = 0;i < nlevels;++i){
            const auto blocks = page_maps [i] -> Get_Blocks();
            Flag_Boundary_Nodes<Struct_Type,T_FLAG,log2_page,d>(*allocators[i],blocks,flags);}
        for(int i = 0;i < nlevels;++i){
            const auto blocks = page_maps [i] -> Get_Blocks();
            Get_Boundary_Blocks<Struct_Type,T_FLAG,log2_page,d>(*allocators[i],blocks,boundary_blocks[i],flags);}
    }
    void Create_Dirichlet_Lists(T_FLAG Struct_Type::* flags)
    {
        const int nlevels = allocators.size();
        for(int i = 0; i < nlevels;++i){
            Create_Dirichlet_List<Struct_Type,T_FLAG,log2_page,d>(*(allocators[i]),page_maps[i]->Get_Blocks(),
                                                                  &Struct_Type::flags,dirichlet_lists[i]);}
    }
    void Build_Matrices(T Struct_Type::* diagonal_channels[6],T_FLAG Struct_Type::* flag_channel,
                        T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel)
    {
        const int nlevels = allocators.size();
        //Linear_Elasticity::Get_Diagonal_Matrix(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels,mu_channel,lambda_channel,dx);
        for(int level = (nlevels == 1) ? 0 : explicit_level;level < nlevels;++level){
            Create_Coarsened_Matrix(mu_channel,lambda_channel,flag_channel,level);
            //Get_Diagonal_Matrix_From_Stencil(level,diagonal_channels,flag_channel);
            Delete_Dirichlet_Equations<Stencil_Struct_Type,T>(*elasticity_matrices[level]->allocator,
                                                              elasticity_matrices[level]->page_map->Get_Blocks());
        }
    }
    void Create_Coarsened_Matrix(T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel,T_FLAG Struct_Type::* flag_channel,const int depth)
    {
        elasticity_matrices[depth] = new Elasticity_Matrix<T>(level_sizes[depth]);
        using Galerkin_Helper   = Galerkin_Coarsen<Stencil_Struct_Type,Struct_Type,T,T_FLAG,log2_page,simd_width>;
        auto mu_array     = allocators[0] -> Get_Const_Array(mu_channel);
        auto lambda_array = allocators[0] -> Get_Const_Array(lambda_channel);
        const auto& page_map = *(page_maps[0]);
        auto flag_array = allocators[depth] -> Get_Const_Array(flag_channel);
        const auto blocks = page_maps [depth] -> Get_Blocks();
        auto& stencil_allocator = *(elasticity_matrices[depth]->allocator);
        auto& stencil_page_map = *(elasticity_matrices[depth]->page_map);
        auto stencil_array = stencil_allocator.Get_Array();
        Galerkin_Helper(stencil_array,stencil_page_map,blocks,flag_array,page_map,mu_array,lambda_array,dx,depth);
    }
    void Get_Diagonal_Matrix_From_Stencil(int level,T Struct_Type::* d_channels[d*2],T_FLAG Struct_Type::* flag_channel)
    {
        using STENCIL_TYPE = const T (&)[3][3][3][3][3];
        auto blocks = page_maps[level] -> Get_Blocks();
        auto stencil_array = elasticity_matrices[level]->allocator->Get_Const_Array();
        auto& allocator = *allocators[level];
        auto flags = allocator.Get_Const_Array(flag_channel);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto data_offset = blocks.first[b];
            auto flag_offset = blocks.first[b];
            auto block_index=Mask_Type::LinearToCoord(data_offset);
            for(int i = block_index[0];i < block_index[0]+block_xsize;++i)
            for(int j = block_index[1];j < block_index[1]+block_ysize;++j)
            for(int k = block_index[2];k < block_index[2]+block_zsize;++k,data_offset += sizeof(T),flag_offset += sizeof(T_FLAG)){
                if(flags(flag_offset) & ACTIVE_NODE){
                    const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(i,j,k);
                    STENCIL_TYPE stencil = reinterpret_cast<STENCIL_TYPE>(stencil_array(stencil_offset).data);
                    allocator.Get_Array(d_channels[0])(data_offset)=stencil[1][1][1][0][0];     // xx
                    allocator.Get_Array(d_channels[1])(data_offset)=stencil[1][1][1][1][1];     // yy
                    allocator.Get_Array(d_channels[2])(data_offset)=stencil[1][1][1][2][2];     // zz
                    allocator.Get_Array(d_channels[3])(data_offset)=stencil[1][1][1][0][1];     // xy
                    allocator.Get_Array(d_channels[4])(data_offset)=stencil[1][1][1][0][2];     // xz
                    allocator.Get_Array(d_channels[5])(data_offset)=stencil[1][1][1][1][2];}}}// yz                
    }
    void Transpose_Diagonal_Channels(T Struct_Type::* diagonal_channels[d])
    {
        // Keep diagonal in transposed format.
        using Transpose_Helper = Transpose<Struct_Type,T,log2_page,d>;
        Transpose_Helper::TransposeToColored(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels[0]);
        Transpose_Helper::TransposeToColored(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels[1]);
        Transpose_Helper::TransposeToColored(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels[2]);
        Transpose_Helper::TransposeToColored(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels[3]);
        Transpose_Helper::TransposeToColored(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels[4]);
        Transpose_Helper::TransposeToColored(*allocators[0],page_maps[0]->Get_Blocks(),diagonal_channels[5]);        
    }
    void Invert_Diagonal_Matrix(T Struct_Type::* d_channels[d],T_FLAG Struct_Type::* flag_channel)
    {
        const int nlevels = allocators.size();
        for(int level = 1;level < nlevels;++level)
            Invert_Diagonal_Matrix(level,d_channels,flag_channel);
    }
    void Invert_Diagonal_Matrix(int level,T Struct_Type::* d_channels[d],T_FLAG Struct_Type::* flag_channel)
    {
        auto blocks = page_maps[level] -> Get_Blocks();
        auto& allocator = *allocators[level];
        auto flags = allocator.Get_Const_Array(flag_channel);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto data_offset = blocks.first[b];
            auto flag_offset = blocks.first[b];
            auto block_index=Mask_Type::LinearToCoord(data_offset);
            for(int i = block_index[0];i < block_index[0]+block_xsize;++i)
            for(int j = block_index[1];j < block_index[1]+block_ysize;++j)
            for(int k = block_index[2];k < block_index[2]+block_zsize;++k,data_offset += sizeof(T),flag_offset += sizeof(T_FLAG)){
                if(flags(flag_offset) & ACTIVE_NODE){
                    T diagonal_matrix[6];
                    for(int i = 0;i < 6;++i)
                        diagonal_matrix[i] = allocator.Get_Array(d_channels[i])(data_offset);
                    if(flags(flag_offset) & DIRICHLET_NODE_X){
                        diagonal_matrix[0] = 1.0f;
                        diagonal_matrix[3] = 0;
                        diagonal_matrix[4] = 0;}
                    if(flags(flag_offset) & DIRICHLET_NODE_Y){
                        diagonal_matrix[1] = 1.0f;
                        diagonal_matrix[3] = 0;
                        diagonal_matrix[5] = 0;}
                    if(flags(flag_offset) & DIRICHLET_NODE_Z){
                        diagonal_matrix[2] = 1.0f;
                        diagonal_matrix[4] = 0;
                        diagonal_matrix[5] = 0;}
                    T inverted_diagonal_matrix[6];
                    Invert_Matrix(diagonal_matrix, inverted_diagonal_matrix);
                    if(flags(flag_offset) & DIRICHLET_NODE_X){
                        inverted_diagonal_matrix[0] = 0;
                        inverted_diagonal_matrix[3] = 0;
                        inverted_diagonal_matrix[4] = 0;}
                    if(flags(flag_offset) & DIRICHLET_NODE_Y){
                        inverted_diagonal_matrix[1] = 0;
                        inverted_diagonal_matrix[3] = 0;
                        inverted_diagonal_matrix[5] = 0;}
                    if(flags(flag_offset) & DIRICHLET_NODE_Z){
                        inverted_diagonal_matrix[2] = 0;
                        inverted_diagonal_matrix[4] = 0;
                        inverted_diagonal_matrix[5] = 0;}
                    for(int i = 0;i < 6;++i){
                        allocator.Get_Array(d_channels[i])(data_offset) = inverted_diagonal_matrix[i];}
                }else{
                    for(int i = 0;i < 6;++i)
                        allocator.Get_Array(d_channels[i])(data_offset) = 0.f;}}}
    }
    void Factorize_Bottom_Level()
    {
        Factorize_Level(allocators.size() - 1);
    }
    void Factorize_Level(int level)
    {
        if(elasticity_matrices[level]){
            elasticity_matrices[level]->Build_CSR_Matrix();
            elasticity_matrices[level]->PARDISO_Factorize();}
    }
    void Bottom_Solve(T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d])
    {
        Exact_Solve(allocators.size()-1,u_channels,b_channels);
    }
    void Exact_Solve(int level,T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d])
    {
        auto blocks = page_maps[level] -> Get_Blocks();
        auto stencil_array = elasticity_matrices[level]->allocator->Get_Array();
        auto& allocator = *allocators[level];
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto data_offset = blocks.first[b];
            auto block_index=Mask_Type::LinearToCoord(data_offset);
            for(int i = block_index[0];i < block_index[0]+block_xsize;++i)
            for(int j = block_index[1];j < block_index[1]+block_ysize;++j)
            for(int k = block_index[2];k < block_index[2]+block_zsize;++k,data_offset += sizeof(T)){
                const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(i,j,k);
                for(int v = 0;v < d;++v)
                    stencil_array(stencil_offset).array[v] = allocator.Get_Const_Array(b_channels[v])(data_offset);}}
        elasticity_matrices[level]->PARDISO_Solve();
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto data_offset = blocks.first[b];
            auto block_index=Mask_Type::LinearToCoord(data_offset);
            for(int i = block_index[0];i < block_index[0]+block_xsize;++i)
            for(int j = block_index[1];j < block_index[1]+block_ysize;++j)
            for(int k = block_index[2];k < block_index[2]+block_zsize;++k,data_offset += sizeof(T)){
                const auto stencil_offset = Stencil_Mask_Type::Linear_Offset(i,j,k);
                for(int v = 0;v < d;++v)
                    allocator.Get_Array(u_channels[v])(data_offset) = stencil_array(stencil_offset).array[v];}}
    }
    void Compute_Residual(int level,T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],T Struct_Type::* r_channels[d],
                          T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel)
    {
        if(level != 0 && elasticity_matrices[level]){
            Stencil_Residual(level,u_channels,b_channels,r_channels);
        }else{
            Linear_Elasticity::Residual(*allocators[0],page_maps[0]->Get_Blocks(),u_channels,b_channels,r_channels,mu_channel,lambda_channel,dx);}
    }
    void Colored_Stencil_Residual(int level,T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],
                                  T Struct_Type::* r_channels[d],const int color,bool use_boundary_blocks)
    {
        auto& stencil_allocator = *(elasticity_matrices[level]->allocator);
        auto stencil_array = stencil_allocator.Get_Array();
        auto& allocator = *allocators[level];
        auto blocks     = page_maps[level] -> Get_Blocks();
        if(use_boundary_blocks) blocks =  std::make_pair(&boundary_blocks[level][0],boundary_blocks[level].size());
        SPGrid_Stencil_Colored_Multiply<Stencil_Struct_Type,Struct_Type,T,log2_page>(stencil_array,allocator,blocks,u_channels,r_channels,color);
        const int x_parity = (color >> 2) & 0x1;
        const int y_parity = (color >> 1) & 0x1;
        const int z_parity = color & 0x1;
        #pragma omp parallel for
        for(int block=0;block<blocks.second;block++){
            const auto block_offset = blocks.first[block];
            for(int i = x_parity;i < block_xsize;i += 2)
            for(int j = y_parity;j < block_ysize;j += 2)
            for(int k = z_parity;k < block_zsize;k += 2){
                const auto offset = block_offset + sizeof(T) * (block_zsize*block_ysize * i + block_zsize * j + k);
                for(int v = 0;v < d;++v){
                    auto r = allocator.Get_Array(r_channels[v]);
                    auto b = allocator.Get_Const_Array(b_channels[v]);
                    r(offset) = b(offset) - r(offset);}}}
        for(int v = 0;v < d;++v)
            Project(allocator.Get_Array(r_channels[v]), dirichlet_lists[level][v]);
    }
    void Stencil_Residual(int level,T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],
                          T Struct_Type::* r_channels[d])
    {
        auto& stencil_allocator = *(elasticity_matrices[level]->allocator);
        auto stencil_array = stencil_allocator.Get_Array();
        auto& allocator = *allocators[level];
        const auto blocks     = page_maps[level] -> Get_Blocks();
        SPGrid_Stencil_Multiply<Stencil_Struct_Type,Struct_Type,T,log2_page>(stencil_array,allocator,blocks,u_channels,r_channels);
        for(int v = 0;v < d;++v){
            auto r = allocator.Get_Array(r_channels[v]);
            auto b = allocator.Get_Const_Array(b_channels[v]);
            #pragma omp parallel for
            for(int block=0;block<blocks.second;block++){
                auto offset = blocks.first[block];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                    r(offset) = b(offset) - r(offset);}}}
        for(int v = 0;v < d;++v)
            Project(allocator.Get_Array(r_channels[v]), dirichlet_lists[level][v]);
    }
    void Copy(int level,T Struct_Type::* b_channels[d],T Struct_Type::* r_channels[d])
    {
        auto& allocator = *allocators[level];
        const auto blocks = page_maps[level] -> Get_Blocks();
        for(int v = 0;v < d;++v){
            auto r = allocator.Get_Array(r_channels[v]);
            auto b = allocator.Get_Const_Array(b_channels[v]);
            #pragma omp parallel for
            for(int block=0;block<blocks.second;block++){
                auto offset = blocks.first[block];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                    r(offset) = b(offset);}}}
    }
    void Cast_To_Float(int level,T Struct_Type::* data_channels[d])
    {
        auto& allocator = *allocators[level];
        const auto blocks = page_maps[level] -> Get_Blocks();
        for(int v = 0;v < d;++v){
            auto d = allocator.Get_Array(data_channels[v]);
            #pragma omp parallel for
            for(int block=0;block<blocks.second;block++){
                auto offset = blocks.first[block];
                for(int e = 0;e < elements_per_block;++e,offset += sizeof(T)){
                    float tmp = d(offset);
                    d(offset) = tmp;}}}
    }

    void Stencil_Multiply(int level,T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d])
    {
        auto& stencil_allocator = *(elasticity_matrices[level]->allocator);
        auto stencil_array = stencil_allocator.Get_Array();
        auto& allocator = *allocators[level];
        const auto blocks     = page_maps[level] -> Get_Blocks();
        SPGrid_Stencil_Multiply<Stencil_Struct_Type,Struct_Type,T,log2_page>(stencil_array,allocator,blocks,u_channels,b_channels);
        for(int v = 0;v < d;++v)
            Project(allocator.Get_Array(b_channels[v]), dirichlet_lists[level][v]);
    }    
    void Transpose_Top_Channels_To_Color(T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],
                                         T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel)
    {
        using Transpose_Helper       = Transpose<Struct_Type,T,log2_page,d>;
        using Flag_Transpose_Helper  = Transpose<Struct_Type,T_FLAG,log2_page,d>;
        auto blocks = page_maps[0]->Get_Blocks();
        Flag_Transpose_Helper::TransposeToColored(*allocators[0],blocks,&Struct_Type::flags);

        Transpose_Helper::TransposeToColored(*allocators[0],blocks,u_channels[0]);
        Transpose_Helper::TransposeToColored(*allocators[0],blocks,u_channels[1]);
        Transpose_Helper::TransposeToColored(*allocators[0],blocks,u_channels[2]);

        Transpose_Helper::TransposeToColored(*allocators[0],blocks,b_channels[0]);
        Transpose_Helper::TransposeToColored(*allocators[0],blocks,b_channels[1]);
        Transpose_Helper::TransposeToColored(*allocators[0],blocks,b_channels[2]);
            
        Transpose_Helper::TransposeToColored(*allocators[0],blocks,mu_channel);
        Transpose_Helper::TransposeToColored(*allocators[0],blocks,lambda_channel);        

    }
    void Transpose_Top_Channels_To_Flat(T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],
                                        T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel)
    {
        using Transpose_Helper       = Transpose<Struct_Type,T,log2_page,d>;
        using Flag_Transpose_Helper  = Transpose<Struct_Type,T_FLAG,log2_page,d>;
        auto blocks = page_maps[0]->Get_Blocks();

        Flag_Transpose_Helper::TransposeToFlat(*allocators[0],blocks,&Struct_Type::flags);

        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,u_channels[0]);
        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,u_channels[1]);
        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,u_channels[2]);
            
        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,b_channels[0]);
        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,b_channels[1]);
        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,b_channels[2]);

        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,mu_channel);
        Transpose_Helper::TransposeToFlat(*allocators[0],blocks,lambda_channel);            
    }

    void Colored_Gauss_Seidel_Smooth(int level,T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],T Struct_Type::* d_channels[d*2],
                                     T Struct_Type::* r_channels[d],T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel,
                                     bool forward,bool boundary_smooth = false)
    {
        if(level == 0){
            using GS_Helper = SPGrid_Gauss_Seidel<T,Struct_Type,log2_page,simd_width>;
            auto blocks = page_maps[0]->Get_Blocks();
            if(boundary_smooth) blocks = std::make_pair(&boundary_blocks[level][0],boundary_blocks[level].size());
            if(forward){
                GS_Helper::template Smooth<0>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<1>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<2>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<3>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<4>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<5>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<6>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<7>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
            }else{
                GS_Helper::template Smooth<7>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<6>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<5>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<4>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<3>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<2>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<1>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
                GS_Helper::template Smooth<0>(*allocators[0],blocks,u_channels,&Struct_Type::flags,b_channels,mu_channel,lambda_channel,dx,boundary_smooth);
            }
            return;
        }
        static const int color[2][8] = {
            1, 5, 4, 6, 7, 3, 2, 0,
            0, 2, 3, 7, 6, 4, 5, 1
        };
        const int* c = color[1];
        if(forward) c = color[0];
        auto& stencil_allocator = *(elasticity_matrices[level]->allocator);
        auto stencil_array = stencil_allocator.Get_Array();
        auto& allocator = *allocators[level];
        auto blocks     = page_maps[level] -> Get_Blocks();
        if(boundary_smooth) blocks =  std::make_pair(&boundary_blocks[level][0],boundary_blocks[level].size());        
        for(int i = 0;i < 8;++i){
            SPGrid_Stencil_Colored_Smooth<Stencil_Struct_Type,Struct_Type,T,T_FLAG,log2_page>(stencil_array,allocator,blocks,u_channels,b_channels,&Struct_Type::flags,c[i],boundary_smooth);
            for(int v = 0;v < d;++v)
                Project(allocator.Get_Array(u_channels[v]), dirichlet_lists[level][v]);}
    }
    // Level here is the level of destination. i.e. the fine level
    void Prolongate(int level,T Struct_Type::* fine_channels[d],T Struct_Type::* coarse_channels[d])
    {
        auto fine_blocks = page_maps[level] -> Get_Blocks();
        auto& fine_allocator = *allocators[level];
        auto& coarse_allocator = *allocators[level+1];
        Prolongate_Helper(fine_allocator,coarse_allocator,fine_blocks,fine_channels,coarse_channels);
        for(int v = 0;v < d;++v)
            Project(fine_allocator.Get_Array(fine_channels[v]), dirichlet_lists[level][v]);
    }
    // Level here is the level of destination. i.e. the coarse level
    void Restrict(int level,T Struct_Type::* fine_channels[d],T Struct_Type::* coarse_channels[d])
    {
        auto coarse_blocks = page_maps[level] -> Get_Blocks();
        auto& fine_allocator = *allocators[level - 1];
        auto& coarse_allocator = *allocators[level];
        Restrict_Helper(coarse_allocator,fine_allocator,coarse_blocks,coarse_channels,fine_channels);
        for(int v = 0;v < d;++v)
            Project(coarse_allocator.Get_Array(coarse_channels[v]), dirichlet_lists[level][v]);
    }    
    void Run_VCycle(T Struct_Type::* u_channels[d],T Struct_Type::* b_channels[d],T Struct_Type::* r_channels[d],
                    T Struct_Type::* d_channels[d],T Struct_Type::* mu_channel,T Struct_Type::* lambda_channel,
                    T_FLAG Struct_Type::* flag_channel,
                    const int smooth_iteration,const int boundary_smooth_iteration)
    {
        const int nlevels = allocators.size();
        for(int level = 0;level < nlevels - 1;++level){
            if(level==0) Transpose_Top_Channels_To_Color(u_channels,b_channels,mu_channel,lambda_channel);  
            if(level==0||level>=explicit_level){
                for(int itr = 0;itr < boundary_smooth_iteration;++itr){
                    Colored_Gauss_Seidel_Smooth(level,u_channels,b_channels,d_channels,r_channels,mu_channel,lambda_channel,true,true);
                    Cast_To_Float(level,u_channels);}
                for(int itr = 0;itr < smooth_iteration;++itr){
                    Colored_Gauss_Seidel_Smooth(level,u_channels,b_channels,d_channels,r_channels,mu_channel,lambda_channel,true);
                    Cast_To_Float(level,u_channels);}
                for(int itr = 0;itr < boundary_smooth_iteration;++itr){
                    Colored_Gauss_Seidel_Smooth(level,u_channels,b_channels,d_channels,r_channels,mu_channel,lambda_channel,true,true);
                    Cast_To_Float(level,u_channels);}}
            if(level==0) Transpose_Top_Channels_To_Flat(u_channels,b_channels,mu_channel,lambda_channel);

            Clear_Data(level+1,u_channels);
            Clear_Data(level+1,b_channels);
            if(level==0||level>=explicit_level){
                Compute_Residual(level,u_channels,b_channels,r_channels,mu_channel,lambda_channel);
                Cast_To_Float(level,r_channels);
                Restrict(level+1,r_channels,b_channels);
            }else{
                Restrict(level+1,b_channels,b_channels);}
        }
        Bottom_Solve(u_channels,b_channels);
        Cast_To_Float(nlevels-1,u_channels);
        for(int level = nlevels-2;level >= 0;--level){
            Prolongate(level,u_channels,u_channels);
            Cast_To_Float(level,u_channels);
            Project_Null_Space<Struct_Type,T,T_FLAG,d,log2_page>(*allocators[level],page_maps[level]->Get_Blocks(),u_channels,flag_channel,ACTIVE_NODE);
            if(level==0) Transpose_Top_Channels_To_Color(u_channels,b_channels,mu_channel,lambda_channel);  
            if(level==0||level>=explicit_level){
                for(int itr = 0;itr < boundary_smooth_iteration;++itr){
                    Colored_Gauss_Seidel_Smooth(level,u_channels,b_channels,d_channels,r_channels,mu_channel,lambda_channel,false,true);
                    Cast_To_Float(level,u_channels);
                }
                for(int itr = 0;itr < smooth_iteration;++itr){
                    Colored_Gauss_Seidel_Smooth(level,u_channels,b_channels,d_channels,r_channels,mu_channel,lambda_channel,false);
                    Cast_To_Float(level,u_channels);
                }
                for(int itr = 0;itr < boundary_smooth_iteration;++itr){
                    Colored_Gauss_Seidel_Smooth(level,u_channels,b_channels,d_channels,r_channels,mu_channel,lambda_channel,false,true);
                    Cast_To_Float(level,u_channels);
                }}
            if(level==0) Transpose_Top_Channels_To_Flat(u_channels,b_channels,mu_channel,lambda_channel);}
    }
    void Clear_Data(int level,T Struct_Type::* data_channels[d])
    {
        auto blocks = page_maps[level]->Get_Blocks();
        auto& allocator = *allocators[level];
        for(int v = 0;v < d;++v)
            Clear_Helper(allocator,blocks,data_channels[v]);
    }
    ~Multigrid_Hierarchy()
    {
        const int nlevels = allocators.size();
        for(int i = 1;i < nlevels;++i){
            delete allocators[i];
            delete page_maps[i];}
        for(int i = 0;i < nlevels;++i){
            if(elasticity_matrices[i]) delete elasticity_matrices[i];}
    }

};
#endif
