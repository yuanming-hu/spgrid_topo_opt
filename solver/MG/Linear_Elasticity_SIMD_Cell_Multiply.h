//#####################################################################
// Copyright (c) 2018, Haixiang Liu, Eftychios Sifakis
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Linear_Elasticity_SIMD_Cell_Multiply_h__
#define __Linear_Elasticity_SIMD_Cell_Multiply_h__
#include <SPGrid/Tools/SPGrid_SIMD_Utilities.h>
#include "Linear_Elasticity_Stiffness_Matrix.h"
#include "Trilinear_Cell_Transfer_Operators_SIMD.h"
using namespace SPGrid;

template<typename T,int SIMD_WIDTH> struct Broadcast_Stiffness_Matrix;
struct Broadcast_Stiffness_Matrix<float,8>
{
    using T = float;
    static constexpr int SIMD_WIDTH = 8;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static SIMD_Type Get(const T* matrix,int offset) {return SIMD_Operations<T,SIMD_WIDTH>::load(matrix);}
    __forceinline static SIMD_Type U_Times_Parameter(const T* parameter,const T* u,const int node)
    {return SIMD_Operations<T,SIMD_WIDTH>::set(u[node]*parameter[0]);}
};

struct Broadcast_Stiffness_Matrix<double,4>
{
    using T = double;
    static constexpr int SIMD_WIDTH = 4;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static SIMD_Type Get(const T* matrix,int offset) {return SIMD_Operations<T,SIMD_WIDTH>::load(matrix+offset);}
    __forceinline static SIMD_Type U_Times_Parameter(const T* parameter,const T* u,const int node)
    {return SIMD_Operations<T,SIMD_WIDTH>::set(u[node]*parameter[0]);}
};

struct Broadcast_Stiffness_Matrix<float,16>
{
    using T = float;
    static constexpr int SIMD_WIDTH = 16;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static SIMD_Type Get(const T* matrix,int offset) {return _mm512_broadcast_f32x8(_mm256_load_ps(matrix));}
    __forceinline static SIMD_Type U_Times_Parameter(const T* parameter,const T* u,const int node)
    {
        __m512 result=_mm512_castps256_ps512(_mm256_set1_ps(u[node]*parameter[0]));
        result= _mm512_insertf32x8(result,_mm256_set1_ps(u[node+Nodes_Per_Cell]*parameter[1]),1);
        return result;
    }
};

struct Broadcast_Stiffness_Matrix<double,8>
{
    using T = double;
    static constexpr int SIMD_WIDTH = 8;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static SIMD_Type Get(const T* matrix,int offset) {return SIMD_Operations<T,SIMD_WIDTH>::load(matrix);}
    __forceinline static SIMD_Type U_Times_Parameter(const T* parameter,const T* u,const int node)
    {return SIMD_Operations<T,SIMD_WIDTH>::set(u[node]*parameter[0]);}
};

template<typename T,int SIMD_WIDTH,int ncells>
struct Linear_Elasticity_SIMD_Cell_Operators
{
    static constexpr int d              = 3;
    static constexpr int Nodes_Per_Cell = 1 << d;
    static constexpr int SIMD_Per_Cell  = (8/SIMD_WIDTH >= 1) ? 8/SIMD_WIDTH : 1;
    static constexpr int nSIMDs         = ncells * Nodes_Per_Cell / SIMD_WIDTH;
    using SIMD_Operations               = SIMD_Operations<T,SIMD_WIDTH>; 
    using SIMD_Utilities                = SPGrid_SIMD_Utilities<T,SIMD_WIDTH>;
    using SIMD_Type                     = typename SIMD_type<T,SIMD_WIDTH>::type;
    using Broadcast_Matrix              = Broadcast_Stiffness_Matrix<T,SIMD_WIDTH>;
    static void Multiply(SIMD_Type u[nSIMDs],SIMD_Type f[d][d][nSIMDs],const T mu[ncells],const T la[ncells])
    {
        static_assert(nSIMDs == 1 || ncells == 1, "Only one cell or one SIMD line!");
        alignas (64) T mU[nSIMDs * SIMD_WIDTH];
        for(int l=0;l<nSIMDs;++l) SIMD_Operations::store(mU+l*SIMD_WIDTH,u[l]);
        for(int v=0;v<d;v++) for(int w=0;w<d;w++) for(int l=0;l<nSIMDs;++l) f[v][w][l]=SIMD_Operations::zero();
        for(int node=0;node<Nodes_Per_Cell;++node){
            auto mu_times_u = Broadcast_Matrix::U_Times_Parameter(mu,mU,node);
            auto la_times_u = Broadcast_Matrix::U_Times_Parameter(la,mU,node);
            for(int v=0;v<d;v++) for(int w=0;w<d;w++) for(int l=0;l<nSIMDs;++l){
                f[v][w][l] = SIMD_Operations::fmadd(Broadcast_Matrix::Get(Km_cell<T>[w][v][node],l*SIMD_WIDTH),mu_times_u,f[v][w][l]);
                f[v][w][l] = SIMD_Operations::fmadd(Broadcast_Matrix::Get(Kl_cell<T>[w][v][node],l*SIMD_WIDTH),la_times_u,f[v][w][l]);}
        }
    }
    static void Multiply_Scalar(SIMD_Type u[nSIMDs],SIMD_Type f[d][d][nSIMDs],const T mu[ncells],const T la[ncells])
    {
        static_assert(nSIMDs == 1 || ncells == 1, "Only one cell or one SIMD line!");
        alignas (64) T mU[nSIMDs * SIMD_WIDTH];
        for(int l=0;l<nSIMDs;++l) SIMD_Operations::store(mU+l*SIMD_WIDTH,u[l]);
        alignas (32) T mF[3][3][ncells][Nodes_Per_Cell] = {};
        for(int v=0;v<3;v++)
        for(int w=0;w<3;w++)
        for(int cell=0;cell<ncells;++cell)
        for(int i=0;i<8;i++)
        for(int j=0;j<8;j++)
            mF[v][w][cell][i] += ( Km_cell<T>[v][w][i][j] * mu[cell] + Kl_cell<T>[v][w][i][j] * la[cell] ) * mU[j + cell * Nodes_Per_Cell];

        for(int v=0;v<3;v++)
        for(int w=0;w<3;w++)
        for(int l=0;l<nSIMDs;++l)
            f[v][w][l] = SIMD_Operations::load(&mF[v][w][0][0] + l * SIMD_WIDTH);

    }
    using Transfer_Operators=Trilinear_Cell_Transfer_Operators_SIMD<T,SIMD_WIDTH>;
    template<typename Data_Array_Type,typename Page_Map_Type>
    static void SPGrid_Galerkin_Multiply(SIMD_Type vU[nSIMDs],SIMD_Type vF[d][d][nSIMDs],
                                  Data_Array_Type mu, Data_Array_Type la,
                                  const Page_Map_Type& page_map,const uint64_t linear_offset[ncells],
                                  const int depth)
    {
        using Mask_Type = typename Data_Array_Type::MASK;
        static constexpr int block_xsize = 1<<Mask_Type::block_xbits;
        static constexpr int block_ysize = 1<<Mask_Type::block_ybits;
        static constexpr int block_zsize = 1<<Mask_Type::block_zbits;        
        if(depth == 0){
            for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int l=0;l<nSIMDs;++l) vF[i][j][l]=SIMD_Operations::zero();
            T mu_cells[ncells];
            T la_cells[ncells];
            bool non_zero=false;
            for(int i=0;i<ncells;++i){
                mu_cells[i]=0;la_cells[i]=0;
                if(page_map.Test_Page(linear_offset[i])){
                    mu_cells[i]=mu(linear_offset[i]);la_cells[i]=la(linear_offset[i]);non_zero=true;}}
            if(non_zero)
                Multiply(vU,vF,mu_cells,la_cells);
            return;}
        // Prolongate input
        SIMD_Type vU_fine[8][nSIMDs];
        Transfer_Operators::Prolongate(vU,vU_fine);

        // Apply operator on refined cells
        SIMD_Type vF_fine_cell[8][3][3][nSIMDs];
        uint64_t upsampled_offset[ncells];
        for(int i=0;i<ncells;++i){upsampled_offset[i]=Mask_Type::template UpsampleOffset<block_xsize,block_ysize,block_zsize>(linear_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[0],vF_fine_cell[0],mu,la,page_map,upsampled_offset,depth-1);
        uint64_t neighbor_offset[ncells];
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<0,0,1>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[1],vF_fine_cell[1],mu,la,page_map,neighbor_offset,depth-1);
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<0,1,0>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[2],vF_fine_cell[2],mu,la,page_map,neighbor_offset,depth-1);
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<0,1,1>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[3],vF_fine_cell[3],mu,la,page_map,neighbor_offset,depth-1);
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<1,0,0>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[4],vF_fine_cell[4],mu,la,page_map,neighbor_offset,depth-1);
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<1,0,1>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[5],vF_fine_cell[5],mu,la,page_map,neighbor_offset,depth-1);
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<1,1,0>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[6],vF_fine_cell[6],mu,la,page_map,neighbor_offset,depth-1);
        for(int i=0;i<ncells;++i){neighbor_offset[i]=Mask_Type::Packed_Offset<1,1,1>(upsampled_offset[i]);}
        SPGrid_Galerkin_Multiply(vU_fine[7],vF_fine_cell[7],mu,la,page_map,neighbor_offset,depth-1);
    
        // Restrict forces
        for(int v=0;v<3;v++)
            for(int w=0;w<3;w++){
                SIMD_Type vF_fine[8][nSIMDs];
                for(int i=0;i<8;i++)
                    for(int l=0;l<nSIMDs;++l)
                        vF_fine[i][l]=vF_fine_cell[i][v][w][l];
                Transfer_Operators::Restrict(vF[v][w],vF_fine);}
    }
    // For performance testing
    void Galerkin_Multiply(SIMD_Type vU[nSIMDs],SIMD_Type vF[d][d][nSIMDs],const T mu[ncells],const T la[ncells],const int depth)
    {
        if(depth == 0){
            for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int l=0;l<nSIMDs;++l) vF[i][j][l]=SIMD_Operations::zero();
            Multiply(vU,vF,mu,la);
            return;}
        // Prolongate input
        SIMD_Type vU_fine[8][nSIMDs];
        Transfer_Operators::Prolongate(vU,vU_fine);

        // Apply operator on refined cells
        SIMD_Type vF_fine_cell[8][3][3][nSIMDs];
        Galerkin_Multiply(vU_fine[0],vF_fine_cell[0],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[1],vF_fine_cell[1],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[2],vF_fine_cell[2],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[3],vF_fine_cell[3],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[4],vF_fine_cell[4],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[5],vF_fine_cell[5],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[6],vF_fine_cell[6],mu,la,depth-1);
        Galerkin_Multiply(vU_fine[7],vF_fine_cell[7],mu,la,depth-1);
        // Restrict forces
        for(int v=0;v<3;v++)
            for(int w=0;w<3;w++){
                SIMD_Type vF_fine[8][nSIMDs];
                for(int i=0;i<8;i++)
                    for(int l=0;l<nSIMDs;++l)
                        vF_fine[i][l]=vF_fine_cell[i][v][w][l];
                Transfer_Operators::Restrict(vF[v][w],vF_fine);}
    }
};

#endif
