#ifndef __SPGRID_GAUSS_SEIDEL_H__
#define __SPGRID_GAUSS_SEIDEL_H__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Tools/SPGrid_Colored_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <SPGrid/Tools/SPGrid_SIMD_Utilities.h>
#include "Stiffness_Matrices.h"
#include "../Flags.h"
#include <iostream>

namespace SPGrid{
template<typename T,typename Struct_Type,int log2_page,int SIMD_WIDTH>
struct SPGrid_Gauss_Seidel
{
    static constexpr int d = 3;
    using Allocator_Type   = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Array_Type       = typename Allocator_Type::Array_type<T>;
    using Const_Array_Type = typename Allocator_Type::Array_type<const T>;
    using Mask_Type        = typename Allocator_Type::Array_mask<T>;
    using SIMD_Operations  = SIMD_Operations<T,SIMD_WIDTH>; 
    using SIMD_Utilities   = SPGrid_SIMD_Utilities<T,SIMD_WIDTH>;
    using SIMD_Type        = typename SIMD_type<T,SIMD_WIDTH>::type;
    using SIMD_Int_Type    = typename SIMD_type<T,SIMD_WIDTH>::int_type;

    using Colored_Allocator_Type   = SPGrid_Colored_Allocator<Struct_Type,d,log2_page>;
    using Colored_Array_Type       = typename Colored_Allocator_Type::Array_type<T>;
    using Const_Colored_Array_Type = typename Colored_Allocator_Type::Array_type<const T>;
    using Colored_Mask_Type        = typename Colored_Allocator_Type::Array_mask<T>;

    static constexpr int elements_per_block = Colored_Mask_Type::elements_per_block;
    static constexpr int block_xsize = 1<<Colored_Mask_Type::block_xbits;
    static constexpr int block_ysize = 1<<Colored_Mask_Type::block_ybits;
    static constexpr int block_zsize = 1<<Colored_Mask_Type::block_zbits;
    
    template<int parity_x,int parity_y,int parity_z,
             int Ix,int Iy,int Iz, //output
             int Jx,int Jy,int Jz> //input
    __forceinline static void Apply_Spoke(uint64_t base_offset,
                                          SIMD_Type f[3],
                                          const Const_Colored_Array_Type u_x[2][2][2],
                                          const Const_Colored_Array_Type u_y[2][2][2],
                                          const Const_Colored_Array_Type u_z[2][2][2],
                                          const Const_Colored_Array_Type mu[2][2][2],
                                          const Const_Colored_Array_Type lambda[2][2][2],
                                          const T (&K_mu_array)[8][8][3][3],const T (&K_la_array)[8][8][3][3])
    {
        constexpr int c_parity_x = ((parity_x + Ix) % 2);
        constexpr int c_parity_y = ((parity_y + Iy) % 2);
        constexpr int c_parity_z = ((parity_z + Iz) % 2);
        constexpr int cell_i = (parity_x == 0 && Ix == 1) ? -1 : 0;
        constexpr int cell_j = (parity_y == 0 && Iy == 1) ? -1 : 0;
        constexpr int cell_k = (parity_z == 0 && Iz == 1) ? -1 : 0;

        constexpr int spoke_x = Jx - Ix;
        constexpr int spoke_y = Jy - Iy;
        constexpr int spoke_z = Jz - Iz;
        constexpr int n_parity_x = ((parity_x + Ix + Jx) % 2);
        constexpr int n_parity_y = ((parity_y + Iy + Jy) % 2);
        constexpr int n_parity_z = ((parity_z + Iz + Jz) % 2);
        constexpr int node_i = (parity_x == 0 && spoke_x == -1) ? -1 :
            ((parity_x == 1 && spoke_x ==  1) ?  1 : 0);
        constexpr int node_j = (parity_y == 0 && spoke_y == -1) ? -1 :
            ((parity_y == 1 && spoke_y ==  1) ?  1 : 0);
        constexpr int node_k = (parity_z == 0 && spoke_z == -1) ? -1 :
            ((parity_z == 1 && spoke_z ==  1) ?  1 : 0);
        
        constexpr int index_in = Jx*4+Jy*2+Jz;
        constexpr int index_ou = Ix*4+Iy*2+Iz;
        
        SIMD_Type Vu_x,Vu_y,Vu_z;
        SIMD_Utilities::template Get_Vector<node_i,node_j,node_k>(base_offset,
                                                                  u_x[n_parity_x][n_parity_y][n_parity_z],
                                                                  u_y[n_parity_x][n_parity_y][n_parity_z],
                                                                  u_z[n_parity_x][n_parity_y][n_parity_z],
                                                                  Vu_x,Vu_y,Vu_z);
      
        SIMD_Type Vmu,Vla;
        SIMD_Utilities::template Get_Vector<cell_i,cell_j,cell_k>(base_offset,
                                                                  mu[c_parity_x][c_parity_y][c_parity_z],
                                                                  lambda[c_parity_x][c_parity_y][c_parity_z],
                                                                  Vmu,Vla);

        SIMD_Type K_xx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][0]);
        SIMD_Type K_xy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][1]);
        SIMD_Type K_xz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][2]);

        SIMD_Type K_yx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][0]);
        SIMD_Type K_yy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][1]);
        SIMD_Type K_yz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][2]);

        SIMD_Type K_zx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][0]);
        SIMD_Type K_zy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][1]);
        SIMD_Type K_zz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][2]);
        
        f[0] = SIMD_Operations::fmadd(K_xx_mu,Vu_x,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xy_mu,Vu_y,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xz_mu,Vu_z,f[0]);

        f[1] = SIMD_Operations::fmadd(K_yx_mu,Vu_x,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yy_mu,Vu_y,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yz_mu,Vu_z,f[1]);

        f[2] = SIMD_Operations::fmadd(K_zx_mu,Vu_x,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zy_mu,Vu_y,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zz_mu,Vu_z,f[2]);
        
        SIMD_Type K_xx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][0]);
        SIMD_Type K_xy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][1]);
        SIMD_Type K_xz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][2]);

        SIMD_Type K_yx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][0]);
        SIMD_Type K_yy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][1]);
        SIMD_Type K_yz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][2]);

        SIMD_Type K_zx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][0]);
        SIMD_Type K_zy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][1]);
        SIMD_Type K_zz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][2]);
        
        f[0] = SIMD_Operations::fmadd(K_xx_la,Vu_x,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xy_la,Vu_y,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xz_la,Vu_z,f[0]);

        f[1] = SIMD_Operations::fmadd(K_yx_la,Vu_x,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yy_la,Vu_y,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yz_la,Vu_z,f[1]);

        f[2] = SIMD_Operations::fmadd(K_zx_la,Vu_x,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zy_la,Vu_y,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zz_la,Vu_z,f[2]);
    }

    template<int color>
    static void Multiply(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                         T Struct_Type::* const u_fields[d],T Struct_Type::* f_fields[d],
                         T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                         const T dx)
    {
        Colored_Allocator_Type colored_allocator(allocator);
        constexpr int parity_x = (color >> 2) & 0x1;
        constexpr int parity_y = (color >> 1) & 0x1;
        constexpr int parity_z =  color       & 0x1;

        const Const_Colored_Array_Type u_x[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[0]),
                colored_allocator.template Get_Const_Array<1>(u_fields[0]),
                colored_allocator.template Get_Const_Array<2>(u_fields[0]),
                colored_allocator.template Get_Const_Array<3>(u_fields[0]),
                colored_allocator.template Get_Const_Array<4>(u_fields[0]),
                colored_allocator.template Get_Const_Array<5>(u_fields[0]),
                colored_allocator.template Get_Const_Array<6>(u_fields[0]),
                colored_allocator.template Get_Const_Array<7>(u_fields[0])
            };        

        const Const_Colored_Array_Type u_y[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[1]),
                colored_allocator.template Get_Const_Array<1>(u_fields[1]),
                colored_allocator.template Get_Const_Array<2>(u_fields[1]),
                colored_allocator.template Get_Const_Array<3>(u_fields[1]),
                colored_allocator.template Get_Const_Array<4>(u_fields[1]),
                colored_allocator.template Get_Const_Array<5>(u_fields[1]),
                colored_allocator.template Get_Const_Array<6>(u_fields[1]),
                colored_allocator.template Get_Const_Array<7>(u_fields[1])
            };

        const Const_Colored_Array_Type u_z[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[2]),
                colored_allocator.template Get_Const_Array<1>(u_fields[2]),
                colored_allocator.template Get_Const_Array<2>(u_fields[2]),
                colored_allocator.template Get_Const_Array<3>(u_fields[2]),
                colored_allocator.template Get_Const_Array<4>(u_fields[2]),
                colored_allocator.template Get_Const_Array<5>(u_fields[2]),
                colored_allocator.template Get_Const_Array<6>(u_fields[2]),
                colored_allocator.template Get_Const_Array<7>(u_fields[2])
            };

        const Const_Colored_Array_Type mu[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(mu_field),
                colored_allocator.template Get_Const_Array<1>(mu_field),
                colored_allocator.template Get_Const_Array<2>(mu_field),
                colored_allocator.template Get_Const_Array<3>(mu_field),
                colored_allocator.template Get_Const_Array<4>(mu_field),
                colored_allocator.template Get_Const_Array<5>(mu_field),
                colored_allocator.template Get_Const_Array<6>(mu_field),
                colored_allocator.template Get_Const_Array<7>(mu_field)
            };

        const Const_Colored_Array_Type la[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(lambda_field),
                colored_allocator.template Get_Const_Array<1>(lambda_field),
                colored_allocator.template Get_Const_Array<2>(lambda_field),
                colored_allocator.template Get_Const_Array<3>(lambda_field),
                colored_allocator.template Get_Const_Array<4>(lambda_field),
                colored_allocator.template Get_Const_Array<5>(lambda_field),
                colored_allocator.template Get_Const_Array<6>(lambda_field),
                colored_allocator.template Get_Const_Array<7>(lambda_field)
            };
        
        Colored_Array_Type f_x = colored_allocator.template Get_Array<color>(f_fields[0]);
        Colored_Array_Type f_y = colored_allocator.template Get_Array<color>(f_fields[1]);
        Colored_Array_Type f_z = colored_allocator.template Get_Array<color>(f_fields[2]);

        SIMD_Type Vdx = SIMD_Operations::set(dx);
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e += SIMD_WIDTH,offset += SIMD_WIDTH * sizeof(T)){
                SIMD_Type f[d];

                f[0] = SIMD_Operations::zero();
                f[1] = SIMD_Operations::zero();
                f[2] = SIMD_Operations::zero();
                
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);


                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                f[0] = SIMD_Operations::mul(f[0],Vdx);
                f[1] = SIMD_Operations::mul(f[1],Vdx);
                f[2] = SIMD_Operations::mul(f[2],Vdx);
                
                SIMD_Operations::store(&f_x(offset),f[0]);
                SIMD_Operations::store(&f_y(offset),f[1]);
                SIMD_Operations::store(&f_z(offset),f[2]);
            }
        }
    }

    template<int color>
    static void Smooth(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                       T Struct_Type::* u_fields[d],
                       T Struct_Type::* const f_fields[d],T Struct_Type::* const d_fields[d],
                       T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                       const T dx)
    {
        Colored_Allocator_Type colored_allocator(allocator);
        constexpr int parity_x = (color >> 2) & 0x1;
        constexpr int parity_y = (color >> 1) & 0x1;
        constexpr int parity_z =  color       & 0x1;
        const Const_Colored_Array_Type u_x[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[0]),
                colored_allocator.template Get_Const_Array<1>(u_fields[0]),
                colored_allocator.template Get_Const_Array<2>(u_fields[0]),
                colored_allocator.template Get_Const_Array<3>(u_fields[0]),
                colored_allocator.template Get_Const_Array<4>(u_fields[0]),
                colored_allocator.template Get_Const_Array<5>(u_fields[0]),
                colored_allocator.template Get_Const_Array<6>(u_fields[0]),
                colored_allocator.template Get_Const_Array<7>(u_fields[0])
            };        

        const Const_Colored_Array_Type u_y[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[1]),
                colored_allocator.template Get_Const_Array<1>(u_fields[1]),
                colored_allocator.template Get_Const_Array<2>(u_fields[1]),
                colored_allocator.template Get_Const_Array<3>(u_fields[1]),
                colored_allocator.template Get_Const_Array<4>(u_fields[1]),
                colored_allocator.template Get_Const_Array<5>(u_fields[1]),
                colored_allocator.template Get_Const_Array<6>(u_fields[1]),
                colored_allocator.template Get_Const_Array<7>(u_fields[1])
            };

        const Const_Colored_Array_Type u_z[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[2]),
                colored_allocator.template Get_Const_Array<1>(u_fields[2]),
                colored_allocator.template Get_Const_Array<2>(u_fields[2]),
                colored_allocator.template Get_Const_Array<3>(u_fields[2]),
                colored_allocator.template Get_Const_Array<4>(u_fields[2]),
                colored_allocator.template Get_Const_Array<5>(u_fields[2]),
                colored_allocator.template Get_Const_Array<6>(u_fields[2]),
                colored_allocator.template Get_Const_Array<7>(u_fields[2])
            };

        const Const_Colored_Array_Type mu[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(mu_field),
                colored_allocator.template Get_Const_Array<1>(mu_field),
                colored_allocator.template Get_Const_Array<2>(mu_field),
                colored_allocator.template Get_Const_Array<3>(mu_field),
                colored_allocator.template Get_Const_Array<4>(mu_field),
                colored_allocator.template Get_Const_Array<5>(mu_field),
                colored_allocator.template Get_Const_Array<6>(mu_field),
                colored_allocator.template Get_Const_Array<7>(mu_field)
            };

        const Const_Colored_Array_Type la[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(lambda_field),
                colored_allocator.template Get_Const_Array<1>(lambda_field),
                colored_allocator.template Get_Const_Array<2>(lambda_field),
                colored_allocator.template Get_Const_Array<3>(lambda_field),
                colored_allocator.template Get_Const_Array<4>(lambda_field),
                colored_allocator.template Get_Const_Array<5>(lambda_field),
                colored_allocator.template Get_Const_Array<6>(lambda_field),
                colored_allocator.template Get_Const_Array<7>(lambda_field)
            };
        
        Const_Colored_Array_Type f_x = colored_allocator.template Get_Const_Array<color>(f_fields[0]);
        Const_Colored_Array_Type f_y = colored_allocator.template Get_Const_Array<color>(f_fields[1]);
        Const_Colored_Array_Type f_z = colored_allocator.template Get_Const_Array<color>(f_fields[2]);

        Colored_Array_Type u_x_center = colored_allocator.template Get_Array<color>(u_fields[0]);
        Colored_Array_Type u_y_center = colored_allocator.template Get_Array<color>(u_fields[1]);
        Colored_Array_Type u_z_center = colored_allocator.template Get_Array<color>(u_fields[2]);
        
        Const_Colored_Array_Type d_xx = colored_allocator.template Get_Const_Array<color>(d_fields[0]);
        Const_Colored_Array_Type d_yy = colored_allocator.template Get_Const_Array<color>(d_fields[1]);
        Const_Colored_Array_Type d_zz = colored_allocator.template Get_Const_Array<color>(d_fields[2]);
        Const_Colored_Array_Type d_xy = colored_allocator.template Get_Const_Array<color>(d_fields[3]);
        Const_Colored_Array_Type d_xz = colored_allocator.template Get_Const_Array<color>(d_fields[4]);
        Const_Colored_Array_Type d_yz = colored_allocator.template Get_Const_Array<color>(d_fields[5]);
        
        SIMD_Type Vdx = SIMD_Operations::set(dx);

        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e += SIMD_WIDTH,offset += SIMD_WIDTH * sizeof(T)){
                SIMD_Type Vf[d];
                SIMD_Type Vr[d];
                SIMD_Type Vu[d];
                SIMD_Type Vd_xx,Vd_yy,Vd_zz,Vd_xy,Vd_xz,Vd_yz;
                
                Vu[0] = SIMD_Operations::load(&u_x_center(offset));
                Vu[1] = SIMD_Operations::load(&u_y_center(offset));
                Vu[2] = SIMD_Operations::load(&u_z_center(offset));

                Vr[0] = SIMD_Operations::load(&f_x(offset));
                Vr[1] = SIMD_Operations::load(&f_y(offset));
                Vr[2] = SIMD_Operations::load(&f_z(offset));

                Vd_xx = SIMD_Operations::load(&d_xx(offset));
                Vd_yy = SIMD_Operations::load(&d_yy(offset));
                Vd_zz = SIMD_Operations::load(&d_zz(offset));
                Vd_xy = SIMD_Operations::load(&d_xy(offset));
                Vd_xz = SIMD_Operations::load(&d_xz(offset));
                Vd_yz = SIMD_Operations::load(&d_yz(offset));
                
                Vf[0] = SIMD_Operations::zero();
                Vf[1] = SIMD_Operations::zero();
                Vf[2] = SIMD_Operations::zero();
                
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);


                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Vf[0] = SIMD_Operations::mul(Vf[0],Vdx);
                Vf[1] = SIMD_Operations::mul(Vf[1],Vdx);
                Vf[2] = SIMD_Operations::mul(Vf[2],Vdx);
                
                Vr[0] = SIMD_Operations::sub(Vr[0],Vf[0]);
                Vr[1] = SIMD_Operations::sub(Vr[1],Vf[1]);
                Vr[2] = SIMD_Operations::sub(Vr[2],Vf[2]);

                Vu[0] = SIMD_Operations::fmadd(Vr[0],Vd_xx,Vu[0]);
                Vu[0] = SIMD_Operations::fmadd(Vr[1],Vd_xy,Vu[0]);
                Vu[0] = SIMD_Operations::fmadd(Vr[2],Vd_xz,Vu[0]);

                Vu[1] = SIMD_Operations::fmadd(Vr[0],Vd_xy,Vu[1]);
                Vu[1] = SIMD_Operations::fmadd(Vr[1],Vd_yy,Vu[1]);
                Vu[1] = SIMD_Operations::fmadd(Vr[2],Vd_yz,Vu[1]);
                
                Vu[2] = SIMD_Operations::fmadd(Vr[0],Vd_xz,Vu[2]);
                Vu[2] = SIMD_Operations::fmadd(Vr[1],Vd_yz,Vu[2]);
                Vu[2] = SIMD_Operations::fmadd(Vr[2],Vd_zz,Vu[2]);

                SIMD_Operations::store(&u_x_center(offset),Vu[0]);
                SIMD_Operations::store(&u_y_center(offset),Vu[1]);
                SIMD_Operations::store(&u_z_center(offset),Vu[2]);
            }
        }
    }

    template<int color,typename T_FLAG>
    static void Smooth(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                       T Struct_Type::* u_fields[d],T_FLAG Struct_Type::* flag_field,
                       T Struct_Type::* const f_fields[d],T Struct_Type::* const d_fields[d],
                       T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                       const T dx,const bool boundary_smooth)
    {
        static_assert(sizeof(T)==sizeof(T_FLAG),"Please make sure that sizeof(T)==sizeof(T_FLAG)");
        Colored_Allocator_Type colored_allocator(allocator);
        constexpr int parity_x = (color >> 2) & 0x1;
        constexpr int parity_y = (color >> 1) & 0x1;
        constexpr int parity_z =  color       & 0x1;
        const Const_Colored_Array_Type u_x[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[0]),
                colored_allocator.template Get_Const_Array<1>(u_fields[0]),
                colored_allocator.template Get_Const_Array<2>(u_fields[0]),
                colored_allocator.template Get_Const_Array<3>(u_fields[0]),
                colored_allocator.template Get_Const_Array<4>(u_fields[0]),
                colored_allocator.template Get_Const_Array<5>(u_fields[0]),
                colored_allocator.template Get_Const_Array<6>(u_fields[0]),
                colored_allocator.template Get_Const_Array<7>(u_fields[0])
            };        

        const Const_Colored_Array_Type u_y[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[1]),
                colored_allocator.template Get_Const_Array<1>(u_fields[1]),
                colored_allocator.template Get_Const_Array<2>(u_fields[1]),
                colored_allocator.template Get_Const_Array<3>(u_fields[1]),
                colored_allocator.template Get_Const_Array<4>(u_fields[1]),
                colored_allocator.template Get_Const_Array<5>(u_fields[1]),
                colored_allocator.template Get_Const_Array<6>(u_fields[1]),
                colored_allocator.template Get_Const_Array<7>(u_fields[1])
            };

        const Const_Colored_Array_Type u_z[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[2]),
                colored_allocator.template Get_Const_Array<1>(u_fields[2]),
                colored_allocator.template Get_Const_Array<2>(u_fields[2]),
                colored_allocator.template Get_Const_Array<3>(u_fields[2]),
                colored_allocator.template Get_Const_Array<4>(u_fields[2]),
                colored_allocator.template Get_Const_Array<5>(u_fields[2]),
                colored_allocator.template Get_Const_Array<6>(u_fields[2]),
                colored_allocator.template Get_Const_Array<7>(u_fields[2])
            };

        const Const_Colored_Array_Type mu[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(mu_field),
                colored_allocator.template Get_Const_Array<1>(mu_field),
                colored_allocator.template Get_Const_Array<2>(mu_field),
                colored_allocator.template Get_Const_Array<3>(mu_field),
                colored_allocator.template Get_Const_Array<4>(mu_field),
                colored_allocator.template Get_Const_Array<5>(mu_field),
                colored_allocator.template Get_Const_Array<6>(mu_field),
                colored_allocator.template Get_Const_Array<7>(mu_field)
            };

        const Const_Colored_Array_Type la[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(lambda_field),
                colored_allocator.template Get_Const_Array<1>(lambda_field),
                colored_allocator.template Get_Const_Array<2>(lambda_field),
                colored_allocator.template Get_Const_Array<3>(lambda_field),
                colored_allocator.template Get_Const_Array<4>(lambda_field),
                colored_allocator.template Get_Const_Array<5>(lambda_field),
                colored_allocator.template Get_Const_Array<6>(lambda_field),
                colored_allocator.template Get_Const_Array<7>(lambda_field)
            };
        
        Const_Colored_Array_Type f_x = colored_allocator.template Get_Const_Array<color>(f_fields[0]);
        Const_Colored_Array_Type f_y = colored_allocator.template Get_Const_Array<color>(f_fields[1]);
        Const_Colored_Array_Type f_z = colored_allocator.template Get_Const_Array<color>(f_fields[2]);
        
        Colored_Array_Type u_x_center = colored_allocator.template Get_Array<color>(u_fields[0]);
        Colored_Array_Type u_y_center = colored_allocator.template Get_Array<color>(u_fields[1]);
        Colored_Array_Type u_z_center = colored_allocator.template Get_Array<color>(u_fields[2]);
        
        Const_Colored_Array_Type d_xx = colored_allocator.template Get_Const_Array<color>(d_fields[0]);
        Const_Colored_Array_Type d_yy = colored_allocator.template Get_Const_Array<color>(d_fields[1]);
        Const_Colored_Array_Type d_zz = colored_allocator.template Get_Const_Array<color>(d_fields[2]);
        Const_Colored_Array_Type d_xy = colored_allocator.template Get_Const_Array<color>(d_fields[3]);
        Const_Colored_Array_Type d_xz = colored_allocator.template Get_Const_Array<color>(d_fields[4]);
        Const_Colored_Array_Type d_yz = colored_allocator.template Get_Const_Array<color>(d_fields[5]);
        
        SIMD_Type Vdx = SIMD_Operations::set(dx);
        auto flag = colored_allocator.template Get_Const_Array<color>(flag_field);

        SIMD_Int_Type VDirichlet_Flag_X = SIMD_Operations::set(T_FLAG(DIRICHLET_NODE_X));
        SIMD_Int_Type VDirichlet_Flag_Y = SIMD_Operations::set(T_FLAG(DIRICHLET_NODE_Y));
        SIMD_Int_Type VDirichlet_Flag_Z = SIMD_Operations::set(T_FLAG(DIRICHLET_NODE_Z));        
        SIMD_Int_Type VBoundary_Flag = SIMD_Operations::set(T_FLAG(BOUNDARY_NODE));

        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e += SIMD_WIDTH,offset += SIMD_WIDTH * sizeof(T)){
                SIMD_Type Vf[d];
                SIMD_Type Vr[d];
                SIMD_Type Vu[d];
                SIMD_Type Vu_orig[d];                
                SIMD_Type Vd_xx,Vd_yy,Vd_zz,Vd_xy,Vd_xz,Vd_yz;
                
                Vu[0] = SIMD_Operations::load(&u_x_center(offset));
                Vu[1] = SIMD_Operations::load(&u_y_center(offset));
                Vu[2] = SIMD_Operations::load(&u_z_center(offset));

                Vu_orig[0] = Vu[0];
                Vu_orig[1] = Vu[1];
                Vu_orig[2] = Vu[2];
                    
                Vr[0] = SIMD_Operations::load(&f_x(offset));
                Vr[1] = SIMD_Operations::load(&f_y(offset));
                Vr[2] = SIMD_Operations::load(&f_z(offset));

                Vd_xx = SIMD_Operations::load(&d_xx(offset));
                Vd_yy = SIMD_Operations::load(&d_yy(offset));
                Vd_zz = SIMD_Operations::load(&d_zz(offset));
                Vd_xy = SIMD_Operations::load(&d_xy(offset));
                Vd_xz = SIMD_Operations::load(&d_xz(offset));
                Vd_yz = SIMD_Operations::load(&d_yz(offset));
                
                SIMD_Int_Type Vflag = SIMD_Operations::load((SIMD_Int_Type*)&flag(offset));
                
                Vf[0] = SIMD_Operations::zero();
                Vf[1] = SIMD_Operations::zero();
                Vf[2] = SIMD_Operations::zero();
                
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);


                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,0>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,1>(offset,Vf,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Vf[0] = SIMD_Operations::mul(Vf[0],Vdx);
                Vf[1] = SIMD_Operations::mul(Vf[1],Vdx);
                Vf[2] = SIMD_Operations::mul(Vf[2],Vdx);
                
                Vr[0] = SIMD_Operations::sub(Vr[0],Vf[0]);
                Vr[1] = SIMD_Operations::sub(Vr[1],Vf[1]);
                Vr[2] = SIMD_Operations::sub(Vr[2],Vf[2]);

                Vu[0] = SIMD_Operations::fmadd(Vr[0],Vd_xx,Vu[0]);
                Vu[0] = SIMD_Operations::fmadd(Vr[1],Vd_xy,Vu[0]);
                Vu[0] = SIMD_Operations::fmadd(Vr[2],Vd_xz,Vu[0]);

                Vu[1] = SIMD_Operations::fmadd(Vr[0],Vd_xy,Vu[1]);
                Vu[1] = SIMD_Operations::fmadd(Vr[1],Vd_yy,Vu[1]);
                Vu[1] = SIMD_Operations::fmadd(Vr[2],Vd_yz,Vu[1]);
                
                Vu[2] = SIMD_Operations::fmadd(Vr[0],Vd_xz,Vu[2]);
                Vu[2] = SIMD_Operations::fmadd(Vr[1],Vd_yz,Vu[2]);
                Vu[2] = SIMD_Operations::fmadd(Vr[2],Vd_zz,Vu[2]);

                if(boundary_smooth){
                    SIMD_Int_Type boundary = SIMD_Operations::andv(Vflag,VBoundary_Flag);
                    Vu[0]=SIMD_Operations::blendv(Vu_orig[0],Vu[0],boundary);
                    Vu[1]=SIMD_Operations::blendv(Vu_orig[1],Vu[1],boundary);
                    Vu[2]=SIMD_Operations::blendv(Vu_orig[2],Vu[2],boundary);}

                SIMD_Int_Type dirichlet_x = SIMD_Operations::andv(Vflag,VDirichlet_Flag_X);
                SIMD_Int_Type dirichlet_y = SIMD_Operations::andv(Vflag,VDirichlet_Flag_Y);
                SIMD_Int_Type dirichlet_z = SIMD_Operations::andv(Vflag,VDirichlet_Flag_Z);

                Vu[0]=SIMD_Operations::blendv(Vu[0],SIMD_Operations::zero(),dirichlet_x);
                Vu[1]=SIMD_Operations::blendv(Vu[1],SIMD_Operations::zero(),dirichlet_y);
                Vu[2]=SIMD_Operations::blendv(Vu[2],SIMD_Operations::zero(),dirichlet_z);
                
                SIMD_Operations::store(&u_x_center(offset),Vu[0]);
                SIMD_Operations::store(&u_y_center(offset),Vu[1]);
                SIMD_Operations::store(&u_z_center(offset),Vu[2]);
            }
        }
    }    

    template<int parity_x,int parity_y,int parity_z,
             int Ix,int Iy,int Iz, //output
             int Jx,int Jy,int Jz> //input
    __forceinline static void Apply_Spoke(uint64_t base_offset,
                                          SIMD_Type f[3],
                                          SIMD_Type d[6],
                                          const Const_Colored_Array_Type u_x[2][2][2],
                                          const Const_Colored_Array_Type u_y[2][2][2],
                                          const Const_Colored_Array_Type u_z[2][2][2],
                                          const Const_Colored_Array_Type mu[2][2][2],
                                          const Const_Colored_Array_Type lambda[2][2][2],
                                          const T (&K_mu_array)[8][8][3][3],const T (&K_la_array)[8][8][3][3])
    {
        constexpr int c_parity_x = ((parity_x + Ix) % 2);
        constexpr int c_parity_y = ((parity_y + Iy) % 2);
        constexpr int c_parity_z = ((parity_z + Iz) % 2);
        constexpr int cell_i = (parity_x == 0 && Ix == 1) ? -1 : 0;
        constexpr int cell_j = (parity_y == 0 && Iy == 1) ? -1 : 0;
        constexpr int cell_k = (parity_z == 0 && Iz == 1) ? -1 : 0;

        constexpr int spoke_x = Jx - Ix;
        constexpr int spoke_y = Jy - Iy;
        constexpr int spoke_z = Jz - Iz;
        constexpr int n_parity_x = ((parity_x + Ix + Jx) % 2);
        constexpr int n_parity_y = ((parity_y + Iy + Jy) % 2);
        constexpr int n_parity_z = ((parity_z + Iz + Jz) % 2);
        constexpr int node_i = (parity_x == 0 && spoke_x == -1) ? -1 :
            ((parity_x == 1 && spoke_x ==  1) ?  1 : 0);
        constexpr int node_j = (parity_y == 0 && spoke_y == -1) ? -1 :
            ((parity_y == 1 && spoke_y ==  1) ?  1 : 0);
        constexpr int node_k = (parity_z == 0 && spoke_z == -1) ? -1 :
            ((parity_z == 1 && spoke_z ==  1) ?  1 : 0);
        
        constexpr int index_in = Jx*4+Jy*2+Jz;
        constexpr int index_ou = Ix*4+Iy*2+Iz;
        
        SIMD_Type Vu_x,Vu_y,Vu_z;
        SIMD_Utilities::template Get_Vector<node_i,node_j,node_k>(base_offset,
                                                                  u_x[n_parity_x][n_parity_y][n_parity_z],
                                                                  u_y[n_parity_x][n_parity_y][n_parity_z],
                                                                  u_z[n_parity_x][n_parity_y][n_parity_z],
                                                                  Vu_x,Vu_y,Vu_z);
      
        SIMD_Type Vmu,Vla;
        SIMD_Utilities::template Get_Vector<cell_i,cell_j,cell_k>(base_offset,
                                                                  mu[c_parity_x][c_parity_y][c_parity_z],
                                                                  lambda[c_parity_x][c_parity_y][c_parity_z],
                                                                  Vmu,Vla);

        SIMD_Type K_xx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][0]);
        SIMD_Type K_xy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][1]);
        SIMD_Type K_xz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][2]);

        SIMD_Type K_yx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][0]);
        SIMD_Type K_yy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][1]);
        SIMD_Type K_yz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][2]);

        SIMD_Type K_zx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][0]);
        SIMD_Type K_zy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][1]);
        SIMD_Type K_zz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][2]);

        if(Jx==Ix&&Jy==Iy&&Jz==Iz){ 
            d[0]=SIMD_Operations::add(d[0],K_xx_mu);
            d[1]=SIMD_Operations::add(d[1],K_yy_mu);
            d[2]=SIMD_Operations::add(d[2],K_zz_mu);
            d[3]=SIMD_Operations::add(d[3],K_xy_mu);
            d[4]=SIMD_Operations::add(d[4],K_xz_mu);
            d[5]=SIMD_Operations::add(d[5],K_yz_mu);}
        
        f[0] = SIMD_Operations::fmadd(K_xx_mu,Vu_x,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xy_mu,Vu_y,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xz_mu,Vu_z,f[0]);

        f[1] = SIMD_Operations::fmadd(K_yx_mu,Vu_x,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yy_mu,Vu_y,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yz_mu,Vu_z,f[1]);

        f[2] = SIMD_Operations::fmadd(K_zx_mu,Vu_x,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zy_mu,Vu_y,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zz_mu,Vu_z,f[2]);
        
        SIMD_Type K_xx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][0]);
        SIMD_Type K_xy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][1]);
        SIMD_Type K_xz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][2]);

        SIMD_Type K_yx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][0]);
        SIMD_Type K_yy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][1]);
        SIMD_Type K_yz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][2]);

        SIMD_Type K_zx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][0]);
        SIMD_Type K_zy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][1]);
        SIMD_Type K_zz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][2]);
        
        if(Jx==Ix&&Jy==Iy&&Jz==Iz){
            d[0]=SIMD_Operations::add(d[0],K_xx_la);
            d[1]=SIMD_Operations::add(d[1],K_yy_la);
            d[2]=SIMD_Operations::add(d[2],K_zz_la);
            d[3]=SIMD_Operations::add(d[3],K_xy_la);
            d[4]=SIMD_Operations::add(d[4],K_xz_la);
            d[5]=SIMD_Operations::add(d[5],K_yz_la);}

        f[0] = SIMD_Operations::fmadd(K_xx_la,Vu_x,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xy_la,Vu_y,f[0]);
        f[0] = SIMD_Operations::fmadd(K_xz_la,Vu_z,f[0]);

        f[1] = SIMD_Operations::fmadd(K_yx_la,Vu_x,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yy_la,Vu_y,f[1]);
        f[1] = SIMD_Operations::fmadd(K_yz_la,Vu_z,f[1]);

        f[2] = SIMD_Operations::fmadd(K_zx_la,Vu_x,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zy_la,Vu_y,f[2]);
        f[2] = SIMD_Operations::fmadd(K_zz_la,Vu_z,f[2]);
    }
    
    template<int color,typename T_FLAG>
    static void Smooth(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                       T Struct_Type::* u_fields[d],T_FLAG Struct_Type::* flag_field,T Struct_Type::* const f_fields[d],
                       T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                       const T dx,const bool boundary_smooth)
    {
        static_assert(sizeof(T)==sizeof(T_FLAG),"Please make sure that sizeof(T)==sizeof(T_FLAG)");
        Colored_Allocator_Type colored_allocator(allocator);
        constexpr int parity_x = (color >> 2) & 0x1;
        constexpr int parity_y = (color >> 1) & 0x1;
        constexpr int parity_z =  color       & 0x1;
        const Const_Colored_Array_Type u_x[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[0]),
                colored_allocator.template Get_Const_Array<1>(u_fields[0]),
                colored_allocator.template Get_Const_Array<2>(u_fields[0]),
                colored_allocator.template Get_Const_Array<3>(u_fields[0]),
                colored_allocator.template Get_Const_Array<4>(u_fields[0]),
                colored_allocator.template Get_Const_Array<5>(u_fields[0]),
                colored_allocator.template Get_Const_Array<6>(u_fields[0]),
                colored_allocator.template Get_Const_Array<7>(u_fields[0])
            };        

        const Const_Colored_Array_Type u_y[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[1]),
                colored_allocator.template Get_Const_Array<1>(u_fields[1]),
                colored_allocator.template Get_Const_Array<2>(u_fields[1]),
                colored_allocator.template Get_Const_Array<3>(u_fields[1]),
                colored_allocator.template Get_Const_Array<4>(u_fields[1]),
                colored_allocator.template Get_Const_Array<5>(u_fields[1]),
                colored_allocator.template Get_Const_Array<6>(u_fields[1]),
                colored_allocator.template Get_Const_Array<7>(u_fields[1])
            };

        const Const_Colored_Array_Type u_z[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(u_fields[2]),
                colored_allocator.template Get_Const_Array<1>(u_fields[2]),
                colored_allocator.template Get_Const_Array<2>(u_fields[2]),
                colored_allocator.template Get_Const_Array<3>(u_fields[2]),
                colored_allocator.template Get_Const_Array<4>(u_fields[2]),
                colored_allocator.template Get_Const_Array<5>(u_fields[2]),
                colored_allocator.template Get_Const_Array<6>(u_fields[2]),
                colored_allocator.template Get_Const_Array<7>(u_fields[2])
            };

        const Const_Colored_Array_Type mu[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(mu_field),
                colored_allocator.template Get_Const_Array<1>(mu_field),
                colored_allocator.template Get_Const_Array<2>(mu_field),
                colored_allocator.template Get_Const_Array<3>(mu_field),
                colored_allocator.template Get_Const_Array<4>(mu_field),
                colored_allocator.template Get_Const_Array<5>(mu_field),
                colored_allocator.template Get_Const_Array<6>(mu_field),
                colored_allocator.template Get_Const_Array<7>(mu_field)
            };

        const Const_Colored_Array_Type la[2][2][2] =
            {
                colored_allocator.template Get_Const_Array<0>(lambda_field),
                colored_allocator.template Get_Const_Array<1>(lambda_field),
                colored_allocator.template Get_Const_Array<2>(lambda_field),
                colored_allocator.template Get_Const_Array<3>(lambda_field),
                colored_allocator.template Get_Const_Array<4>(lambda_field),
                colored_allocator.template Get_Const_Array<5>(lambda_field),
                colored_allocator.template Get_Const_Array<6>(lambda_field),
                colored_allocator.template Get_Const_Array<7>(lambda_field)
            };
        
        Const_Colored_Array_Type f_x = colored_allocator.template Get_Const_Array<color>(f_fields[0]);
        Const_Colored_Array_Type f_y = colored_allocator.template Get_Const_Array<color>(f_fields[1]);
        Const_Colored_Array_Type f_z = colored_allocator.template Get_Const_Array<color>(f_fields[2]);

        Colored_Array_Type u_x_center = colored_allocator.template Get_Array<color>(u_fields[0]);
        Colored_Array_Type u_y_center = colored_allocator.template Get_Array<color>(u_fields[1]);
        Colored_Array_Type u_z_center = colored_allocator.template Get_Array<color>(u_fields[2]);
        
        SIMD_Type Vdx = SIMD_Operations::set(dx);
        SIMD_Type Vones = SIMD_Operations::set(T(1.0));
        auto flag = colored_allocator.template Get_Const_Array<color>(flag_field);

        SIMD_Int_Type VDirichlet_Flag_X = SIMD_Operations::set(T_FLAG(DIRICHLET_NODE_X));
        SIMD_Int_Type VDirichlet_Flag_Y = SIMD_Operations::set(T_FLAG(DIRICHLET_NODE_Y));
        SIMD_Int_Type VDirichlet_Flag_Z = SIMD_Operations::set(T_FLAG(DIRICHLET_NODE_Z));        
        SIMD_Int_Type VBoundary_Flag = SIMD_Operations::set(T_FLAG(BOUNDARY_NODE));
        SIMD_Int_Type VActive_Flag = SIMD_Operations::set(T_FLAG(ACTIVE_NODE));

        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e += SIMD_WIDTH,offset += SIMD_WIDTH * sizeof(T)){
                SIMD_Type Vf[d];
                SIMD_Type Vr[d];
                SIMD_Type Vu[d];
                SIMD_Type Vu_orig[d];
                SIMD_Type Vd[6];
                
                Vu[0] = SIMD_Operations::load(&u_x_center(offset));
                Vu[1] = SIMD_Operations::load(&u_y_center(offset));
                Vu[2] = SIMD_Operations::load(&u_z_center(offset));

                Vu_orig[0] = Vu[0];
                Vu_orig[1] = Vu[1];
                Vu_orig[2] = Vu[2];
                    
                Vr[0] = SIMD_Operations::load(&f_x(offset));
                Vr[1] = SIMD_Operations::load(&f_y(offset));
                Vr[2] = SIMD_Operations::load(&f_z(offset));

                Vd[0] = SIMD_Operations::zero();
                Vd[1] = SIMD_Operations::zero();
                Vd[2] = SIMD_Operations::zero();
                Vd[3] = SIMD_Operations::zero();
                Vd[4] = SIMD_Operations::zero();
                Vd[5] = SIMD_Operations::zero();
                
                SIMD_Int_Type Vflag = SIMD_Operations::load((SIMD_Int_Type*)&flag(offset));
                
                Vf[0] = SIMD_Operations::zero();
                Vf[1] = SIMD_Operations::zero();
                Vf[2] = SIMD_Operations::zero();

                {
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,0,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,0,1,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,0,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,0,1,1,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);


                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,0,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,0,1,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,0,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,0,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,0,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,0>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                    Apply_Spoke<parity_x,parity_y,parity_z,1,1,1,1,1,1>(offset,Vf,Vd,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                }
                
                Vf[0] = SIMD_Operations::mul(Vf[0],Vdx);
                Vf[1] = SIMD_Operations::mul(Vf[1],Vdx);
                Vf[2] = SIMD_Operations::mul(Vf[2],Vdx);
                
                Vr[0] = SIMD_Operations::sub(Vr[0],Vf[0]);
                Vr[1] = SIMD_Operations::sub(Vr[1],Vf[1]);
                Vr[2] = SIMD_Operations::sub(Vr[2],Vf[2]);

                Vd[0] = SIMD_Operations::mul(Vd[0],Vdx);
                Vd[1] = SIMD_Operations::mul(Vd[1],Vdx);
                Vd[2] = SIMD_Operations::mul(Vd[2],Vdx);
                Vd[3] = SIMD_Operations::mul(Vd[3],Vdx);
                Vd[4] = SIMD_Operations::mul(Vd[4],Vdx);
                Vd[5] = SIMD_Operations::mul(Vd[5],Vdx);

                SIMD_Int_Type dirichlet_x = SIMD_Operations::andv(Vflag,VDirichlet_Flag_X);
                SIMD_Int_Type dirichlet_y = SIMD_Operations::andv(Vflag,VDirichlet_Flag_Y);
                SIMD_Int_Type dirichlet_z = SIMD_Operations::andv(Vflag,VDirichlet_Flag_Z);
                
                {
                    // Clear Dirichlet
                    Vd[0]=SIMD_Operations::blendv(Vd[0],Vones,dirichlet_x);
                    Vd[3]=SIMD_Operations::blendv(Vd[3],SIMD_Operations::zero(),dirichlet_x);
                    Vd[4]=SIMD_Operations::blendv(Vd[4],SIMD_Operations::zero(),dirichlet_x);
                   
                    Vd[1]=SIMD_Operations::blendv(Vd[1],Vones,dirichlet_y);
                    Vd[3]=SIMD_Operations::blendv(Vd[3],SIMD_Operations::zero(),dirichlet_y);
                    Vd[5]=SIMD_Operations::blendv(Vd[5],SIMD_Operations::zero(),dirichlet_y);

                    Vd[2]=SIMD_Operations::blendv(Vd[2],Vones,dirichlet_z);
                    Vd[4]=SIMD_Operations::blendv(Vd[4],SIMD_Operations::zero(),dirichlet_z);
                    Vd[5]=SIMD_Operations::blendv(Vd[5],SIMD_Operations::zero(),dirichlet_z);

                    //Now invert Vd
                    const SIMD_Type Minor_11 = SIMD_Operations::sub(SIMD_Operations::mul(Vd[1],Vd[2]),
                                                                    SIMD_Operations::mul(Vd[5],Vd[5]));
                    const SIMD_Type Minor_12 = SIMD_Operations::sub(SIMD_Operations::mul(Vd[5],Vd[4]),
                                                                    SIMD_Operations::mul(Vd[2],Vd[3]));
                    const SIMD_Type Minor_13 = SIMD_Operations::sub(SIMD_Operations::mul(Vd[3],Vd[5]),
                                                                    SIMD_Operations::mul(Vd[4],Vd[1]));
                    const SIMD_Type Minor_22 = SIMD_Operations::sub(SIMD_Operations::mul(Vd[2],Vd[0]),
                                                                    SIMD_Operations::mul(Vd[4],Vd[4]));
                    const SIMD_Type Minor_23 = SIMD_Operations::sub(SIMD_Operations::mul(Vd[4],Vd[3]),
                                                                    SIMD_Operations::mul(Vd[0],Vd[5]));
                    const SIMD_Type Minor_33 = SIMD_Operations::sub(SIMD_Operations::mul(Vd[0],Vd[1]),
                                                                    SIMD_Operations::mul(Vd[3],Vd[3]));
                    const SIMD_Type determinant = SIMD_Operations::add(SIMD_Operations::add(SIMD_Operations::mul(Vd[0],Minor_11),
                                                                                            SIMD_Operations::mul(Vd[3],Minor_12)),
                                                                       SIMD_Operations::mul(Vd[4],Minor_13));
                    //TODO: Expensive, remove it.
                    const SIMD_Type one_over_determinant = SIMD_Operations::div(Vones,determinant); 

                    Vd[0] = SIMD_Operations::mul(Minor_11,one_over_determinant);
                    Vd[1] = SIMD_Operations::mul(Minor_22,one_over_determinant);
                    Vd[2] = SIMD_Operations::mul(Minor_33,one_over_determinant);
                    Vd[3] = SIMD_Operations::mul(Minor_12,one_over_determinant);
                    Vd[4] = SIMD_Operations::mul(Minor_13,one_over_determinant);
                    Vd[5] = SIMD_Operations::mul(Minor_23,one_over_determinant);

                    // Clear Dirichlet
                    Vd[0]=SIMD_Operations::blendv(Vd[0],SIMD_Operations::zero(),dirichlet_x);
                    Vd[3]=SIMD_Operations::blendv(Vd[3],SIMD_Operations::zero(),dirichlet_x);
                    Vd[4]=SIMD_Operations::blendv(Vd[4],SIMD_Operations::zero(),dirichlet_x);
                    
                    Vd[1]=SIMD_Operations::blendv(Vd[1],SIMD_Operations::zero(),dirichlet_y);
                    Vd[3]=SIMD_Operations::blendv(Vd[3],SIMD_Operations::zero(),dirichlet_y);
                    Vd[5]=SIMD_Operations::blendv(Vd[5],SIMD_Operations::zero(),dirichlet_y);

                    Vd[2]=SIMD_Operations::blendv(Vd[2],SIMD_Operations::zero(),dirichlet_z);
                    Vd[4]=SIMD_Operations::blendv(Vd[4],SIMD_Operations::zero(),dirichlet_z);
                    Vd[5]=SIMD_Operations::blendv(Vd[5],SIMD_Operations::zero(),dirichlet_z);
                }

                Vu[0] = SIMD_Operations::fmadd(Vr[0],Vd[0],Vu[0]);
                Vu[0] = SIMD_Operations::fmadd(Vr[1],Vd[3],Vu[0]);
                Vu[0] = SIMD_Operations::fmadd(Vr[2],Vd[4],Vu[0]);

                Vu[1] = SIMD_Operations::fmadd(Vr[0],Vd[3],Vu[1]);
                Vu[1] = SIMD_Operations::fmadd(Vr[1],Vd[1],Vu[1]);
                Vu[1] = SIMD_Operations::fmadd(Vr[2],Vd[5],Vu[1]);
                
                Vu[2] = SIMD_Operations::fmadd(Vr[0],Vd[4],Vu[2]);
                Vu[2] = SIMD_Operations::fmadd(Vr[1],Vd[5],Vu[2]);
                Vu[2] = SIMD_Operations::fmadd(Vr[2],Vd[2],Vu[2]);

                SIMD_Int_Type active = SIMD_Operations::andv(Vflag,VActive_Flag);
                Vu[0]=SIMD_Operations::blendv(Vu_orig[0],Vu[0],active);
                Vu[1]=SIMD_Operations::blendv(Vu_orig[1],Vu[1],active);
                Vu[2]=SIMD_Operations::blendv(Vu_orig[2],Vu[2],active);
              
                if(boundary_smooth){
                    SIMD_Int_Type boundary = SIMD_Operations::andv(Vflag,VBoundary_Flag);
                    Vu[0]=SIMD_Operations::blendv(Vu_orig[0],Vu[0],boundary);
                    Vu[1]=SIMD_Operations::blendv(Vu_orig[1],Vu[1],boundary);
                    Vu[2]=SIMD_Operations::blendv(Vu_orig[2],Vu[2],boundary);}

                Vu[0]=SIMD_Operations::blendv(Vu[0],SIMD_Operations::zero(),dirichlet_x);
                Vu[1]=SIMD_Operations::blendv(Vu[1],SIMD_Operations::zero(),dirichlet_y);
                Vu[2]=SIMD_Operations::blendv(Vu[2],SIMD_Operations::zero(),dirichlet_z);
                
                SIMD_Operations::store(&u_x_center(offset),Vu[0]);
                SIMD_Operations::store(&u_y_center(offset),Vu[1]);
                SIMD_Operations::store(&u_z_center(offset),Vu[2]);
            }
        }
    }    
};
}
#endif
