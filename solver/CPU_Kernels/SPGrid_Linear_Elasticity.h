#ifndef __SPGRID_LINEAR_ELASTICITY_H__
#define __SPGRID_LINEAR_ELASTICITY_H__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>
#include <SPGrid/Tools/SPGrid_SIMD_Utilities.h>
#include "Stiffness_Matrices.h"

namespace SPGrid{
template<typename T,typename Struct_Type,int log2_page,int SIMD_WIDTH>
struct SPGrid_Linear_Elasticity
{
    static constexpr int d = 3;
    using Allocator_Type   = SPGrid_Allocator<Struct_Type,d,log2_page>;
    using Array_Type       = typename Allocator_Type::Array_type<T>;
    using Const_Array_Type = typename Allocator_Type::Array_type<const T>;
    using Mask_Type        = typename Allocator_Type::Array_mask<T>;
    using SIMD_Operations  = SIMD_Operations<T,SIMD_WIDTH>; 
    using SIMD_Utilities   = SPGrid_SIMD_Utilities<T,SIMD_WIDTH>;
    using SIMD_Type        = typename SIMD_type<T,SIMD_WIDTH>::type;
    static constexpr int elements_per_block = Mask_Type::elements_per_block;
    
    template<int Ix,int Iy,int Iz, //output
             int Jx,int Jy,int Jz> //input
    __forceinline static void Apply_Spoke(uint64_t base_offset,
                                          SIMD_Type f[3],
                                          Const_Array_Type u_x,Const_Array_Type u_y,Const_Array_Type u_z,
                                          Const_Array_Type mu,Const_Array_Type lambda,
                                          const T (&K_mu_array)[8][8][3][3],const T (&K_la_array)[8][8][3][3])
    {
        constexpr int cell_x = -Ix;
        constexpr int cell_y = -Iy;
        constexpr int cell_z = -Iz;

        constexpr int node_x = Jx-Ix;
        constexpr int node_y = Jy-Iy;
        constexpr int node_z = Jz-Iz;

        constexpr int index_in = Jx*4+Jy*2+Jz;
        constexpr int index_ou = Ix*4+Iy*2+Iz;
        
        SIMD_Type Vu_x,Vu_y,Vu_z;
        SIMD_Utilities::template Get_Vector<node_x,node_y,node_z>(base_offset,u_x,u_y,u_z,Vu_x,Vu_y,Vu_z);
      
        SIMD_Type Vmu,Vla;
        SIMD_Utilities::template Get_Vector<cell_x,cell_y,cell_z>(base_offset,mu,lambda,Vmu,Vla);

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

    static void Multiply(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                         T Struct_Type::* const u_fields[d],T Struct_Type::* f_fields[d],
                         T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                         T dx)
    {
        auto u_x = allocator.Get_Const_Array(u_fields[0]);
        auto u_y = allocator.Get_Const_Array(u_fields[1]);
        auto u_z = allocator.Get_Const_Array(u_fields[2]);

        auto f_x = allocator.Get_Array(f_fields[0]);
        auto f_y = allocator.Get_Array(f_fields[1]);
        auto f_z = allocator.Get_Array(f_fields[2]);

        auto mu = allocator.Get_Const_Array(mu_field);
        auto la = allocator.Get_Const_Array(lambda_field);

        SIMD_Type Vdx = SIMD_Operations::set(dx);
        
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e+=SIMD_WIDTH,offset+=SIMD_WIDTH*sizeof(T)){
                SIMD_Type f[d];

                f[0] = SIMD_Operations::zero();
                f[1] = SIMD_Operations::zero();
                f[2] = SIMD_Operations::zero();

                Apply_Spoke<0,0,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,0,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<0,0,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,0,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,1,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,1,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<0,1,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,1,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,0,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,0,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<1,0,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,0,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,1,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,1,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<1,1,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,1,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);                

                f[0] = SIMD_Operations::mul(f[0],Vdx);
                f[1] = SIMD_Operations::mul(f[1],Vdx);
                f[2] = SIMD_Operations::mul(f[2],Vdx);
                
                SIMD_Operations::store(&f_x(offset),f[0]);
                SIMD_Operations::store(&f_y(offset),f[1]);
                SIMD_Operations::store(&f_z(offset),f[2]);
            }
        }
    }
    static void Residual(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                         T Struct_Type::* const u_fields[d],T Struct_Type::* const f_fields[d], T Struct_Type::* const r_fields[d],
                         T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                         T dx)
    {
        auto u_x = allocator.Get_Const_Array(u_fields[0]);
        auto u_y = allocator.Get_Const_Array(u_fields[1]);
        auto u_z = allocator.Get_Const_Array(u_fields[2]);

        auto f_x = allocator.Get_Const_Array(f_fields[0]);
        auto f_y = allocator.Get_Const_Array(f_fields[1]);
        auto f_z = allocator.Get_Const_Array(f_fields[2]);

        auto r_x = allocator.Get_Array(r_fields[0]);
        auto r_y = allocator.Get_Array(r_fields[1]);
        auto r_z = allocator.Get_Array(r_fields[2]);

        auto mu = allocator.Get_Const_Array(mu_field);
        auto la = allocator.Get_Const_Array(lambda_field);

        SIMD_Type Vdx = SIMD_Operations::set(dx);
        
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e+=SIMD_WIDTH,offset+=SIMD_WIDTH*sizeof(T)){
                SIMD_Type f[d];
                SIMD_Type r[d];

                r[0] = SIMD_Operations::load(&f_x(offset));
                r[1] = SIMD_Operations::load(&f_y(offset));
                r[2] = SIMD_Operations::load(&f_z(offset));

                f[0] = SIMD_Operations::zero();
                f[1] = SIMD_Operations::zero();
                f[2] = SIMD_Operations::zero();
                
                Apply_Spoke<0,0,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,0,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<0,0,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,0,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,0,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,1,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,1,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<0,1,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<0,1,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<0,1,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,0,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,0,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<1,0,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,0,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,0,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,1,0,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,1,0,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,0,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                
                Apply_Spoke<1,1,1,0,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,0,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,0,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,0,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);

                Apply_Spoke<1,1,1,1,0,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,1,0,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,1,1,0>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);
                Apply_Spoke<1,1,1,1,1,1>(offset,f,u_x,u_y,u_z,mu,la,K_mu<T>,K_la<T>);                

                f[0] = SIMD_Operations::mul(f[0],Vdx);
                f[1] = SIMD_Operations::mul(f[1],Vdx);
                f[2] = SIMD_Operations::mul(f[2],Vdx);

                f[0] = SIMD_Operations::sub(r[0],f[0]);
                f[1] = SIMD_Operations::sub(r[1],f[1]);
                f[2] = SIMD_Operations::sub(r[2],f[2]);
                
                SIMD_Operations::store(&r_x(offset),f[0]);
                SIMD_Operations::store(&r_y(offset),f[1]);
                SIMD_Operations::store(&r_z(offset),f[2]);
            }
        }
    }
    template<int Ix,int Iy,int Iz, //output
             int Jx,int Jy,int Jz> //input
    __forceinline static void Get_Diagonal_Matrix(uint64_t base_offset,
                                                  SIMD_Type K[6],// We only care about the symetric part.
                                                  Const_Array_Type mu,Const_Array_Type lambda,
                                                  const T (&K_mu_array)[8][8][3][3],const T (&K_la_array)[8][8][3][3])
    {
        constexpr int cell_x = -Ix;
        constexpr int cell_y = -Iy;
        constexpr int cell_z = -Iz;

        constexpr int node_x = Jx-Ix;
        constexpr int node_y = Jy-Iy;
        constexpr int node_z = Jz-Iz;

        constexpr int index_in = Jx*4+Jy*2+Jz;
        constexpr int index_ou = Ix*4+Iy*2+Iz;
        
        SIMD_Type Vmu = SIMD_Utilities::template Get_Vector<cell_x,cell_y,cell_z>(base_offset,mu);
        SIMD_Type Vla = SIMD_Utilities::template Get_Vector<cell_x,cell_y,cell_z>(base_offset,lambda);

        SIMD_Type K_xx_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][0]);
        SIMD_Type K_xy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][1]);
        SIMD_Type K_xz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][0][2]);
        SIMD_Type K_yy_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][1]);
        SIMD_Type K_yz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][1][2]);
        SIMD_Type K_zz_mu = SIMD_Operations::mul(Vmu,K_mu_array[index_ou][index_in][2][2]);
        
        SIMD_Type K_xx_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][0]);
        SIMD_Type K_xy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][1]);
        SIMD_Type K_xz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][0][2]);
        SIMD_Type K_yy_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][1]);
        SIMD_Type K_yz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][1][2]);
        SIMD_Type K_zz_la = SIMD_Operations::mul(Vla,K_la_array[index_ou][index_in][2][2]);

        K[0] = SIMD_Operations::add(K_xx_mu,K[0]);
        K[0] = SIMD_Operations::add(K_xx_la,K[0]);

        K[1] = SIMD_Operations::add(K_yy_mu,K[1]);
        K[1] = SIMD_Operations::add(K_yy_la,K[1]);

        K[2] = SIMD_Operations::add(K_zz_mu,K[2]);
        K[2] = SIMD_Operations::add(K_zz_la,K[2]);

        K[3] = SIMD_Operations::add(K_xy_mu,K[3]);
        K[3] = SIMD_Operations::add(K_xy_la,K[3]);

        K[4] = SIMD_Operations::add(K_xz_mu,K[4]);
        K[4] = SIMD_Operations::add(K_xz_la,K[4]);

        K[5] = SIMD_Operations::add(K_yz_mu,K[5]);
        K[5] = SIMD_Operations::add(K_yz_la,K[5]);       
    }
    static void Get_Diagonal_Matrix(Allocator_Type& allocator, std::pair<const uint64_t*,unsigned> blocks,
                                    T Struct_Type::* const d_fields[6],T Struct_Type::* const mu_field,T Struct_Type::* const lambda_field,
                                    T dx)
    {
        auto d_xx = allocator.Get_Array(d_fields[0]);
        auto d_yy = allocator.Get_Array(d_fields[1]);
        auto d_zz = allocator.Get_Array(d_fields[2]);
        auto d_xy = allocator.Get_Array(d_fields[3]);
        auto d_xz = allocator.Get_Array(d_fields[4]);
        auto d_yz = allocator.Get_Array(d_fields[5]);

        auto mu = allocator.Get_Const_Array(mu_field);
        auto la = allocator.Get_Const_Array(lambda_field);

        SIMD_Type Vdx = SIMD_Operations::set(dx);
        
        #pragma omp parallel for
        for(int b = 0;b < blocks.second;++b){
            auto offset = blocks.first[b];
            for(int e = 0;e < elements_per_block;e+=SIMD_WIDTH,offset+=SIMD_WIDTH*sizeof(T)){
                SIMD_Type K[6];
                for(int i = 0;i<6;++i) K[i] = SIMD_Operations::zero();
                
                Get_Diagonal_Matrix<0,0,0,0,0,0>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<0,0,1,0,0,1>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<0,1,0,0,1,0>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<0,1,1,0,1,1>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<1,0,0,1,0,0>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<1,0,1,1,0,1>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<1,1,0,1,1,0>(offset,K,mu,la,K_mu<T>,K_la<T>);
                Get_Diagonal_Matrix<1,1,1,1,1,1>(offset,K,mu,la,K_mu<T>,K_la<T>);

                for(int i = 0;i<6;++i) K[i] = SIMD_Operations::mul(K[i],Vdx);
                SIMD_Operations::store(&d_xx(offset),K[0]);
                SIMD_Operations::store(&d_yy(offset),K[1]);
                SIMD_Operations::store(&d_zz(offset),K[2]);
                SIMD_Operations::store(&d_xy(offset),K[3]);
                SIMD_Operations::store(&d_xz(offset),K[4]);
                SIMD_Operations::store(&d_yz(offset),K[5]);
            }
        }
    }    
};
}
#endif
