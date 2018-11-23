//#####################################################################
// Copyright (c) 2018, Haixiang Liu, Eftychios Sifakis
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Trilinear_Cell_Transfer_Operators_SIMD_h__
#define __Trilinear_Cell_Transfer_Operators_SIMD_h__
#include <immintrin.h>
#include <SPGrid/Tools/SPGrid_SIMD_Utilities.h>

using namespace SPGrid;

template<typename T,int SIMD_WIDTH> struct Transfer_Helper;
struct Transfer_Helper<float,8>
{
    using T = float;
    static constexpr int SIMD_WIDTH = 8;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static void lX(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm256_permute2f128_ps(in[0],in[0],0x00);}
    __forceinline static void hX(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm256_permute2f128_ps(in[0],in[0],0x11);}
    __forceinline static void lY(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm256_permute_ps(in[0],0x44);}
    __forceinline static void hY(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm256_permute_ps(in[0],0xee);}
    __forceinline static void lZ(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm256_permute_ps(in[0],0xa0);}
    __forceinline static void hZ(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm256_permute_ps(in[0],0xf5);}
    __forceinline static void outerZ(const SIMD_Type inlZ[1],const SIMD_Type inhZ[1],SIMD_Type ou[1])
    {ou[0]=_mm256_blend_ps(inlZ[0],inhZ[0],0xaa);}
    __forceinline static void innerZ(const SIMD_Type inlZ[1],const SIMD_Type inhZ[1],SIMD_Type ou[1])
    {ou[0]=_mm256_blend_ps(inlZ[0],inhZ[0],0x55);ou[0]=_mm256_permute_ps(ou[0],0xb1);}
    __forceinline static void outerY(const SIMD_Type inlY[1],const SIMD_Type inhY[1],SIMD_Type ou[1])
    {ou[0]=_mm256_blend_ps(inlY[0],inhY[0],0xcc);}
    __forceinline static void innerY(const SIMD_Type inlY[1],const SIMD_Type inhY[1],SIMD_Type ou[1])
    {ou[0]=_mm256_blend_ps(inlY[0],inhY[0],0x33);ou[0]=_mm256_permute_ps(ou[0],0x4e);}
    __forceinline static void outerX(const SIMD_Type inlX[1],const SIMD_Type inhX[1],SIMD_Type ou[1])
    {ou[0]=_mm256_blend_ps(inlX[0],inhX[0],0xf0);}
    __forceinline static void innerX(const SIMD_Type inlX[1],const SIMD_Type inhX[1],SIMD_Type ou[1])
    {ou[0]=_mm256_blend_ps(inlX[0],inhX[0],0x0f);ou[0]=_mm256_permute2f128_ps(ou[0],ou[0],0x01);}
};

struct Transfer_Helper<double,4>
{
    using T = double;
    static constexpr int SIMD_WIDTH = 4;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static void lX(const SIMD_Type in[2],SIMD_Type ou[2])
    {ou[0]=in[0];ou[1]=in[0];}
    __forceinline static void hX(const SIMD_Type in[2],SIMD_Type ou[2])
    {ou[0]=in[1];ou[1]=in[1];}
    __forceinline static void lY(const SIMD_Type in[2],SIMD_Type ou[2])
    {ou[0]=_mm256_permute2f128_pd(in[0],in[0],0x00);ou[1]=_mm256_permute2f128_pd(in[1],in[1],0x00);}
    __forceinline static void hY(const SIMD_Type in[2],SIMD_Type ou[2])
    {ou[0]=_mm256_permute2f128_pd(in[0],in[0],0x11);ou[1]=_mm256_permute2f128_pd(in[1],in[1],0x11);}
    __forceinline static void lZ(const SIMD_Type in[2],SIMD_Type ou[2])        
    {ou[0]=_mm256_permute_pd(in[0],0x0);ou[1]=_mm256_permute_pd(in[1],0x0);}
    __forceinline static void hZ(const SIMD_Type in[2],SIMD_Type ou[2])       
    {ou[0]=_mm256_permute_pd(in[0],0xf);ou[1]=_mm256_permute_pd(in[1],0xf);}
    __forceinline static void outerZ(const SIMD_Type inlZ[2],const SIMD_Type inhZ[2],SIMD_Type ou[2])
    {ou[0]=_mm256_blend_pd(inlZ[0],inhZ[0],0xa);ou[1]=_mm256_blend_pd(inlZ[1],inhZ[1],0xa);}
    __forceinline static void innerZ(const SIMD_Type inlZ[2],const SIMD_Type inhZ[2],SIMD_Type ou[2])
    {ou[0]=_mm256_blend_pd(inlZ[0],inhZ[0],0x5);ou[1]=_mm256_blend_pd(inlZ[1],inhZ[1],0x5);
     ou[0]=_mm256_permute_pd(ou[0],0x5);ou[1]=_mm256_permute_pd(ou[1],0x5);}
    __forceinline static void outerY(const SIMD_Type inlY[2],const SIMD_Type inhY[2],SIMD_Type ou[2])
    {ou[0]=_mm256_blend_pd(inlY[0],inhY[0],0xc);ou[1]=_mm256_blend_pd(inlY[1],inhY[1],0xc);}
    __forceinline static void innerY(const SIMD_Type inlY[2],const SIMD_Type inhY[2],SIMD_Type ou[2])
    {ou[0]=_mm256_blend_pd(inlY[0],inhY[0],0x3);ou[1]=_mm256_blend_pd(inlY[1],inhY[1],0x3);
     ou[0]=_mm256_permute2f128_pd(ou[0],ou[0],0x1);ou[1]=_mm256_permute2f128_pd(ou[1],ou[1],0x1);}
    __forceinline static void outerX(const SIMD_Type inlX[2],const SIMD_Type inhX[2],SIMD_Type ou[2])
    {ou[0]=inlX[0];ou[1]=inhX[1];}
    __forceinline static void innerX(const SIMD_Type inlX[2],const SIMD_Type inhX[2],SIMD_Type ou[2])
    {ou[0]=inlX[1];ou[1]=inhX[0];}
};

struct Transfer_Helper<float,16>
{
    using T = float;
    static constexpr int SIMD_WIDTH = 16;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static void lX(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_shuffle_f32x4(in[0],in[0],0xaa);}
    __forceinline static void hX(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_shuffle_f32x4(in[0],in[0],0x55);}
    __forceinline static void lY(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permute_ps(in[0],0x44);}
    __forceinline static void hY(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permute_ps(in[0],0xee);}
    __forceinline static void lZ(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permute_ps(in[0],0xa0);}
    __forceinline static void hZ(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permute_ps(in[0],0xf5);}
    __forceinline static void outerZ(const SIMD_Type inlZ[1],const SIMD_Type inhZ[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_ps(_mm512_int2mask(0xaaaa),inlZ[0],inhZ[0]);}
    __forceinline static void innerZ(const SIMD_Type inlZ[1],const SIMD_Type inhZ[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_ps(_mm512_int2mask(0x5555),inlZ[0],inhZ[0]);ou[0]=_mm512_permute_ps(ou[0],0xb1);}
    __forceinline static void outerY(const SIMD_Type inlY[1],const SIMD_Type inhY[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_ps(_mm512_int2mask(0xcccc),inlY[0],inhY[0]);}
    __forceinline static void innerY(const SIMD_Type inlY[1],const SIMD_Type inhY[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_ps(_mm512_int2mask(0x3333),inlY[0],inhY[0]);ou[0]=_mm512_permute_ps(ou[0],0x4e);}
    __forceinline static void outerX(const SIMD_Type inlX[1],const SIMD_Type inhX[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_ps(_mm512_int2mask(0xf0f0),inlX[0],inhX[0]);}
    __forceinline static void innerX(const SIMD_Type inlX[1],const SIMD_Type inhX[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_ps(_mm512_int2mask(0x0f0f),inlX[0],inhX[0]);ou[0]=_mm512_shuffle_f32x4(ou[0],ou[0],0x11);}
};

struct Transfer_Helper<double,8>
{
    using T = double;
    static constexpr int SIMD_WIDTH = 8;
    static constexpr int Nodes_Per_Cell = 8;
    using SIMD_Type = typename SIMD_type<T,SIMD_WIDTH>::type;
    __forceinline static void lX(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_shuffle_f64x2(in[0],in[0],0x44);}
    __forceinline static void hX(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_shuffle_f64x2(in[0],in[0],0xee);}
    __forceinline static void lY(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permutex_pd(in[0],0x44);}
    __forceinline static void hY(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permutex_pd(in[0],0xee);}
    __forceinline static void lZ(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permute_pd(in[0],0x00);}
    __forceinline static void hZ(const SIMD_Type in[1],SIMD_Type ou[1])
    {ou[0]=_mm512_permute_pd(in[0],0xff);}
    __forceinline static void outerZ(const SIMD_Type inlZ[1],const SIMD_Type inhZ[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_pd(_mm512_int2mask(0xaa),inlZ[0],inhZ[0]);}
    __forceinline static void innerZ(const SIMD_Type inlZ[1],const SIMD_Type inhZ[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_pd(_mm512_int2mask(0x55),inlZ[0],inhZ[0]);ou[0]=_mm512_permute_pd(ou[0],0x55);}
    __forceinline static void outerY(const SIMD_Type inlY[1],const SIMD_Type inhY[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_pd(_mm512_int2mask(0xcc),inlY[0],inhY[0]);}
    __forceinline static void innerY(const SIMD_Type inlY[1],const SIMD_Type inhY[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_pd(_mm512_int2mask(0x33),inlY[0],inhY[0]);ou[0]=_mm512_permutex_pd(ou[0],0x4e);}
    __forceinline static void outerX(const SIMD_Type inlX[1],const SIMD_Type inhX[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_pd(_mm512_int2mask(0xf0),inlX[0],inhX[0]);}
    __forceinline static void innerX(const SIMD_Type inlX[1],const SIMD_Type inhX[1],SIMD_Type ou[1])
    {ou[0]=_mm512_mask_blend_pd(_mm512_int2mask(0x0f),inlX[0],inhX[0]);ou[0]=_mm512_shuffle_f64x2(ou[0],ou[0],0x4e);}
};

template<typename T,int SIMD_WIDTH>
struct Trilinear_Cell_Transfer_Operators_SIMD
{
    static constexpr int d              = 3;
    static constexpr int Nodes_Per_Cell = 1 << d;
    static constexpr int nSIMDs         = (8/SIMD_WIDTH >= 1) ? 8/SIMD_WIDTH : 1;
    using SIMD_Operations               = SIMD_Operations<T,SIMD_WIDTH>; 
    using SIMD_Utilities                = SPGrid_SIMD_Utilities<T,SIMD_WIDTH>;
    using SIMD_Type                     = typename SIMD_type<T,SIMD_WIDTH>::type;
    static void Prolongate(const SIMD_Type vCoarse[nSIMDs],SIMD_Type vFine[8][nSIMDs])
    {
        for(int l=0;l<nSIMDs;++l)
        for(int q=0;q<8;q++)
            vFine[q][l]=vCoarse[l];

        SIMD_Type vCoarseScaled[nSIMDs];
        for(int l=0;l<nSIMDs;++l)
            vCoarseScaled[l]=SIMD_Operations::mul(vCoarse[l],T(0.125));   
        
        // Split along the x-axis
        SIMD_Type vFine_lX[nSIMDs];
        SIMD_Type vFine_hX[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::lX(vCoarseScaled,vFine_lX);
        Transfer_Helper<T,SIMD_WIDTH>::hX(vCoarseScaled,vFine_hX);
        for(int l=0;l<nSIMDs;++l){
            vFine_lX[l]=SIMD_Operations::add(vFine_lX[l],vCoarseScaled[l]);
            vFine_hX[l]=SIMD_Operations::add(vFine_hX[l],vCoarseScaled[l]);}

        // Split along the y-axis
        SIMD_Type vFine_lXlY[nSIMDs];
        SIMD_Type vFine_hXlY[nSIMDs];
        SIMD_Type vFine_lXhY[nSIMDs];
        SIMD_Type vFine_hXhY[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::lY(vFine_lX,vFine_lXlY);
        Transfer_Helper<T,SIMD_WIDTH>::lY(vFine_hX,vFine_hXlY);
        Transfer_Helper<T,SIMD_WIDTH>::hY(vFine_lX,vFine_lXhY);
        Transfer_Helper<T,SIMD_WIDTH>::hY(vFine_hX,vFine_hXhY);
        for(int l=0;l<nSIMDs;++l){
            vFine_lXlY[l]=SIMD_Operations::add(vFine_lXlY[l],vFine_lX[l]);
            vFine_lXhY[l]=SIMD_Operations::add(vFine_lXhY[l],vFine_lX[l]);
            vFine_hXlY[l]=SIMD_Operations::add(vFine_hXlY[l],vFine_hX[l]);
            vFine_hXhY[l]=SIMD_Operations::add(vFine_hXhY[l],vFine_hX[l]);}

        // Split along the z-axis        
        Transfer_Helper<T,SIMD_WIDTH>::lZ(vFine_lXlY,vFine[0]);
        Transfer_Helper<T,SIMD_WIDTH>::hZ(vFine_lXlY,vFine[1]);
        Transfer_Helper<T,SIMD_WIDTH>::lZ(vFine_lXhY,vFine[2]);
        Transfer_Helper<T,SIMD_WIDTH>::hZ(vFine_lXhY,vFine[3]);
        Transfer_Helper<T,SIMD_WIDTH>::lZ(vFine_hXlY,vFine[4]);
        Transfer_Helper<T,SIMD_WIDTH>::hZ(vFine_hXlY,vFine[5]);
        Transfer_Helper<T,SIMD_WIDTH>::lZ(vFine_hXhY,vFine[6]);
        Transfer_Helper<T,SIMD_WIDTH>::hZ(vFine_hXhY,vFine[7]);
        for(int l=0;l<nSIMDs;++l){
            vFine[0][l]=SIMD_Operations::add(vFine[0][l],vFine_lXlY[l]);
            vFine[1][l]=SIMD_Operations::add(vFine[1][l],vFine_lXlY[l]);
            vFine[2][l]=SIMD_Operations::add(vFine[2][l],vFine_lXhY[l]);
            vFine[3][l]=SIMD_Operations::add(vFine[3][l],vFine_lXhY[l]);
            vFine[4][l]=SIMD_Operations::add(vFine[4][l],vFine_hXlY[l]);
            vFine[5][l]=SIMD_Operations::add(vFine[5][l],vFine_hXlY[l]);
            vFine[6][l]=SIMD_Operations::add(vFine[6][l],vFine_hXhY[l]);
            vFine[7][l]=SIMD_Operations::add(vFine[7][l],vFine_hXhY[l]);}
    }
    static void Restrict(SIMD_Type vCoarse[nSIMDs],const SIMD_Type vFine[8][nSIMDs])
    {
        // Z direction
        SIMD_Type vFine_lXlY_inner[nSIMDs];
        SIMD_Type vFine_lXlY_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerZ(vFine[0],vFine[1],vFine_lXlY_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerZ(vFine[0],vFine[1],vFine_lXlY_outer);
        SIMD_Type vFine_lXlY[nSIMDs];
        for(int l=0;l<nSIMDs;++l){
            vFine_lXlY[l]=SIMD_Operations::add(vFine[0][l],vFine[1][l]);
            vFine_lXlY[l]=SIMD_Operations::add(vFine_lXlY[l],vFine_lXlY_inner[l]);
            vFine_lXlY[l]=SIMD_Operations::add(vFine_lXlY[l],vFine_lXlY_outer[l]);}

        SIMD_Type vFine_lXhY_inner[nSIMDs];
        SIMD_Type vFine_lXhY_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerZ(vFine[2],vFine[3],vFine_lXhY_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerZ(vFine[2],vFine[3],vFine_lXhY_outer);
        SIMD_Type vFine_lXhY[nSIMDs];
        for(int l=0;l<nSIMDs;++l){
            vFine_lXhY[l]=SIMD_Operations::add(vFine[2][l],vFine[3][l]);
            vFine_lXhY[l]=SIMD_Operations::add(vFine_lXhY[l],vFine_lXhY_inner[l]);
            vFine_lXhY[l]=SIMD_Operations::add(vFine_lXhY[l],vFine_lXhY_outer[l]);}
        
        SIMD_Type vFine_hXlY_inner[nSIMDs];
        SIMD_Type vFine_hXlY_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerZ(vFine[4],vFine[5],vFine_hXlY_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerZ(vFine[4],vFine[5],vFine_hXlY_outer);
        SIMD_Type vFine_hXlY[nSIMDs];
        for(int l=0;l<nSIMDs;++l){
            vFine_hXlY[l]=SIMD_Operations::add(vFine[4][l],vFine[5][l]);
            vFine_hXlY[l]=SIMD_Operations::add(vFine_hXlY[l],vFine_hXlY_inner[l]);
            vFine_hXlY[l]=SIMD_Operations::add(vFine_hXlY[l],vFine_hXlY_outer[l]);}

        SIMD_Type vFine_hXhY_inner[nSIMDs];
        SIMD_Type vFine_hXhY_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerZ(vFine[6],vFine[7],vFine_hXhY_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerZ(vFine[6],vFine[7],vFine_hXhY_outer);
        SIMD_Type vFine_hXhY[nSIMDs];
        for(int l=0;l<nSIMDs;++l){
            vFine_hXhY[l]=SIMD_Operations::add(vFine[6][l],vFine[7][l]);
            vFine_hXhY[l]=SIMD_Operations::add(vFine_hXhY[l],vFine_hXhY_inner[l]);
            vFine_hXhY[l]=SIMD_Operations::add(vFine_hXhY[l],vFine_hXhY_outer[l]);}

        // Y direction
        SIMD_Type vFine_lX_inner[nSIMDs];
        SIMD_Type vFine_lX_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerY(vFine_lXlY,vFine_lXhY,vFine_lX_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerY(vFine_lXlY,vFine_lXhY,vFine_lX_outer);
        SIMD_Type vFine_lX[nSIMDs];
        for(int l=0;l<nSIMDs;++l){
            vFine_lX[l]=SIMD_Operations::add(vFine_lXlY[l],vFine_lXhY[l]);
            vFine_lX[l]=SIMD_Operations::add(vFine_lX[l],vFine_lX_inner[l]);
            vFine_lX[l]=SIMD_Operations::add(vFine_lX[l],vFine_lX_outer[l]);}

        SIMD_Type vFine_hX_inner[nSIMDs];
        SIMD_Type vFine_hX_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerY(vFine_hXlY,vFine_hXhY,vFine_hX_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerY(vFine_hXlY,vFine_hXhY,vFine_hX_outer);
        SIMD_Type vFine_hX[nSIMDs];
        for(int l=0;l<nSIMDs;++l){
            vFine_hX[l]=SIMD_Operations::add(vFine_hXlY[l],vFine_hXhY[l]);
            vFine_hX[l]=SIMD_Operations::add(vFine_hX[l],vFine_hX_inner[l]);
            vFine_hX[l]=SIMD_Operations::add(vFine_hX[l],vFine_hX_outer[l]);}

        // X direction
        SIMD_Type vFine_inner[nSIMDs];
        SIMD_Type vFine_outer[nSIMDs];
        Transfer_Helper<T,SIMD_WIDTH>::innerX(vFine_lX,vFine_hX,vFine_inner);
        Transfer_Helper<T,SIMD_WIDTH>::outerX(vFine_lX,vFine_hX,vFine_outer);
        for(int l=0;l<nSIMDs;++l){
            vCoarse[l]=SIMD_Operations::add(vFine_lX[l],vFine_hX[l]);
            vCoarse[l]=SIMD_Operations::add(vCoarse[l],vFine_inner[l]);
            vCoarse[l]=SIMD_Operations::add(vCoarse[l],vFine_outer[l]);}

        for(int l=0;l<nSIMDs;++l){vCoarse[l]=SIMD_Operations::mul(vCoarse[l],(T)0.125);}        
    }
};

#endif
