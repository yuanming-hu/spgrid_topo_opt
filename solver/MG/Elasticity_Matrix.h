//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Class Elasticity_Matrix
//#####################################################################
#ifndef __Elasticity_Matrix_h__
#define __Elasticity_Matrix_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Page_Map.h>

using namespace SPGrid;

namespace Elasticity_Matrix_Flags
{

template<int di,int dj,int dk>
struct Spoke{
    static constexpr uint32_t Active = 1u << (9*di+3*dj+dk+13);
};

using Spoke_Active_Mask_Stencil = const uint32_t (&)[3][3][3];

constexpr uint32_t Spoke_Active_Uncentered[3][3][3] = {
    Spoke<-1,-1,-1>::Active, Spoke<-1,-1, 0>::Active, Spoke<-1,-1,+1>::Active,
    Spoke<-1, 0,-1>::Active, Spoke<-1, 0, 0>::Active, Spoke<-1, 0,+1>::Active,
    Spoke<-1,+1,-1>::Active, Spoke<-1,+1, 0>::Active, Spoke<-1,+1,+1>::Active,
    Spoke< 0,-1,-1>::Active, Spoke< 0,-1, 0>::Active, Spoke< 0,-1,+1>::Active,
    Spoke< 0, 0,-1>::Active, Spoke< 0, 0, 0>::Active, Spoke< 0, 0,+1>::Active,
    Spoke< 0,+1,-1>::Active, Spoke< 0,+1, 0>::Active, Spoke< 0,+1,+1>::Active,
    Spoke<+1,-1,-1>::Active, Spoke<+1,-1, 0>::Active, Spoke<+1,-1,+1>::Active,
    Spoke<+1, 0,-1>::Active, Spoke<+1, 0, 0>::Active, Spoke<+1, 0,+1>::Active,
    Spoke<+1,+1,-1>::Active, Spoke<+1,+1, 0>::Active, Spoke<+1,+1,+1>::Active
};

constexpr Spoke_Active_Mask_Stencil Spoke_Active() {return reinterpret_cast<Spoke_Active_Mask_Stencil>(Spoke_Active_Uncentered[1][1][1]);}

constexpr uint32_t Any_Spoke_Active = 0x07ffffff;

template<int v>
struct Coordinate{
    static constexpr uint32_t Dirichlet = 0x10000000u << v;
};

constexpr uint32_t Coordinate_Dirichlet[3] = {
    Coordinate<0>::Dirichlet,
    Coordinate<1>::Dirichlet,
    Coordinate<2>::Dirichlet
};

};

template<class T>
struct Elasticity_Matrix_Data{
    T data[243];
    uint32_t flags;
    long long index;
    T array[3];
};

template<class T>
struct Elasticity_Matrix{

    static constexpr int d = 3;
    using SPG_Matrix_Allocator = SPGrid_Allocator<Elasticity_Matrix_Data<T>,d>;
    using SPG_Matrix_Page_Map  = SPGrid_Page_Map<>;
    using Stencil_Type = T (&) [3][3][3][3][3];
    
    SPG_Matrix_Allocator *allocator;
    SPG_Matrix_Page_Map *page_map;

    void *pt[64];
    long long mtype;
    long long nrhs; 
    long long iparm[64];
    long long maxfct, mnum, phase, error, msglvl;
    T ddum;                                           /* Scalar dummy.  */
    long long idum;                                   /* Integer dummy. */

    struct {
        long long n;
        T *a;
        long long *ia;
        long long *ja;
        T *x;
        T *b;
    } PARDISO_data;
    
    Elasticity_Matrix(std::array<ucoord_t,d>& size)
    {
        allocator = new SPG_Matrix_Allocator(size);
        page_map  = new SPG_Matrix_Page_Map(*allocator);
        PARDISO_data.n = 0;
        PARDISO_data.a = NULL;
        PARDISO_data.ia = NULL;
        PARDISO_data.ja = NULL;
        PARDISO_data.x = NULL;
        PARDISO_data.b = NULL;
        for(int i = 0;i < 64;i++)
            pt[i] = 0;
    }
    ~Elasticity_Matrix()
    {delete page_map; delete allocator;Clear_Pardiso_Data();}

    Stencil_Type Data(const uint64_t linear_offset)
    {
        auto data_uncentered=reinterpret_cast<Stencil_Type>(allocator->Get_Array()(linear_offset).data);
        return reinterpret_cast<Stencil_Type>(data_uncentered[1][1][1][0][0]);
    }
        
    uint32_t& Flags(const uint64_t linear_offset)
    {return allocator->Get_Array()(linear_offset).flags;}


    void Build_CSR_Matrix();

    void PARDISO_Factorize();

    void PARDISO_Solve();

    void Clear_Pardiso_Data();
};

#endif
