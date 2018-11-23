//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Class Elasticity_Matrix
//#####################################################################
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include "Elasticity_Matrix.h"
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include <iostream>
//#####################################################################
// Function Build_CSR_Matrix
//#####################################################################
template<class T> void Elasticity_Matrix<T>::
Build_CSR_Matrix()
{
    using SPGrid_Matrix_mask = typename SPG_Matrix_Allocator::Array_mask<>;

    constexpr uint64_t Spoke_Offsets_Uncentered[3][3][3] = {
        SPGrid_Matrix_mask::template LinearOffset<-1,-1,-1>::value, SPGrid_Matrix_mask::template LinearOffset<-1,-1, 0>::value, SPGrid_Matrix_mask::template LinearOffset<-1,-1,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset<-1, 0,-1>::value, SPGrid_Matrix_mask::template LinearOffset<-1, 0, 0>::value, SPGrid_Matrix_mask::template LinearOffset<-1, 0,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset<-1,+1,-1>::value, SPGrid_Matrix_mask::template LinearOffset<-1,+1, 0>::value, SPGrid_Matrix_mask::template LinearOffset<-1,+1,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset< 0,-1,-1>::value, SPGrid_Matrix_mask::template LinearOffset< 0,-1, 0>::value, SPGrid_Matrix_mask::template LinearOffset< 0,-1,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset< 0, 0,-1>::value, SPGrid_Matrix_mask::template LinearOffset< 0, 0, 0>::value, SPGrid_Matrix_mask::template LinearOffset< 0, 0,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset< 0,+1,-1>::value, SPGrid_Matrix_mask::template LinearOffset< 0,+1, 0>::value, SPGrid_Matrix_mask::template LinearOffset< 0,+1,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset<+1,-1,-1>::value, SPGrid_Matrix_mask::template LinearOffset<+1,-1, 0>::value, SPGrid_Matrix_mask::template LinearOffset<+1,-1,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset<+1, 0,-1>::value, SPGrid_Matrix_mask::template LinearOffset<+1, 0, 0>::value, SPGrid_Matrix_mask::template LinearOffset<+1, 0,+1>::value,
        SPGrid_Matrix_mask::template LinearOffset<+1,+1,-1>::value, SPGrid_Matrix_mask::template LinearOffset<+1,+1, 0>::value, SPGrid_Matrix_mask::template LinearOffset<+1,+1,+1>::value
    };
    using Spoke_Offsets_Stencil = const uint64_t (&)[3][3][3];
    auto spoke_offsets=reinterpret_cast<Spoke_Offsets_Stencil>(Spoke_Offsets_Uncentered[1][1][1]);

    auto blocks=page_map->Get_Blocks();
    auto matrix_data_array=allocator->Get_Array();

    // First pass -- count number of variables and matrix coefficients per block

    std::vector<long long> nvars(blocks.second,0ul);
    std::vector<long long> ncoeffs(blocks.second,0ul);

    #pragma omp parallel for
    for(int b=0;b<blocks.second;b++){
        auto offset=blocks.first[b];
        for(int e=0;e<SPGrid_Matrix_mask::elements_per_block;e++,offset+=SPGrid_Matrix_mask::field_size){
            const auto flags=matrix_data_array(offset).flags;
            if(flags & Elasticity_Matrix_Flags::Any_Spoke_Active){
                nvars[b] += 3;
                for(int di=-1;di<=1;di++)
                for(int dj=-1;dj<=1;dj++)
                for(int dk=-1;dk<=1;dk++)
                    if(flags & Elasticity_Matrix_Flags::Spoke_Active()[di][dj][dk]){
                        auto neighbor_offset=SPGrid_Matrix_mask::Packed_Add(offset,spoke_offsets[di][dj][dk]);
                        if(neighbor_offset>offset)
                            ncoeffs[b] += 9;
                        else if(neighbor_offset==offset)
                            ncoeffs[b] += 6;
                    }
            }
        }
    }

    // Second pass -- count aggregate variables and matrix coefficients (this is serial)

    std::vector<long long> nvars_aggregate(blocks.second+1,0ul);
    std::vector<long long> ncoeffs_aggregate(blocks.second+1,0ul);

    for(int b=0;b<blocks.second;b++){
        nvars_aggregate[b+1]=nvars_aggregate[b]+nvars[b];
        ncoeffs_aggregate[b+1]=ncoeffs_aggregate[b]+ncoeffs[b];}

    std::cout<<"Total variables    = "<<nvars_aggregate[blocks.second]<<std::endl;
    std::cout<<"Total coefficients = "<<ncoeffs_aggregate[blocks.second]<<std::endl;

    Clear_Pardiso_Data();
    PARDISO_data.n=nvars_aggregate[blocks.second];
    PARDISO_data.a=new T[ncoeffs_aggregate[blocks.second]];
    PARDISO_data.ia=new long long[PARDISO_data.n+1];
    PARDISO_data.ja=new long long[ncoeffs_aggregate[blocks.second]];
    PARDISO_data.x=new T[PARDISO_data.n];
    PARDISO_data.b=new T[PARDISO_data.n];
    // Third pass -- compute DOF indices

    #pragma omp parallel for
    for(int b=0;b<blocks.second;b++){
        auto offset=blocks.first[b];
        long long next_index=nvars_aggregate[b];
        for(int e=0;e<SPGrid_Matrix_mask::elements_per_block;e++,offset+=SPGrid_Matrix_mask::field_size){

            const auto flags=matrix_data_array(offset).flags;
            if(flags & Elasticity_Matrix_Flags::Any_Spoke_Active){
                matrix_data_array(offset).index=next_index;
                next_index += 3;
            }
        }
    }

    // Fourth pass -- compute coefficients

    #pragma omp parallel for
    for(int b=0;b<blocks.second;b++){

        auto offset=blocks.first[b];
        long long next_offset=ncoeffs_aggregate[b];
        for(int e=0;e<SPGrid_Matrix_mask::elements_per_block;e++,offset+=SPGrid_Matrix_mask::field_size){

            const auto flags=matrix_data_array(offset).flags;
            const auto data=Data(offset);

            if(flags & Elasticity_Matrix_Flags::Any_Spoke_Active){
                const long long index=matrix_data_array(offset).index;

                using row_entry_type = std::pair<T,long long>;
                std::array<std::vector<row_entry_type>,3> nodal_equations;

                for(int di=-1;di<=1;di++)
                for(int dj=-1;dj<=1;dj++)
                for(int dk=-1;dk<=1;dk++)
                    if(flags & Elasticity_Matrix_Flags::Spoke_Active()[di][dj][dk]){
                        auto neighbor_offset=SPGrid_Matrix_mask::Packed_Add(offset,spoke_offsets[di][dj][dk]);
                        const long long neighbor_index=matrix_data_array(neighbor_offset).index;
                        const auto neighbor_flags=matrix_data_array(neighbor_offset).flags;

                        for(int v=0;v<3;v++)
                        for(int w=0;w<3;w++)
                            if(index+v<=neighbor_index+w){
                                T coefficient=data[di][dj][dk][v][w];
                                if((flags & Elasticity_Matrix_Flags::Coordinate_Dirichlet[v]) || (neighbor_flags & Elasticity_Matrix_Flags::Coordinate_Dirichlet[w]))
                                    if(index+v==neighbor_index+w)
                                        coefficient=1.f;
                                    else
                                        coefficient=0.f;
                                nodal_equations[v].push_back(std::make_pair(coefficient,neighbor_index+w));
                            }
                    }
                auto cmp = [](const row_entry_type& x,const row_entry_type& y) { return x.second<y.second; };

                for(int v=0;v<3;v++){
#ifdef ICC_18
                    std::vector<std::pair<long long,T>> tmp;
                    for(auto itr = nodal_equations[v].begin();itr != nodal_equations[v].end();++itr){
                        tmp.push_back(std::make_pair(itr->second,itr->first));
                    }
                    std::sort(tmp.begin(),tmp.end());
                    PARDISO_data.ia[index+v]=next_offset;
                    for(int k=0;k<tmp.size();k++){
                        PARDISO_data.a[next_offset]=tmp[k].second;
                        PARDISO_data.ja[next_offset++]=tmp[k].first;
                    }
#else
                    std::sort(nodal_equations[v].begin(),nodal_equations[v].end(),cmp);
                    PARDISO_data.ia[index+v]=next_offset;
                    for(int k=0;k<nodal_equations[v].size();k++){
                        PARDISO_data.a[next_offset]=nodal_equations[v][k].first;
                        PARDISO_data.ja[next_offset++]=nodal_equations[v][k].second;
                    }
#endif
                }
            }
        }
    }
    PARDISO_data.ia[nvars_aggregate[blocks.second]]=ncoeffs_aggregate[blocks.second];

}
//#####################################################################
// Function PARDISO_Factorize
//#####################################################################
template<class T> void Elasticity_Matrix<T>::
PARDISO_Factorize()
{
/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only       */
/* necessary for the FIRST call of the PARDISO solver.                  */
/* -------------------------------------------------------------------- */
    for ( int i = 0; i < 64; i++ )
    {
        iparm[i] = 0;
    }
    iparm[0] = 1;         /* No solver default */
    iparm[1] = 3;         /* Parallel nested dissection */
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 0;         /* Max numbers of iterative refinement steps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Not in use */
    iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    iparm[23] = 1;        /* Two-level factorization*/
    iparm[26] = 1;        /* Check matrix for errors */
    iparm[27] = std::is_same<T,float>::value;        /* use float */
                          /* float(1) or double(0) precision */
    iparm[34] = 1;        /* 0-based indexing */
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;             /* Which factorization to use. */
    msglvl = 0;           /* Print statistical information in file */
    error = 0;            /* Initialize error flag */
    mtype = 2;                              /* Real symmetric positive definite matrix */
    nrhs = 1;                               /* Number of right hand sides. */

    phase = 11;
    pardiso_64 (pt, &maxfct, &mnum, &mtype, &phase,
             &PARDISO_data.n, PARDISO_data.a, PARDISO_data.ia, PARDISO_data.ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during symbolic factorization: %d", error);
        throw std::runtime_error("symbolic factorization: " + std::to_string(error));
    }
    printf ("\nReordering completed ... ");
    std::cout<<"\nNumber of nonzeros in factors = "<<iparm[17];
    printf ("\nNumber of factorization MFLOPS = %d", iparm[18]);
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */
    phase = 22;
    pardiso_64 (pt, &maxfct, &mnum, &mtype, &phase,
             &PARDISO_data.n, PARDISO_data.a, PARDISO_data.ia, PARDISO_data.ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during numerical factorization: %d", error);
        throw std::runtime_error("numerical factorization: " + std::to_string(error));
    }
    printf ("\nFactorization completed ... ");
}
//#####################################################################
// Function PARDISO_Solve
//#####################################################################
template<class T> void Elasticity_Matrix<T>::
PARDISO_Solve()
{
    using SPGrid_Matrix_mask = typename SPG_Matrix_Allocator::Array_mask<>;
    auto blocks=page_map->Get_Blocks();
    auto matrix_data_array=allocator->Get_Array();
#pragma omp parallel for
    for(int b=0;b<blocks.second;b++){
        auto offset=blocks.first[b];
        for(int e=0;e<SPGrid_Matrix_mask::elements_per_block;e++,offset+=SPGrid_Matrix_mask::field_size){
            const auto flags=matrix_data_array(offset).flags;
            if(flags & Elasticity_Matrix_Flags::Any_Spoke_Active){
                for(int v = 0;v < d;++v)
                    PARDISO_data.b[matrix_data_array(offset).index + v] = matrix_data_array(offset).array[v];}}}

    phase = 33;
    pardiso_64 (pt, &maxfct, &mnum, &mtype, &phase,
                &PARDISO_data.n, PARDISO_data.a, PARDISO_data.ia, PARDISO_data.ja, &idum, &nrhs, iparm, &msglvl, PARDISO_data.b, PARDISO_data.x, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during solution: %d", error);
        throw std::runtime_error("solution: " + std::to_string(error));
    }
    #pragma omp parallel for
    for(int b=0;b<blocks.second;b++){
        auto offset=blocks.first[b];
        for(int e=0;e<SPGrid_Matrix_mask::elements_per_block;e++,offset+=SPGrid_Matrix_mask::field_size){
            const auto flags=matrix_data_array(offset).flags;
            if(flags & Elasticity_Matrix_Flags::Any_Spoke_Active){
                for(int v = 0;v < d;++v)
                    matrix_data_array(offset).array[v] = PARDISO_data.x[matrix_data_array(offset).index + v];}}}

}
//#####################################################################
// Function Clear_Pardiso_Data
//#####################################################################
template<class T> void Elasticity_Matrix<T>::
Clear_Pardiso_Data()
{
    if(pt[0]){
        phase = -1;           /* Release internal memory. */
        pardiso_64 (pt, &maxfct, &mnum, &mtype, &phase,
                    &PARDISO_data.n, PARDISO_data.a, PARDISO_data.ia, PARDISO_data.ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    }
    if(PARDISO_data.a) delete PARDISO_data.a;
    if(PARDISO_data.ia) delete PARDISO_data.ia;
    if(PARDISO_data.ja) delete PARDISO_data.ja;
    if(PARDISO_data.x) delete PARDISO_data.x;
    if(PARDISO_data.b) delete PARDISO_data.b;
}
//#####################################################################
template class Elasticity_Matrix<float>;
template class Elasticity_Matrix<double>;
