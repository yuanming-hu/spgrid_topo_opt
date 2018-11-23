//#####################################################################
// Copyright 2006-2008, Geoffrey Irving, Craig Schroeder, Andrew Selle, Tamar Shinar, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CONJUGATE_GRADIENT
//#####################################################################
#ifndef __CONJUGATE_GRADIENT_4Ch__
#define __CONJUGATE_GRADIENT_4Ch__

#include "KRYLOV_SYSTEM_BASE.h"
// see Golub and Van Loan, 10.2.6, p. 529 for details

// cost per iteration = 1 matrix multiply/project/precondition, 2 inner products, 1 convergence norm, 3 saxpy's
// approximate total flops = 2v + 10n

template<class T>
class CONJUGATE_GRADIENT_4Ch
{

public:
    bool print_diagnostics,print_residuals;
    T nullspace_tolerance; // don't attempt to invert eigenvalues approximately less than nullspace_tolerance*max_eigenvalue
    int iterations_used;
    T residual_magnitude_squared,nullspace_measure; // extra convergence information
    int restart_iterations;

public:
    CONJUGATE_GRADIENT_4Ch(){}

    ~CONJUGATE_GRADIENT_4Ch();
//#####################################################################
    bool Solve(const KRYLOV_SYSTEM_BASE<T>& system,KRYLOV_VECTOR_BASE<T>& x,KRYLOV_VECTOR_BASE<T>& r,KRYLOV_VECTOR_BASE<T>& p,KRYLOV_VECTOR_BASE<T>& q,KRYLOV_VECTOR_BASE<T>& z,const T tolerance,const int min_iterations,const int max_iterations);
//#####################################################################
};

#endif
