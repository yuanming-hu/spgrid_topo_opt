//#####################################################################
// Copyright 2002-2008, Ronald Fedkiw, Geoffrey Irving, Igor Neverov, Craig Schroeder, Tamar Shinar, Eftychios Sifakis, Huamin Wang, Rachel Weinstein.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CONJUGATE_GRADIENT
//#####################################################################
#include "CONJUGATE_GRADIENT.h"
#include "KRYLOV_VECTOR_BASE.h"
#include <cfloat>
#include <limits>
#include <stdexcept> 
//#####################################################################
// Destructor
//#####################################################################
template<class T> CONJUGATE_GRADIENT_4Ch<T>::
~CONJUGATE_GRADIENT_4Ch()
{}
//#####################################################################
// Function Solve
//#####################################################################
template<class T> bool CONJUGATE_GRADIENT_4Ch<T>::
Solve(const KRYLOV_SYSTEM_BASE<T>& system,KRYLOV_VECTOR_BASE<T>& x, KRYLOV_VECTOR_BASE<T>& r,KRYLOV_VECTOR_BASE<T>& s,KRYLOV_VECTOR_BASE<T>& q,
      KRYLOV_VECTOR_BASE<T>& z,const T tolerance,const int min_iterations,const int max_iterations)
{
    // NOTE: you should never try to make copies of VECTOR_T's inside here as they could be indirect.
    static const T small_number=std::numeric_limits<T>::epsilon();
    system.Set_Boundary_Conditions(x);
    T rho_old=(T)FLT_MAX;T convergence_norm=0;
    system.Multiply(x,q);r-=q;system.Project(r);
    int iterations;for(iterations=0;;iterations++){
        // stopping conditions
        system.Project_Nullspace(r);
        convergence_norm=system.Convergence_Norm(r);
        if(print_residuals) {
            std::cout<<convergence_norm<<std::endl;;}
        if(convergence_norm<=tolerance && (iterations>=min_iterations || convergence_norm<small_number)){
            if(print_diagnostics) std::cout<<"cg iterations: "<<iterations<<std::endl;
            if(iterations_used) iterations_used=iterations;return true;}
        if(iterations==max_iterations) break;
        // actual iteration
        const KRYLOV_VECTOR_BASE<T>& mr=system.Precondition(r,z);
        T rho=(T)system.Inner_Product(mr,r);
        //std::cout<<"rho: "<<rho<<std::endl;
        s.Copy(rho/rho_old,s,mr);
        system.Multiply(s,q);
        system.Project(q);
        T s_dot_q=(T)system.Inner_Product(s,q);
        //std::cout<<"s_dot_q: "<<s_dot_q<<std::endl;
        if(s_dot_q<=0) {
            std::cout<<"CG: matrix appears indefinite or singular, s_dot_q/s_dot_s="<<s_dot_q/(T)system.Inner_Product(s,s)<<std::endl;
            throw std::runtime_error("CG detected singular");
            return false;}
        T alpha=s_dot_q?rho/s_dot_q:(T)FLT_MAX;
        //std::cout<<"alpha: "<<alpha<<std::endl;
        x.Copy(alpha,s,x);
        r.Copy(-alpha,q,r);
        rho_old=rho;}
    if(print_diagnostics) std::cout<<"cg iterations: "<<iterations<<std::endl;if(iterations_used) iterations_used=iterations;
    if(print_diagnostics) {std::cout<<"cg not converged after "<<max_iterations<<" iterations, error = "<<convergence_norm<<std::endl;}
    return false;
}
//#####################################################################
template class CONJUGATE_GRADIENT_4Ch<float>;
template class CONJUGATE_GRADIENT_4Ch<double>;
