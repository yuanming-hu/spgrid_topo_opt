//#####################################################################
// Copyright 2008, Geoffrey Irving.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "KRYLOV_SYSTEM_BASE.h"
#include "KRYLOV_VECTOR_BASE.h"
template<class T> KRYLOV_SYSTEM_BASE<T>::
KRYLOV_SYSTEM_BASE(const bool use_preconditioner,const bool preconditioner_commutes_with_projection)
    :use_preconditioner(use_preconditioner),preconditioner_commutes_with_projection(preconditioner_commutes_with_projection)
{
}
template<class T> KRYLOV_SYSTEM_BASE<T>::
~KRYLOV_SYSTEM_BASE()
{
}
template<class T> void KRYLOV_SYSTEM_BASE<T>::
Apply_Preconditioner(const KRYLOV_VECTOR_BASE<T>& r,KRYLOV_VECTOR_BASE<T>& z) const
{
}
template<class T> void KRYLOV_SYSTEM_BASE<T>::
Project_Nullspace(KRYLOV_VECTOR_BASE<T>& x) const
{
}
template<class T> const KRYLOV_VECTOR_BASE<T>& KRYLOV_SYSTEM_BASE<T>::
Precondition(const KRYLOV_VECTOR_BASE<T>& r,KRYLOV_VECTOR_BASE<T>& z) const
{
    if(!use_preconditioner) return r;
    Apply_Preconditioner(r,z);
    if(!preconditioner_commutes_with_projection){Project(z);Project_Nullspace(z);}
    return z;
}
template class KRYLOV_SYSTEM_BASE<float>;
template class KRYLOV_SYSTEM_BASE<double>;
