//#####################################################################
// Copyright 2009, Craig Schroeder.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "KRYLOV_VECTOR_BASE.h"
//#####################################################################
// Constructor
//#####################################################################
template<class T> KRYLOV_VECTOR_BASE<T>::
KRYLOV_VECTOR_BASE()
{
}
//#####################################################################
// Destructor
//#####################################################################
template<class T> KRYLOV_VECTOR_BASE<T>::
~KRYLOV_VECTOR_BASE()
{
}
//#####################################################################
// Function Raw_Get
//#####################################################################
template<class T> const T& KRYLOV_VECTOR_BASE<T>::
Raw_Get(int i) const
{
}
template class KRYLOV_VECTOR_BASE<float>;
template class KRYLOV_VECTOR_BASE<double>;
