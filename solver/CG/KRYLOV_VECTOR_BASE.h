//#####################################################################
// Copyright 2009, Craig Schroeder.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class KRYLOV_VECTOR_BASE
//#####################################################################
#ifndef __KRYLOV_VECTOR_BASE__
#define __KRYLOV_VECTOR_BASE__

#include <iostream>

template<class T>
class KRYLOV_VECTOR_BASE
{
public:
    KRYLOV_VECTOR_BASE();
    virtual ~KRYLOV_VECTOR_BASE();

    const KRYLOV_VECTOR_BASE& operator= (const KRYLOV_VECTOR_BASE& bv)
    {Copy((T)1,bv);return *this;}

    virtual KRYLOV_VECTOR_BASE& operator+=(const KRYLOV_VECTOR_BASE& bv)=0;
    virtual KRYLOV_VECTOR_BASE& operator-=(const KRYLOV_VECTOR_BASE& bv)=0;
    virtual KRYLOV_VECTOR_BASE& operator*=(const T a)=0;
    virtual void Copy(const T c,const KRYLOV_VECTOR_BASE& bv)=0;
    virtual void Copy(const T c1,const KRYLOV_VECTOR_BASE& bv1,const KRYLOV_VECTOR_BASE& bv2)=0;
    virtual int Raw_Size() const=0;
    virtual T& Raw_Get(int i)=0;
    const T& Raw_Get(int i) const;
//#####################################################################
};

#endif
