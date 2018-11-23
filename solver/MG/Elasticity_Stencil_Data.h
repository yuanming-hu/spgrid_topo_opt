//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Class Elasticity_Stencil_Data
//#####################################################################
#ifndef __Elasticity_Stencil_Data_h__
#define __Elasticity_Stencil_Data_h__

template<class T,class T_FLAG>
struct Elasticity_Stencil_Struct{
    T data[243];
    T_FLAG flags;
};

#endif
