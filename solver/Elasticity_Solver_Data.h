//#####################################################################
// Copyright 2017, Haixiang Liu, Eftychios Sifakis.
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
// Class Elasticity_Solver_Data
//#####################################################################
#ifndef __Elasticity_Solver_Data_h__
#define __Elasticity_Solver_Data_h__

template<class T,class T_FLAGS>
struct Elasticity_Solver_Data{
    T ch0; // u_x
    T ch1; // u_y
    T ch2; // u_z
    T ch3; // f(r)_x
    T ch4; // f(r)_y
    T ch5; // f(r)_z
    T ch6; // s_x,mg_r_x //s_x is backed up before preconditioning
    T ch7; // s_y,mg_r_y //s_y is backed up before preconditioning
    T ch8; // s_z,mg_r_z //s_z is backed up before preconditioning
    T ch9; // q_x,z_x //reused
    T ch10;// q_y,z_y //reused
    T ch11;// q_z,z_z //reused
    T ch12;// mu
    T ch13;// lambda
    T ch14;// not used
    T_FLAGS flags;
};

#endif
