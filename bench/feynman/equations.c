/* equations.c - the 100 Feynman equations, as data.
 *
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Every entry is (reference, ground-truth formula, variables, sampling
 * intervals, evaluator). The formula string is printed in the report; the
 * engine never sees it.
 *
 * Sampling intervals are (1,5) unless the physics forces otherwise. Each
 * exception is spelled out in a comment on its line - a speed kept under c,
 * a resonant denominator kept away from zero, an arcsine argument kept inside
 * its domain. Nothing is tuned to flatter the engine: the intervals only ever
 * keep an equation well-posed.
 */
#include <math.h>
#include "equations.h"

#define PI 3.14159265358979323846

/* arcsin, guarded: the sampling intervals keep the argument in range, but a
 * benchmark should not be one rounding error away from a NaN. */
static double asin_safe(double x){
    if (x >  1.0) x =  1.0;
    if (x < -1.0) x = -1.0;
    return asin(x);
}

/* ---- I. Mechanics, waves, thermodynamics ------------------------------- */
static double e_I_6_2a  (const double *v){ return exp(-v[0]*v[0]/2.0)/sqrt(2*PI); }
static double e_I_6_2   (const double *v){ double t=v[1]/v[0]; return exp(-t*t/2.0)/(sqrt(2*PI)*v[0]); }
static double e_I_6_2b  (const double *v){ double t=(v[1]-v[2])/v[0]; return exp(-t*t/2.0)/(sqrt(2*PI)*v[0]); }
static double e_I_8_14  (const double *v){ return sqrt((v[1]-v[0])*(v[1]-v[0]) + (v[3]-v[2])*(v[3]-v[2])); }
static double e_I_9_18  (const double *v){
    double dx=v[4]-v[3], dy=v[6]-v[5], dz=v[8]-v[7];
    return v[0]*v[1]*v[2]/(dx*dx + dy*dy + dz*dz);
}
static double e_I_10_7  (const double *v){ return v[0]/sqrt(1.0 - v[1]*v[1]/(v[2]*v[2])); }
static double e_I_11_19 (const double *v){ return v[0]*v[1] + v[2]*v[3] + v[4]*v[5]; }
static double e_I_12_1  (const double *v){ return v[0]*v[1]; }
static double e_I_12_2  (const double *v){ return v[0]*v[1]/(4*PI*v[2]*v[3]*v[3]); }
static double e_I_12_4  (const double *v){ return v[0]/(4*PI*v[1]*v[2]*v[2]); }
static double e_I_12_5  (const double *v){ return v[0]*v[1]; }
static double e_I_12_11 (const double *v){ return v[0]*(v[1] + v[2]*v[3]*sin(v[4])); }
static double e_I_13_4  (const double *v){ return 0.5*v[0]*(v[1]*v[1] + v[2]*v[2] + v[3]*v[3]); }
static double e_I_13_12 (const double *v){ return v[0]*v[1]*v[2]*(1.0/v[4] - 1.0/v[3]); }
static double e_I_14_3  (const double *v){ return v[0]*v[1]*v[2]; }
static double e_I_14_4  (const double *v){ return 0.5*v[0]*v[1]*v[1]; }
static double e_I_15_3x (const double *v){ return (v[0]-v[1]*v[3])/sqrt(1.0 - v[1]*v[1]/(v[2]*v[2])); }
static double e_I_15_3t (const double *v){ return (v[3]-v[2]*v[0]/(v[1]*v[1]))/sqrt(1.0 - v[2]*v[2]/(v[1]*v[1])); }
static double e_I_15_10 (const double *v){ return v[0]*v[1]/sqrt(1.0 - v[1]*v[1]/(v[2]*v[2])); }
static double e_I_16_6  (const double *v){ return (v[1]+v[2])/(1.0 + v[1]*v[2]/(v[0]*v[0])); }
static double e_I_18_4  (const double *v){ return (v[0]*v[1] + v[2]*v[3])/(v[0]+v[2]); }
static double e_I_18_12 (const double *v){ return v[0]*v[1]*sin(v[2]); }
static double e_I_18_16 (const double *v){ return v[0]*v[1]*v[2]*sin(v[3]); }
static double e_I_24_6  (const double *v){ return 0.25*v[0]*(v[1]*v[1] + v[2]*v[2])*v[3]*v[3]; }
static double e_I_25_13 (const double *v){ return v[0]/v[1]; }
static double e_I_26_2  (const double *v){ return asin_safe(v[0]*sin(v[1])); }
static double e_I_27_6  (const double *v){ return 1.0/(1.0/v[0] + v[2]/v[1]); }
static double e_I_29_4  (const double *v){ return v[0]/v[1]; }
static double e_I_29_16 (const double *v){
    return sqrt(v[0]*v[0] + v[1]*v[1] - 2*v[0]*v[1]*cos(v[2]-v[3]));
}
static double e_I_30_3  (const double *v){
    double s = sin(v[1]/2.0);
    return v[0]*sin(v[2]*v[1]/2.0)*sin(v[2]*v[1]/2.0)/(s*s);
}
static double e_I_30_5  (const double *v){ return asin_safe(v[0]/(v[1]*v[2])); }
static double e_I_32_5  (const double *v){ return v[0]*v[0]*v[1]*v[1]/(6*PI*v[2]*v[3]*v[3]*v[3]); }
static double e_I_32_17 (const double *v){
    double d = v[4]*v[4] - v[5]*v[5];
    return (0.5*v[0]*v[1]*v[2]*v[2])*(8*PI*v[3]*v[3]/3.0)*(v[4]*v[4]*v[4]*v[4]/(d*d));
}
static double e_I_34_8  (const double *v){ return v[0]*v[1]*v[2]/v[3]; }
static double e_I_34_1  (const double *v){ return v[2]/(1.0 - v[1]/v[0]); }
static double e_I_34_14 (const double *v){
    return (1.0 + v[1]/v[0])/sqrt(1.0 - v[1]*v[1]/(v[0]*v[0]))*v[2];
}
static double e_I_34_27 (const double *v){ return v[0]*v[1]; }
static double e_I_37_4  (const double *v){ return v[0] + v[1] + 2*sqrt(v[0]*v[1])*cos(v[2]); }
static double e_I_38_12 (const double *v){ return 4*PI*v[0]*v[1]*v[1]/(v[2]*v[3]*v[3]); }
static double e_I_39_10 (const double *v){ return 1.5*v[0]*v[1]; }
static double e_I_39_11 (const double *v){ return v[1]*v[2]/(v[0]-1.0); }
static double e_I_39_22 (const double *v){ return v[0]*v[1]*v[2]/v[3]; }
static double e_I_40_1  (const double *v){ return v[0]*exp(-v[1]*v[2]*v[3]/(v[4]*v[5])); }
static double e_I_41_16 (const double *v){
    return v[0]*v[1]*v[1]*v[1]/(PI*PI*v[2]*v[2]*(exp(v[0]*v[1]/(v[3]*v[4])) - 1.0));
}
static double e_I_43_16 (const double *v){ return v[0]*v[1]*v[2]/v[3]; }
static double e_I_43_31 (const double *v){ return v[0]*v[1]*v[2]; }
static double e_I_43_43 (const double *v){ return v[1]*v[2]/((v[0]-1.0)*v[3]); }
static double e_I_44_4  (const double *v){ return v[0]*v[1]*v[2]*log(v[4]/v[3]); }
static double e_I_47_23 (const double *v){ return sqrt(v[0]*v[1]/v[2]); }
static double e_I_48_2  (const double *v){ return v[0]*v[2]*v[2]/sqrt(1.0 - v[1]*v[1]/(v[2]*v[2])); }
static double e_I_50_26 (const double *v){
    double c1 = cos(v[1]*v[2]);
    return v[0]*(c1 + v[3]*c1*c1);
}

/* ---- II. Electromagnetism, elasticity ---------------------------------- */
static double e_II_2_42  (const double *v){ return v[0]*(v[2]-v[1])*v[3]/v[4]; }
static double e_II_3_24  (const double *v){ return v[0]/(4*PI*v[1]*v[1]); }
static double e_II_4_23  (const double *v){ return v[0]/(4*PI*v[1]*v[2]); }
static double e_II_6_11  (const double *v){ return v[1]*cos(v[2])/(4*PI*v[0]*v[3]*v[3]); }
static double e_II_6_15a (const double *v){
    double r5 = v[2]*v[2]*v[2]*v[2]*v[2];
    return 3.0*v[1]*v[5]/(4*PI*v[0]*r5)*sqrt(v[3]*v[3] + v[4]*v[4]);
}
static double e_II_6_15b (const double *v){
    return 3.0*v[1]/(4*PI*v[0]*v[2]*v[2]*v[2])*cos(v[3])*sin(v[3]);
}
static double e_II_8_7   (const double *v){ return 0.6*v[0]*v[0]/(4*PI*v[1]*v[2]); }
static double e_II_8_31  (const double *v){ return 0.5*v[0]*v[1]*v[1]; }
static double e_II_10_9  (const double *v){ return v[0]/v[1]/(1.0 + v[2]); }
static double e_II_11_3  (const double *v){ return v[0]*v[1]/(v[2]*(v[3]*v[3] - v[4]*v[4])); }
static double e_II_11_17 (const double *v){
    return v[0]*(1.0 + v[1]*v[2]*cos(v[3])/(v[4]*v[5]));
}
static double e_II_11_20 (const double *v){ return v[0]*v[1]*v[1]*v[2]/(3.0*v[3]*v[4]); }
static double e_II_11_27 (const double *v){
    return v[0]*v[1]/(1.0 - v[0]*v[1]/3.0)*v[2]*v[3];
}
static double e_II_11_28 (const double *v){ return 1.0 + v[0]*v[1]/(1.0 - v[0]*v[1]/3.0); }
static double e_II_13_17 (const double *v){ return 2.0*v[2]/(4*PI*v[0]*v[1]*v[1]*v[3]); }
static double e_II_13_23 (const double *v){ return v[0]/sqrt(1.0 - v[1]*v[1]/(v[2]*v[2])); }
static double e_II_13_34 (const double *v){ return v[0]*v[1]/sqrt(1.0 - v[1]*v[1]/(v[2]*v[2])); }
static double e_II_15_4  (const double *v){ return -v[0]*v[1]*cos(v[2]); }
static double e_II_15_5  (const double *v){ return -v[0]*v[1]*cos(v[2]); }
static double e_II_21_32 (const double *v){ return v[0]/(4*PI*v[1]*v[2]*(1.0 - v[3]/v[4])); }
static double e_II_24_17 (const double *v){
    return sqrt(v[0]*v[0]/(v[1]*v[1]) - PI*PI/(v[2]*v[2]));
}
static double e_II_27_16 (const double *v){ return v[0]*v[1]*v[2]*v[2]; }
static double e_II_27_18 (const double *v){ return v[0]*v[1]*v[1]; }
static double e_II_34_2a (const double *v){ return v[0]*v[1]/(2*PI*v[2]); }
static double e_II_34_2  (const double *v){ return 0.5*v[0]*v[1]*v[2]; }
static double e_II_34_11 (const double *v){ return v[0]*v[1]*v[2]/(2.0*v[3]); }
static double e_II_34_29a(const double *v){ return v[0]*v[1]/(4*PI*v[2]); }
static double e_II_34_29b(const double *v){ return v[0]*v[1]*v[2]*v[3]/v[4]; }
static double e_II_35_18 (const double *v){
    double a = v[1]*v[2]/(v[3]*v[4]);
    return v[0]/(exp(a) + exp(-a));
}
static double e_II_35_21 (const double *v){ return v[0]*v[1]*tanh(v[1]*v[2]/(v[3]*v[4])); }
static double e_II_36_38 (const double *v){
    return v[0]*v[1]/(v[2]*v[3]) + v[0]*v[4]*v[5]/(v[6]*v[7]*v[7]*v[2]*v[3]);
}
static double e_II_37_1  (const double *v){ return v[0]*(1.0 + v[1])*v[2]; }
static double e_II_38_3  (const double *v){ return v[0]*v[1]*v[2]/v[3]; }
static double e_II_38_14 (const double *v){ return v[0]/(2.0*(1.0 + v[1])); }

/* ---- III. Quantum mechanics, solid state ------------------------------- */
static double e_III_4_32  (const double *v){ return 1.0/(exp(v[0]*v[1]/(v[2]*v[3])) - 1.0); }
static double e_III_4_33  (const double *v){
    return v[0]*v[1]/(exp(v[0]*v[1]/(v[2]*v[3])) - 1.0);
}
static double e_III_7_38  (const double *v){ return 2.0*v[0]*v[1]/v[2]; }
static double e_III_8_54  (const double *v){ double s = sin(v[0]*v[1]/v[2]); return s*s; }
static double e_III_9_52  (const double *v){
    double h = (v[4]-v[5])*v[3]/2.0, s = sin(h);
    return (v[0]*v[1]*v[3]/v[2])*(s*s)/(h*h);
}
static double e_III_10_19 (const double *v){
    return v[0]*sqrt(v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
}
static double e_III_12_43 (const double *v){ return v[0]*v[1]; }
static double e_III_13_18 (const double *v){ return 2.0*v[0]*v[1]*v[1]*v[2]/v[3]; }
static double e_III_14_14 (const double *v){ return v[0]*(exp(v[1]*v[2]/(v[3]*v[4])) - 1.0); }
static double e_III_15_12 (const double *v){ return 2.0*v[0]*(1.0 - cos(v[1]*v[2])); }
static double e_III_15_14 (const double *v){ return v[0]*v[0]/(2.0*v[1]*v[2]*v[2]); }
static double e_III_15_27 (const double *v){ return 2*PI*v[0]/(v[1]*v[2]); }
static double e_III_17_37 (const double *v){ return v[0]*(1.0 + v[1]*cos(v[2])); }
static double e_III_19_51 (const double *v){
    double q4 = v[1]*v[1]*v[1]*v[1], fp = 4*PI*v[2];
    return -v[0]*q4/(2.0*fp*fp*v[3]*v[3]*v[4]*v[4]);
}
static double e_III_21_20 (const double *v){ return -v[0]*v[1]*v[2]/v[3]; }

/* ======================================================================== */
const feyn_eq feyn_equations[] = {

/* -- I ------------------------------------------------------------------- */
{ "I.6.2a",  "exp(-theta^2/2)/sqrt(2*pi)", 1,
  {"theta"}, {1}, {3}, e_I_6_2a },
{ "I.6.2",   "exp(-(theta/sigma)^2/2)/(sqrt(2*pi)*sigma)", 2,
  {"sigma","theta"}, {1,1}, {3,3}, e_I_6_2 },
{ "I.6.2b",  "exp(-((theta-theta1)/sigma)^2/2)/(sqrt(2*pi)*sigma)", 3,
  {"sigma","theta","theta1"}, {1,1,1}, {3,3,3}, e_I_6_2b },
{ "I.8.14",  "sqrt((x2-x1)^2+(y2-y1)^2)", 4,
  {"x1","x2","y1","y2"}, {1,1,1,1}, {5,5,5,5}, e_I_8_14 },
{ "I.9.18",  "G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)", 9,   /* offsets kept apart so r>0 */
  {"G","m1","m2","x1","x2","y1","y2","z1","z2"},
  {1,1,1,3,1,3,1,3,1}, {2,2,2,4,2,4,2,4,2}, e_I_9_18 },
{ "I.10.7",  "m0/sqrt(1-v^2/c^2)", 3,                        /* v < c */
  {"m0","v","c"}, {1,1,3}, {5,2,10}, e_I_10_7 },
{ "I.11.19", "x1*y1+x2*y2+x3*y3", 6,
  {"x1","y1","x2","y2","x3","y3"}, {1,1,1,1,1,1}, {5,5,5,5,5,5}, e_I_11_19 },
{ "I.12.1",  "mu*Nn", 2, {"mu","Nn"}, {1,1}, {5,5}, e_I_12_1 },
{ "I.12.2",  "q1*q2/(4*pi*epsilon*r^2)", 4,
  {"q1","q2","epsilon","r"}, {1,1,1,1}, {5,5,5,5}, e_I_12_2 },
{ "I.12.4",  "q1/(4*pi*epsilon*r^2)", 3,
  {"q1","epsilon","r"}, {1,1,1}, {5,5,5}, e_I_12_4 },
{ "I.12.5",  "q2*Ef", 2, {"q2","Ef"}, {1,1}, {5,5}, e_I_12_5 },
{ "I.12.11", "q*(Ef+B*v*sin(theta))", 5,
  {"q","Ef","B","v","theta"}, {1,1,1,1,1}, {5,5,5,5,5}, e_I_12_11 },
{ "I.13.4",  "0.5*m*(v^2+u^2+w^2)", 4,
  {"m","v","u","w"}, {1,1,1,1}, {5,5,5,5}, e_I_13_4 },
{ "I.13.12", "G*m1*m2*(1/r2-1/r1)", 5,
  {"G","m1","m2","r1","r2"}, {1,1,1,1,1}, {5,5,5,5,5}, e_I_13_12 },
{ "I.14.3",  "m*g*z", 3, {"m","g","z"}, {1,1,1}, {5,5,5}, e_I_14_3 },
{ "I.14.4",  "0.5*k_spring*x^2", 2,
  {"k_spring","x"}, {1,1}, {5,5}, e_I_14_4 },
{ "I.15.3x", "(x-u*t)/sqrt(1-u^2/c^2)", 4,                   /* u < c */
  {"x","u","c","t"}, {5,1,3,1}, {10,2,20,2}, e_I_15_3x },
{ "I.15.3t", "(t-u*x/c^2)/sqrt(1-u^2/c^2)", 4,               /* u < c */
  {"x","c","u","t"}, {1,3,1,1}, {5,10,2,5}, e_I_15_3t },
{ "I.15.10", "m0*v/sqrt(1-v^2/c^2)", 3,                      /* v < c */
  {"m0","v","c"}, {1,1,3}, {5,2,10}, e_I_15_10 },
{ "I.16.6",  "(u+v)/(1+u*v/c^2)", 3,
  {"c","v","u"}, {1,1,1}, {5,5,5}, e_I_16_6 },
{ "I.18.4",  "(m1*r1+m2*r2)/(m1+m2)", 4,
  {"m1","r1","m2","r2"}, {1,1,1,1}, {5,5,5,5}, e_I_18_4 },
{ "I.18.12", "r*F*sin(theta)", 3,
  {"r","F","theta"}, {1,1,1}, {5,5,5}, e_I_18_12 },
{ "I.18.16", "m*r*v*sin(theta)", 4,
  {"m","r","v","theta"}, {1,1,1,1}, {5,5,5,5}, e_I_18_16 },
{ "I.24.6",  "0.25*m*(omega^2+omega0^2)*x^2", 4,
  {"m","omega","omega0","x"}, {1,1,1,1}, {3,3,3,3}, e_I_24_6 },
{ "I.25.13", "q/C", 2, {"q","C"}, {1,1}, {5,5}, e_I_25_13 },
{ "I.26.2",  "arcsin(n*sin(theta2))", 2,                     /* |n*sin| <= 1 */
  {"n","theta2"}, {0,1}, {1,5}, e_I_26_2 },
{ "I.27.6",  "1/(1/d1+n/d2)", 3,
  {"d1","d2","n"}, {1,1,1}, {5,5,5}, e_I_27_6 },
{ "I.29.4",  "omega/c", 2, {"omega","c"}, {1,1}, {5,5}, e_I_29_4 },
{ "I.29.16", "sqrt(x1^2+x2^2-2*x1*x2*cos(theta1-theta2))", 4,
  {"x1","x2","theta1","theta2"}, {1,1,1,1}, {5,5,5,5}, e_I_29_16 },
{ "I.30.3",  "Int0*sin(n*theta/2)^2/sin(theta/2)^2", 3,
  {"Int0","theta","n"}, {1,1,1}, {5,5,5}, e_I_30_3 },
{ "I.30.5",  "arcsin(lambda/(n*d))", 3,                      /* argument <= 1 */
  {"lambda","d","n"}, {1,2,1}, {2,5,5}, e_I_30_5 },
{ "I.32.5",  "q^2*a^2/(6*pi*epsilon*c^3)", 4,
  {"q","a","epsilon","c"}, {1,1,1,1}, {5,5,5,5}, e_I_32_5 },
{ "I.32.17", "(0.5*eps*c*Ef^2)*(8*pi*r^2/3)*omega^4/(omega^2-omega0^2)^2", 6,
  {"epsilon","c","Ef","r","omega","omega0"},                 /* omega != omega0 */
  {1,1,1,1,1,3}, {5,5,5,5,2,5}, e_I_32_17 },
{ "I.34.8",  "q*v*B/p", 4, {"q","v","B","p"}, {1,1,1,1}, {5,5,5,5}, e_I_34_8 },
{ "I.34.1",  "omega0/(1-v/c)", 3,                            /* v < c */
  {"c","v","omega0"}, {3,1,1}, {10,2,5}, e_I_34_1 },
{ "I.34.14", "(1+v/c)/sqrt(1-v^2/c^2)*omega0", 3,            /* v < c */
  {"c","v","omega0"}, {3,1,1}, {10,2,5}, e_I_34_14 },
{ "I.34.27", "hbar*omega", 2, {"hbar","omega"}, {1,1}, {5,5}, e_I_34_27 },
{ "I.37.4",  "I1+I2+2*sqrt(I1*I2)*cos(delta)", 3,
  {"I1","I2","delta"}, {1,1,1}, {5,5,5}, e_I_37_4 },
{ "I.38.12", "4*pi*epsilon*hbar^2/(m*q^2)", 4,
  {"epsilon","hbar","m","q"}, {1,1,1,1}, {5,5,5,5}, e_I_38_12 },
{ "I.39.10", "1.5*pr*V", 2, {"pr","V"}, {1,1}, {5,5}, e_I_39_10 },
{ "I.39.11", "pr*V/(gamma-1)", 3,                            /* gamma > 1 */
  {"gamma","pr","V"}, {2,1,1}, {5,5,5}, e_I_39_11 },
{ "I.39.22", "n*kb*T/V", 4, {"n","kb","T","V"}, {1,1,1,1}, {5,5,5,5}, e_I_39_22 },
{ "I.40.1",  "n0*exp(-m*g*x/(kb*T))", 6,
  {"n0","m","g","x","kb","T"}, {1,1,1,1,1,1}, {5,5,5,5,5,5}, e_I_40_1 },
{ "I.41.16", "hbar*omega^3/(pi^2*c^2*(exp(hbar*omega/(kb*T))-1))", 5,
  {"hbar","omega","c","kb","T"}, {1,1,1,1,1}, {5,5,5,5,5}, e_I_41_16 },
{ "I.43.16", "mu*q*Volt/d", 4,
  {"mu","q","Volt","d"}, {1,1,1,1}, {5,5,5,5}, e_I_43_16 },
{ "I.43.31", "mu*kb*T", 3, {"mu","kb","T"}, {1,1,1}, {5,5,5}, e_I_43_31 },
{ "I.43.43", "kb*v/((gamma-1)*A)", 4,                        /* gamma > 1 */
  {"gamma","kb","v","A"}, {2,1,1,1}, {5,5,5,5}, e_I_43_43 },
{ "I.44.4",  "n*kb*T*log(V2/V1)", 5,
  {"n","kb","T","V1","V2"}, {1,1,1,1,1}, {5,5,5,5,5}, e_I_44_4 },
{ "I.47.23", "sqrt(gamma*pr/rho)", 3,
  {"gamma","pr","rho"}, {1,1,1}, {5,5,5}, e_I_47_23 },
{ "I.48.2",  "m*c^2/sqrt(1-v^2/c^2)", 3,                     /* v < c */
  {"m","v","c"}, {1,1,3}, {5,2,10}, e_I_48_2 },
{ "I.50.26", "x1*(cos(omega*t)+alpha*cos(omega*t)^2)", 4,
  {"x1","omega","t","alpha"}, {1,1,1,1}, {3,3,3,3}, e_I_50_26 },

/* -- II ------------------------------------------------------------------ */
{ "II.2.42",  "kappa*(T2-T1)*A/d", 5,
  {"kappa","T1","T2","A","d"}, {1,1,1,1,1}, {5,5,5,5,5}, e_II_2_42 },
{ "II.3.24",  "Pwr/(4*pi*r^2)", 2, {"Pwr","r"}, {1,1}, {5,5}, e_II_3_24 },
{ "II.4.23",  "q/(4*pi*epsilon*r)", 3,
  {"q","epsilon","r"}, {1,1,1}, {5,5,5}, e_II_4_23 },
{ "II.6.11",  "p_d*cos(theta)/(4*pi*epsilon*r^2)", 4,
  {"epsilon","p_d","theta","r"}, {1,1,1,1}, {5,5,5,5}, e_II_6_11 },
{ "II.6.15a", "3*p_d*z*sqrt(x^2+y^2)/(4*pi*epsilon*r^5)", 6,
  {"epsilon","p_d","r","x","y","z"}, {1,1,1,1,1,1}, {5,5,5,5,5,5}, e_II_6_15a },
{ "II.6.15b", "3*p_d*cos(theta)*sin(theta)/(4*pi*epsilon*r^3)", 4,
  {"epsilon","p_d","r","theta"}, {1,1,1,1}, {5,5,5,5}, e_II_6_15b },
{ "II.8.7",   "0.6*q^2/(4*pi*epsilon*d)", 3,
  {"q","epsilon","d"}, {1,1,1}, {5,5,5}, e_II_8_7 },
{ "II.8.31",  "0.5*epsilon*Ef^2", 2,
  {"epsilon","Ef"}, {1,1}, {5,5}, e_II_8_31 },
{ "II.10.9",  "sigma_den/(epsilon*(1+chi))", 3,
  {"sigma_den","epsilon","chi"}, {1,1,1}, {5,5,5}, e_II_10_9 },
{ "II.11.3",  "q*Ef/(m*(omega0^2-omega^2))", 5,              /* omega0 != omega */
  {"q","Ef","m","omega0","omega"}, {1,1,1,3,1}, {5,5,5,5,2}, e_II_11_3 },
{ "II.11.17", "n0*(1+p_d*Ef*cos(theta)/(kb*T))", 6,
  {"n0","p_d","Ef","theta","kb","T"}, {1,1,1,1,1,1}, {3,3,3,3,3,3}, e_II_11_17 },
{ "II.11.20", "n_rho*p_d^2*Ef/(3*kb*T)", 5,
  {"n_rho","p_d","Ef","kb","T"}, {1,1,1,1,1}, {5,5,5,5,5}, e_II_11_20 },
{ "II.11.27", "n*alpha/(1-n*alpha/3)*epsilon*Ef", 4,         /* n*alpha < 3 */
  {"n","alpha","epsilon","Ef"}, {0,0,1,1}, {1,1,5,5}, e_II_11_27 },
{ "II.11.28", "1+n*alpha/(1-n*alpha/3)", 2,                  /* n*alpha < 3 */
  {"n","alpha"}, {0,0}, {1,1}, e_II_11_28 },
{ "II.13.17", "2*I/(4*pi*epsilon*c^2*r)", 4,
  {"epsilon","c","I","r"}, {1,1,1,1}, {5,5,5,5}, e_II_13_17 },
{ "II.13.23", "rho_c0/sqrt(1-v^2/c^2)", 3,                   /* v < c */
  {"rho_c0","v","c"}, {1,1,3}, {5,2,10}, e_II_13_23 },
{ "II.13.34", "rho_c0*v/sqrt(1-v^2/c^2)", 3,                 /* v < c */
  {"rho_c0","v","c"}, {1,1,3}, {5,2,10}, e_II_13_34 },
{ "II.15.4",  "-mom*B*cos(theta)", 3,
  {"mom","B","theta"}, {1,1,1}, {5,5,5}, e_II_15_4 },
{ "II.15.5",  "-p_d*Ef*cos(theta)", 3,
  {"p_d","Ef","theta"}, {1,1,1}, {5,5,5}, e_II_15_5 },
{ "II.21.32", "q/(4*pi*epsilon*r*(1-v/c))", 5,               /* v < c */
  {"q","epsilon","r","v","c"}, {1,1,1,1,3}, {5,5,5,2,10}, e_II_21_32 },
{ "II.24.17", "sqrt(omega^2/c^2-pi^2/d^2)", 3,               /* propagating mode */
  {"omega","c","d"}, {4,1,2}, {6,2,4}, e_II_24_17 },
{ "II.27.16", "epsilon*c*Ef^2", 3,
  {"epsilon","c","Ef"}, {1,1,1}, {5,5,5}, e_II_27_16 },
{ "II.27.18", "epsilon*Ef^2", 2, {"epsilon","Ef"}, {1,1}, {5,5}, e_II_27_18 },
{ "II.34.2a", "q*v/(2*pi*r)", 3, {"q","v","r"}, {1,1,1}, {5,5,5}, e_II_34_2a },
{ "II.34.2",  "0.5*q*v*r", 3, {"q","v","r"}, {1,1,1}, {5,5,5}, e_II_34_2 },
{ "II.34.11", "g_*q*B/(2*m)", 4,
  {"g_","q","B","m"}, {1,1,1,1}, {5,5,5,5}, e_II_34_11 },
{ "II.34.29a","q*h/(4*pi*m)", 3, {"q","h","m"}, {1,1,1}, {5,5,5}, e_II_34_29a },
{ "II.34.29b","g_*mom*B*Jz/hbar", 5,
  {"g_","mom","B","Jz","hbar"}, {1,1,1,1,1}, {5,5,5,5,5}, e_II_34_29b },
{ "II.35.18", "n0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))", 5,
  {"n0","mom","B","kb","T"}, {1,1,1,1,1}, {5,5,5,5,5}, e_II_35_18 },
{ "II.35.21", "n_rho*mom*tanh(mom*B/(kb*T))", 5,
  {"n_rho","mom","B","kb","T"}, {1,1,1,1,1}, {5,5,5,5,5}, e_II_35_21 },
{ "II.36.38", "mom*B/(kb*T)+mom*alpha*M/(epsilon*c^2*kb*T)", 8,
  {"mom","B","kb","T","alpha","M","epsilon","c"},
  {1,1,1,1,1,1,1,1}, {3,3,3,3,3,3,3,3}, e_II_36_38 },
{ "II.37.1",  "mom*(1+chi)*B", 3,
  {"mom","chi","B"}, {1,1,1}, {5,5,5}, e_II_37_1 },
{ "II.38.3",  "Y*A*x/d", 4, {"Y","A","x","d"}, {1,1,1,1}, {5,5,5,5}, e_II_38_3 },
{ "II.38.14", "Y/(2*(1+sigma))", 2,
  {"Y","sigma"}, {1,1}, {5,5}, e_II_38_14 },

/* -- III ----------------------------------------------------------------- */
{ "III.4.32",  "1/(exp(hbar*omega/(kb*T))-1)", 4,
  {"hbar","omega","kb","T"}, {1,1,1,1}, {5,5,5,5}, e_III_4_32 },
{ "III.4.33",  "hbar*omega/(exp(hbar*omega/(kb*T))-1)", 4,
  {"hbar","omega","kb","T"}, {1,1,1,1}, {5,5,5,5}, e_III_4_33 },
{ "III.7.38",  "2*mom*B/hbar", 3,
  {"mom","B","hbar"}, {1,1,1}, {5,5,5}, e_III_7_38 },
{ "III.8.54",  "sin(E_n*t/hbar)^2", 3,
  {"E_n","t","hbar"}, {1,1,1}, {5,5,5}, e_III_8_54 },
{ "III.9.52",  "p_d*Ef*t/hbar*sin((omega-omega0)*t/2)^2/((omega-omega0)*t/2)^2", 6,
  {"p_d","Ef","hbar","t","omega","omega0"},                  /* omega != omega0 */
  {1,1,1,1,1,3}, {3,3,3,5,2,5}, e_III_9_52 },
{ "III.10.19", "mom*sqrt(Bx^2+By^2+Bz^2)", 4,
  {"mom","Bx","By","Bz"}, {1,1,1,1}, {5,5,5,5}, e_III_10_19 },
{ "III.12.43", "n*hbar", 2, {"n","hbar"}, {1,1}, {5,5}, e_III_12_43 },
{ "III.13.18", "2*E_n*d^2*k/hbar", 4,
  {"E_n","d","k","hbar"}, {1,1,1,1}, {5,5,5,5}, e_III_13_18 },
{ "III.14.14", "I0*(exp(q*Volt/(kb*T))-1)", 5,
  {"I0","q","Volt","kb","T"}, {1,1,1,1,1}, {5,2,2,5,5}, e_III_14_14 },
{ "III.15.12", "2*U*(1-cos(k*d))", 3,
  {"U","k","d"}, {1,1,1}, {5,5,5}, e_III_15_12 },
{ "III.15.14", "hbar^2/(2*E_n*d^2)", 3,
  {"hbar","E_n","d"}, {1,1,1}, {5,5,5}, e_III_15_14 },
{ "III.15.27", "2*pi*alpha/(n*d)", 3,
  {"alpha","n","d"}, {1,1,1}, {5,5,5}, e_III_15_27 },
{ "III.17.37", "beta*(1+alpha*cos(theta))", 3,
  {"beta","alpha","theta"}, {1,1,1}, {5,5,5}, e_III_17_37 },
{ "III.19.51", "-m*q^4/(2*(4*pi*epsilon)^2*hbar^2*n^2)", 5,
  {"m","q","epsilon","hbar","n"}, {1,1,1,1,1}, {5,5,5,5,5}, e_III_19_51 },
{ "III.21.20", "-rho_c0*q*A_vec/m", 4,
  {"rho_c0","q","A_vec","m"}, {1,1,1,1}, {5,5,5,5}, e_III_21_20 }
};

const int feyn_count = (int)(sizeof feyn_equations / sizeof feyn_equations[0]);
