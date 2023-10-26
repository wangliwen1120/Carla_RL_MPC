/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) vehicle_model_cost_ext_cost_e_fun_jac_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[50] = {46, 1, 0, 46, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};

/* vehicle_model_cost_ext_cost_e_fun_jac:(i0[3],i1[],i2[],i3[46])->(o0,o1[3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][0] : 0;
  a1=arg[3]? arg[3][3] : 0;
  a1=(a0-a1);
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][1] : 0;
  a4=arg[3]? arg[3][4] : 0;
  a4=(a3-a4);
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a5=arg[0]? arg[0][2] : 0;
  a6=arg[3]? arg[3][5] : 0;
  a5=(a5-a6);
  a6=casadi_sq(a5);
  a2=(a2+a6);
  a6=10000.;
  a7=arg[3]? arg[3][10] : 0;
  a7=(a0-a7);
  a8=casadi_sq(a7);
  a9=arg[3]? arg[3][11] : 0;
  a9=(a3-a9);
  a10=casadi_sq(a9);
  a8=(a8+a10);
  a10=(a6/a8);
  a11=arg[3]? arg[3][14] : 0;
  a11=(a0-a11);
  a12=casadi_sq(a11);
  a13=arg[3]? arg[3][15] : 0;
  a13=(a3-a13);
  a14=casadi_sq(a13);
  a12=(a12+a14);
  a14=(a6/a12);
  a15=(a10+a14);
  a16=arg[3]? arg[3][18] : 0;
  a16=(a0-a16);
  a17=casadi_sq(a16);
  a18=arg[3]? arg[3][19] : 0;
  a18=(a3-a18);
  a19=casadi_sq(a18);
  a17=(a17+a19);
  a19=(a6/a17);
  a15=(a15+a19);
  a20=arg[3]? arg[3][22] : 0;
  a20=(a0-a20);
  a21=casadi_sq(a20);
  a22=arg[3]? arg[3][23] : 0;
  a22=(a3-a22);
  a23=casadi_sq(a22);
  a21=(a21+a23);
  a23=(a6/a21);
  a15=(a15+a23);
  a24=arg[3]? arg[3][26] : 0;
  a24=(a0-a24);
  a25=casadi_sq(a24);
  a26=arg[3]? arg[3][27] : 0;
  a26=(a3-a26);
  a27=casadi_sq(a26);
  a25=(a25+a27);
  a27=(a6/a25);
  a15=(a15+a27);
  a28=arg[3]? arg[3][30] : 0;
  a28=(a0-a28);
  a29=casadi_sq(a28);
  a30=arg[3]? arg[3][31] : 0;
  a30=(a3-a30);
  a31=casadi_sq(a30);
  a29=(a29+a31);
  a31=(a6/a29);
  a15=(a15+a31);
  a32=arg[3]? arg[3][34] : 0;
  a32=(a0-a32);
  a33=casadi_sq(a32);
  a34=arg[3]? arg[3][35] : 0;
  a34=(a3-a34);
  a35=casadi_sq(a34);
  a33=(a33+a35);
  a35=(a6/a33);
  a15=(a15+a35);
  a36=arg[3]? arg[3][38] : 0;
  a0=(a0-a36);
  a36=casadi_sq(a0);
  a37=arg[3]? arg[3][39] : 0;
  a3=(a3-a37);
  a37=casadi_sq(a3);
  a36=(a36+a37);
  a6=(a6/a36);
  a15=(a15+a6);
  a2=(a2+a15);
  if (res[0]!=0) res[0][0]=a2;
  a1=(a1+a1);
  a0=(a0+a0);
  a6=(a6/a36);
  a0=(a0*a6);
  a32=(a32+a32);
  a35=(a35/a33);
  a32=(a32*a35);
  a0=(a0+a32);
  a28=(a28+a28);
  a31=(a31/a29);
  a28=(a28*a31);
  a0=(a0+a28);
  a24=(a24+a24);
  a27=(a27/a25);
  a24=(a24*a27);
  a0=(a0+a24);
  a20=(a20+a20);
  a23=(a23/a21);
  a20=(a20*a23);
  a0=(a0+a20);
  a16=(a16+a16);
  a19=(a19/a17);
  a16=(a16*a19);
  a0=(a0+a16);
  a11=(a11+a11);
  a14=(a14/a12);
  a11=(a11*a14);
  a0=(a0+a11);
  a7=(a7+a7);
  a10=(a10/a8);
  a7=(a7*a10);
  a0=(a0+a7);
  a1=(a1-a0);
  if (res[1]!=0) res[1][0]=a1;
  a4=(a4+a4);
  a3=(a3+a3);
  a3=(a3*a6);
  a34=(a34+a34);
  a34=(a34*a35);
  a3=(a3+a34);
  a30=(a30+a30);
  a30=(a30*a31);
  a3=(a3+a30);
  a26=(a26+a26);
  a26=(a26*a27);
  a3=(a3+a26);
  a22=(a22+a22);
  a22=(a22*a23);
  a3=(a3+a22);
  a18=(a18+a18);
  a18=(a18*a19);
  a3=(a3+a18);
  a13=(a13+a13);
  a13=(a13*a14);
  a3=(a3+a13);
  a9=(a9+a9);
  a9=(a9*a10);
  a3=(a3+a9);
  a4=(a4-a3);
  if (res[1]!=0) res[1][1]=a4;
  a5=(a5+a5);
  if (res[1]!=0) res[1][2]=a5;
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_e_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_e_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_e_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_e_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_e_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_e_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_e_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_e_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int vehicle_model_cost_ext_cost_e_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int vehicle_model_cost_ext_cost_e_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real vehicle_model_cost_ext_cost_e_fun_jac_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_model_cost_ext_cost_e_fun_jac_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_model_cost_ext_cost_e_fun_jac_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_model_cost_ext_cost_e_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_model_cost_ext_cost_e_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_e_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
