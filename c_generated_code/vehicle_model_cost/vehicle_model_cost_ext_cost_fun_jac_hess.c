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
  #define CASADI_PREFIX(ID) vehicle_model_cost_ext_cost_fun_jac_hess_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
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
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[50] = {46, 1, 0, 46, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s6[15] = {5, 5, 0, 1, 2, 4, 6, 7, 0, 1, 2, 3, 2, 3, 4};
static const casadi_int casadi_s7[8] = {0, 5, 0, 0, 0, 0, 0, 0};

/* vehicle_model_cost_ext_cost_fun_jac_hess:(i0[3],i1[2],i2[],i3[46])->(o0,o1[5],o2[5x5,7nz],o3[],o4[0x5]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a7, a8, a9;
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
  a6=5.9999999999999998e-01;
  a7=arg[1]? arg[1][0] : 0;
  a8=(a6*a7);
  a9=(a8*a7);
  a10=arg[1]? arg[1][1] : 0;
  a11=(a6*a10);
  a12=(a11*a10);
  a9=(a9+a12);
  a2=(a2+a9);
  a9=2.9999999999999999e-01;
  a12=arg[3]? arg[3][1] : 0;
  a13=(a7-a12);
  a13=(a9*a13);
  a12=(a7-a12);
  a14=(a13*a12);
  a15=arg[3]? arg[3][2] : 0;
  a16=(a10-a15);
  a16=(a9*a16);
  a15=(a10-a15);
  a17=(a16*a15);
  a14=(a14+a17);
  a2=(a2+a14);
  a14=10000.;
  a17=arg[3]? arg[3][10] : 0;
  a17=(a0-a17);
  a18=casadi_sq(a17);
  a19=arg[3]? arg[3][11] : 0;
  a19=(a3-a19);
  a20=casadi_sq(a19);
  a18=(a18+a20);
  a20=(a14/a18);
  a21=arg[3]? arg[3][14] : 0;
  a21=(a0-a21);
  a22=casadi_sq(a21);
  a23=arg[3]? arg[3][15] : 0;
  a23=(a3-a23);
  a24=casadi_sq(a23);
  a22=(a22+a24);
  a24=(a14/a22);
  a25=(a20+a24);
  a26=arg[3]? arg[3][18] : 0;
  a26=(a0-a26);
  a27=casadi_sq(a26);
  a28=arg[3]? arg[3][19] : 0;
  a28=(a3-a28);
  a29=casadi_sq(a28);
  a27=(a27+a29);
  a29=(a14/a27);
  a25=(a25+a29);
  a30=arg[3]? arg[3][22] : 0;
  a30=(a0-a30);
  a31=casadi_sq(a30);
  a32=arg[3]? arg[3][23] : 0;
  a32=(a3-a32);
  a33=casadi_sq(a32);
  a31=(a31+a33);
  a33=(a14/a31);
  a25=(a25+a33);
  a34=arg[3]? arg[3][26] : 0;
  a34=(a0-a34);
  a35=casadi_sq(a34);
  a36=arg[3]? arg[3][27] : 0;
  a36=(a3-a36);
  a37=casadi_sq(a36);
  a35=(a35+a37);
  a37=(a14/a35);
  a25=(a25+a37);
  a38=arg[3]? arg[3][30] : 0;
  a38=(a0-a38);
  a39=casadi_sq(a38);
  a40=arg[3]? arg[3][31] : 0;
  a40=(a3-a40);
  a41=casadi_sq(a40);
  a39=(a39+a41);
  a41=(a14/a39);
  a25=(a25+a41);
  a42=arg[3]? arg[3][34] : 0;
  a42=(a0-a42);
  a43=casadi_sq(a42);
  a44=arg[3]? arg[3][35] : 0;
  a44=(a3-a44);
  a45=casadi_sq(a44);
  a43=(a43+a45);
  a45=(a14/a43);
  a25=(a25+a45);
  a46=arg[3]? arg[3][38] : 0;
  a0=(a0-a46);
  a46=casadi_sq(a0);
  a47=arg[3]? arg[3][39] : 0;
  a3=(a3-a47);
  a47=casadi_sq(a3);
  a46=(a46+a47);
  a14=(a14/a46);
  a25=(a25+a14);
  a2=(a2+a25);
  if (res[0]!=0) res[0][0]=a2;
  a12=(a9*a12);
  a13=(a13+a12);
  a13=(a13+a8);
  a7=(a6*a7);
  a13=(a13+a7);
  if (res[1]!=0) res[1][0]=a13;
  a9=(a9*a15);
  a16=(a16+a9);
  a16=(a16+a11);
  a6=(a6*a10);
  a16=(a16+a6);
  if (res[1]!=0) res[1][1]=a16;
  a1=(a1+a1);
  a16=(a0+a0);
  a6=(a14/a46);
  a10=(a16*a6);
  a11=(a42+a42);
  a9=(a45/a43);
  a15=(a11*a9);
  a10=(a10+a15);
  a15=(a38+a38);
  a13=(a41/a39);
  a7=(a15*a13);
  a10=(a10+a7);
  a7=(a34+a34);
  a8=(a37/a35);
  a12=(a7*a8);
  a10=(a10+a12);
  a12=(a30+a30);
  a2=(a33/a31);
  a25=(a12*a2);
  a10=(a10+a25);
  a25=(a26+a26);
  a47=(a29/a27);
  a48=(a25*a47);
  a10=(a10+a48);
  a48=(a21+a21);
  a49=(a24/a22);
  a50=(a48*a49);
  a10=(a10+a50);
  a50=(a17+a17);
  a51=(a20/a18);
  a52=(a50*a51);
  a10=(a10+a52);
  a1=(a1-a10);
  if (res[1]!=0) res[1][2]=a1;
  a4=(a4+a4);
  a1=(a3+a3);
  a10=(a1*a6);
  a52=(a44+a44);
  a53=(a52*a9);
  a10=(a10+a53);
  a53=(a40+a40);
  a54=(a53*a13);
  a10=(a10+a54);
  a54=(a36+a36);
  a55=(a54*a8);
  a10=(a10+a55);
  a55=(a32+a32);
  a56=(a55*a2);
  a10=(a10+a56);
  a56=(a28+a28);
  a57=(a56*a47);
  a10=(a10+a57);
  a57=(a23+a23);
  a58=(a57*a49);
  a10=(a10+a58);
  a58=(a19+a19);
  a59=(a58*a51);
  a10=(a10+a59);
  a4=(a4-a10);
  if (res[1]!=0) res[1][3]=a4;
  a5=(a5+a5);
  if (res[1]!=0) res[1][4]=a5;
  a5=1.7999999999999998e+00;
  if (res[2]!=0) res[2][0]=a5;
  if (res[2]!=0) res[2][1]=a5;
  a5=2.;
  a4=(a5*a6);
  a14=(a14/a46);
  a0=(a0+a0);
  a10=(a14*a0);
  a10=(a10/a46);
  a59=(a6/a46);
  a0=(a59*a0);
  a10=(a10+a0);
  a10=(a16*a10);
  a4=(a4-a10);
  a10=(a5*a9);
  a45=(a45/a43);
  a42=(a42+a42);
  a0=(a45*a42);
  a0=(a0/a43);
  a60=(a9/a43);
  a42=(a60*a42);
  a0=(a0+a42);
  a0=(a11*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a10=(a5*a13);
  a41=(a41/a39);
  a38=(a38+a38);
  a0=(a41*a38);
  a0=(a0/a39);
  a42=(a13/a39);
  a38=(a42*a38);
  a0=(a0+a38);
  a0=(a15*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a10=(a5*a8);
  a37=(a37/a35);
  a34=(a34+a34);
  a0=(a37*a34);
  a0=(a0/a35);
  a38=(a8/a35);
  a34=(a38*a34);
  a0=(a0+a34);
  a0=(a7*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a10=(a5*a2);
  a33=(a33/a31);
  a30=(a30+a30);
  a0=(a33*a30);
  a0=(a0/a31);
  a34=(a2/a31);
  a30=(a34*a30);
  a0=(a0+a30);
  a0=(a12*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a10=(a5*a47);
  a29=(a29/a27);
  a26=(a26+a26);
  a0=(a29*a26);
  a0=(a0/a27);
  a30=(a47/a27);
  a26=(a30*a26);
  a0=(a0+a26);
  a0=(a25*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a10=(a5*a49);
  a24=(a24/a22);
  a21=(a21+a21);
  a0=(a24*a21);
  a0=(a0/a22);
  a26=(a49/a22);
  a21=(a26*a21);
  a0=(a0+a21);
  a0=(a48*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a10=(a5*a51);
  a20=(a20/a18);
  a17=(a17+a17);
  a0=(a20*a17);
  a0=(a0/a18);
  a21=(a51/a18);
  a17=(a21*a17);
  a0=(a0+a17);
  a0=(a50*a0);
  a10=(a10-a0);
  a4=(a4+a10);
  a4=(a5-a4);
  if (res[2]!=0) res[2][2]=a4;
  a3=(a3+a3);
  a14=(a14*a3);
  a14=(a14/a46);
  a59=(a59*a3);
  a14=(a14+a59);
  a16=(a16*a14);
  a44=(a44+a44);
  a45=(a45*a44);
  a45=(a45/a43);
  a60=(a60*a44);
  a45=(a45+a60);
  a11=(a11*a45);
  a16=(a16+a11);
  a40=(a40+a40);
  a41=(a41*a40);
  a41=(a41/a39);
  a42=(a42*a40);
  a41=(a41+a42);
  a15=(a15*a41);
  a16=(a16+a15);
  a36=(a36+a36);
  a37=(a37*a36);
  a37=(a37/a35);
  a38=(a38*a36);
  a37=(a37+a38);
  a7=(a7*a37);
  a16=(a16+a7);
  a32=(a32+a32);
  a33=(a33*a32);
  a33=(a33/a31);
  a34=(a34*a32);
  a33=(a33+a34);
  a12=(a12*a33);
  a16=(a16+a12);
  a28=(a28+a28);
  a29=(a29*a28);
  a29=(a29/a27);
  a30=(a30*a28);
  a29=(a29+a30);
  a25=(a25*a29);
  a16=(a16+a25);
  a23=(a23+a23);
  a24=(a24*a23);
  a24=(a24/a22);
  a26=(a26*a23);
  a24=(a24+a26);
  a48=(a48*a24);
  a16=(a16+a48);
  a19=(a19+a19);
  a20=(a20*a19);
  a20=(a20/a18);
  a21=(a21*a19);
  a20=(a20+a21);
  a50=(a50*a20);
  a16=(a16+a50);
  if (res[2]!=0) res[2][3]=a16;
  if (res[2]!=0) res[2][4]=a16;
  a6=(a5*a6);
  a1=(a1*a14);
  a6=(a6-a1);
  a9=(a5*a9);
  a52=(a52*a45);
  a9=(a9-a52);
  a6=(a6+a9);
  a13=(a5*a13);
  a53=(a53*a41);
  a13=(a13-a53);
  a6=(a6+a13);
  a8=(a5*a8);
  a54=(a54*a37);
  a8=(a8-a54);
  a6=(a6+a8);
  a2=(a5*a2);
  a55=(a55*a33);
  a2=(a2-a55);
  a6=(a6+a2);
  a47=(a5*a47);
  a56=(a56*a29);
  a47=(a47-a56);
  a6=(a6+a47);
  a49=(a5*a49);
  a57=(a57*a24);
  a49=(a49-a57);
  a6=(a6+a49);
  a51=(a5*a51);
  a58=(a58*a20);
  a51=(a51-a58);
  a6=(a6+a51);
  a6=(a5-a6);
  if (res[2]!=0) res[2][5]=a6;
  if (res[2]!=0) res[2][6]=a5;
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void vehicle_model_cost_ext_cost_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int vehicle_model_cost_ext_cost_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int vehicle_model_cost_ext_cost_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real vehicle_model_cost_ext_cost_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_model_cost_ext_cost_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_model_cost_ext_cost_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_model_cost_ext_cost_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_model_cost_ext_cost_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s2;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int vehicle_model_cost_ext_cost_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
