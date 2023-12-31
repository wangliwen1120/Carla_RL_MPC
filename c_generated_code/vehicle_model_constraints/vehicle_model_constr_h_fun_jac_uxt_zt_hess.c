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
  #define CASADI_PREFIX(ID) vehicle_model_constr_h_fun_jac_uxt_zt_hess_ ## ID
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
static const casadi_int casadi_s2[36] = {32, 1, 0, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
static const casadi_int casadi_s3[3] = {0, 0, 0};
static const casadi_int casadi_s4[50] = {46, 1, 0, 46, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45};
static const casadi_int casadi_s5[131] = {5, 32, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4};
static const casadi_int casadi_s6[15] = {5, 5, 0, 0, 0, 2, 4, 7, 2, 4, 3, 4, 2, 3, 4};
static const casadi_int casadi_s7[3] = {32, 0, 0};

/* vehicle_model_constr_h_fun_jac_uxt_zt_hess:(i0[3],i1[2],i2[32],i3[],i4[46])->(o0[32],o1[5x32,96nz],o2[5x5,7nz],o3[32x0],o4[]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[0]? arg[0][0] : 0;
  a1=1.2500000000000000e+00;
  a2=arg[0]? arg[0][2] : 0;
  a3=cos(a2);
  a3=(a1*a3);
  a3=(a0-a3);
  a4=arg[4]? arg[4][10] : 0;
  a5=arg[4]? arg[4][12] : 0;
  a6=cos(a5);
  a6=(a1*a6);
  a6=(a4-a6);
  a7=(a3-a6);
  a8=casadi_sq(a7);
  a9=arg[0]? arg[0][1] : 0;
  a10=sin(a2);
  a10=(a1*a10);
  a10=(a9-a10);
  a11=arg[4]? arg[4][11] : 0;
  a12=sin(a5);
  a12=(a1*a12);
  a12=(a11-a12);
  a13=(a10-a12);
  a14=casadi_sq(a13);
  a8=(a8+a14);
  if (res[0]!=0) res[0][0]=a8;
  a8=cos(a5);
  a8=(a1*a8);
  a4=(a4+a8);
  a8=(a3-a4);
  a14=casadi_sq(a8);
  a5=sin(a5);
  a5=(a1*a5);
  a11=(a11+a5);
  a5=(a10-a11);
  a15=casadi_sq(a5);
  a14=(a14+a15);
  if (res[0]!=0) res[0][1]=a14;
  a14=cos(a2);
  a14=(a1*a14);
  a0=(a0+a14);
  a6=(a0-a6);
  a14=casadi_sq(a6);
  a15=sin(a2);
  a15=(a1*a15);
  a9=(a9+a15);
  a12=(a9-a12);
  a15=casadi_sq(a12);
  a14=(a14+a15);
  if (res[0]!=0) res[0][2]=a14;
  a4=(a0-a4);
  a14=casadi_sq(a4);
  a11=(a9-a11);
  a15=casadi_sq(a11);
  a14=(a14+a15);
  if (res[0]!=0) res[0][3]=a14;
  a14=arg[4]? arg[4][14] : 0;
  a15=arg[4]? arg[4][16] : 0;
  a16=cos(a15);
  a16=(a1*a16);
  a16=(a14-a16);
  a17=(a3-a16);
  a18=casadi_sq(a17);
  a19=arg[4]? arg[4][15] : 0;
  a20=sin(a15);
  a20=(a1*a20);
  a20=(a19-a20);
  a21=(a10-a20);
  a22=casadi_sq(a21);
  a18=(a18+a22);
  if (res[0]!=0) res[0][4]=a18;
  a18=cos(a15);
  a18=(a1*a18);
  a14=(a14+a18);
  a18=(a3-a14);
  a22=casadi_sq(a18);
  a15=sin(a15);
  a15=(a1*a15);
  a19=(a19+a15);
  a15=(a10-a19);
  a23=casadi_sq(a15);
  a22=(a22+a23);
  if (res[0]!=0) res[0][5]=a22;
  a16=(a0-a16);
  a22=casadi_sq(a16);
  a20=(a9-a20);
  a23=casadi_sq(a20);
  a22=(a22+a23);
  if (res[0]!=0) res[0][6]=a22;
  a14=(a0-a14);
  a22=casadi_sq(a14);
  a19=(a9-a19);
  a23=casadi_sq(a19);
  a22=(a22+a23);
  if (res[0]!=0) res[0][7]=a22;
  a22=arg[4]? arg[4][18] : 0;
  a23=arg[4]? arg[4][20] : 0;
  a24=cos(a23);
  a24=(a1*a24);
  a24=(a22-a24);
  a25=(a3-a24);
  a26=casadi_sq(a25);
  a27=arg[4]? arg[4][19] : 0;
  a28=sin(a23);
  a28=(a1*a28);
  a28=(a27-a28);
  a29=(a10-a28);
  a30=casadi_sq(a29);
  a26=(a26+a30);
  if (res[0]!=0) res[0][8]=a26;
  a26=cos(a23);
  a26=(a1*a26);
  a22=(a22+a26);
  a26=(a3-a22);
  a30=casadi_sq(a26);
  a23=sin(a23);
  a23=(a1*a23);
  a27=(a27+a23);
  a23=(a10-a27);
  a31=casadi_sq(a23);
  a30=(a30+a31);
  if (res[0]!=0) res[0][9]=a30;
  a24=(a0-a24);
  a30=casadi_sq(a24);
  a28=(a9-a28);
  a31=casadi_sq(a28);
  a30=(a30+a31);
  if (res[0]!=0) res[0][10]=a30;
  a22=(a0-a22);
  a30=casadi_sq(a22);
  a27=(a9-a27);
  a31=casadi_sq(a27);
  a30=(a30+a31);
  if (res[0]!=0) res[0][11]=a30;
  a30=arg[4]? arg[4][22] : 0;
  a31=arg[4]? arg[4][24] : 0;
  a32=cos(a31);
  a32=(a1*a32);
  a32=(a30-a32);
  a33=(a3-a32);
  a34=casadi_sq(a33);
  a35=arg[4]? arg[4][23] : 0;
  a36=sin(a31);
  a36=(a1*a36);
  a36=(a35-a36);
  a37=(a10-a36);
  a38=casadi_sq(a37);
  a34=(a34+a38);
  if (res[0]!=0) res[0][12]=a34;
  a34=cos(a31);
  a34=(a1*a34);
  a30=(a30+a34);
  a34=(a3-a30);
  a38=casadi_sq(a34);
  a31=sin(a31);
  a31=(a1*a31);
  a35=(a35+a31);
  a31=(a10-a35);
  a39=casadi_sq(a31);
  a38=(a38+a39);
  if (res[0]!=0) res[0][13]=a38;
  a32=(a0-a32);
  a38=casadi_sq(a32);
  a36=(a9-a36);
  a39=casadi_sq(a36);
  a38=(a38+a39);
  if (res[0]!=0) res[0][14]=a38;
  a30=(a0-a30);
  a38=casadi_sq(a30);
  a35=(a9-a35);
  a39=casadi_sq(a35);
  a38=(a38+a39);
  if (res[0]!=0) res[0][15]=a38;
  a38=arg[4]? arg[4][26] : 0;
  a39=arg[4]? arg[4][28] : 0;
  a40=cos(a39);
  a40=(a1*a40);
  a40=(a38-a40);
  a41=(a3-a40);
  a42=casadi_sq(a41);
  a43=arg[4]? arg[4][27] : 0;
  a44=sin(a39);
  a44=(a1*a44);
  a44=(a43-a44);
  a45=(a10-a44);
  a46=casadi_sq(a45);
  a42=(a42+a46);
  if (res[0]!=0) res[0][16]=a42;
  a42=cos(a39);
  a42=(a1*a42);
  a38=(a38+a42);
  a42=(a3-a38);
  a46=casadi_sq(a42);
  a39=sin(a39);
  a39=(a1*a39);
  a43=(a43+a39);
  a39=(a10-a43);
  a47=casadi_sq(a39);
  a46=(a46+a47);
  if (res[0]!=0) res[0][17]=a46;
  a40=(a0-a40);
  a46=casadi_sq(a40);
  a44=(a9-a44);
  a47=casadi_sq(a44);
  a46=(a46+a47);
  if (res[0]!=0) res[0][18]=a46;
  a38=(a0-a38);
  a46=casadi_sq(a38);
  a43=(a9-a43);
  a47=casadi_sq(a43);
  a46=(a46+a47);
  if (res[0]!=0) res[0][19]=a46;
  a46=arg[4]? arg[4][30] : 0;
  a47=arg[4]? arg[4][32] : 0;
  a48=cos(a47);
  a48=(a1*a48);
  a48=(a46-a48);
  a49=(a3-a48);
  a50=casadi_sq(a49);
  a51=arg[4]? arg[4][31] : 0;
  a52=sin(a47);
  a52=(a1*a52);
  a52=(a51-a52);
  a53=(a10-a52);
  a54=casadi_sq(a53);
  a50=(a50+a54);
  if (res[0]!=0) res[0][20]=a50;
  a50=cos(a47);
  a50=(a1*a50);
  a46=(a46+a50);
  a50=(a3-a46);
  a54=casadi_sq(a50);
  a47=sin(a47);
  a47=(a1*a47);
  a51=(a51+a47);
  a47=(a10-a51);
  a55=casadi_sq(a47);
  a54=(a54+a55);
  if (res[0]!=0) res[0][21]=a54;
  a48=(a0-a48);
  a54=casadi_sq(a48);
  a52=(a9-a52);
  a55=casadi_sq(a52);
  a54=(a54+a55);
  if (res[0]!=0) res[0][22]=a54;
  a46=(a0-a46);
  a54=casadi_sq(a46);
  a51=(a9-a51);
  a55=casadi_sq(a51);
  a54=(a54+a55);
  if (res[0]!=0) res[0][23]=a54;
  a54=arg[4]? arg[4][34] : 0;
  a55=arg[4]? arg[4][36] : 0;
  a56=cos(a55);
  a56=(a1*a56);
  a56=(a54-a56);
  a57=(a3-a56);
  a58=casadi_sq(a57);
  a59=arg[4]? arg[4][35] : 0;
  a60=sin(a55);
  a60=(a1*a60);
  a60=(a59-a60);
  a61=(a10-a60);
  a62=casadi_sq(a61);
  a58=(a58+a62);
  if (res[0]!=0) res[0][24]=a58;
  a58=cos(a55);
  a58=(a1*a58);
  a54=(a54+a58);
  a58=(a3-a54);
  a62=casadi_sq(a58);
  a55=sin(a55);
  a55=(a1*a55);
  a59=(a59+a55);
  a55=(a10-a59);
  a63=casadi_sq(a55);
  a62=(a62+a63);
  if (res[0]!=0) res[0][25]=a62;
  a56=(a0-a56);
  a62=casadi_sq(a56);
  a60=(a9-a60);
  a63=casadi_sq(a60);
  a62=(a62+a63);
  if (res[0]!=0) res[0][26]=a62;
  a54=(a0-a54);
  a62=casadi_sq(a54);
  a59=(a9-a59);
  a63=casadi_sq(a59);
  a62=(a62+a63);
  if (res[0]!=0) res[0][27]=a62;
  a62=arg[4]? arg[4][38] : 0;
  a63=arg[4]? arg[4][40] : 0;
  a64=cos(a63);
  a64=(a1*a64);
  a64=(a62-a64);
  a65=(a3-a64);
  a66=casadi_sq(a65);
  a67=arg[4]? arg[4][39] : 0;
  a68=sin(a63);
  a68=(a1*a68);
  a68=(a67-a68);
  a69=(a10-a68);
  a70=casadi_sq(a69);
  a66=(a66+a70);
  if (res[0]!=0) res[0][28]=a66;
  a66=cos(a63);
  a66=(a1*a66);
  a62=(a62+a66);
  a3=(a3-a62);
  a66=casadi_sq(a3);
  a63=sin(a63);
  a63=(a1*a63);
  a67=(a67+a63);
  a10=(a10-a67);
  a63=casadi_sq(a10);
  a66=(a66+a63);
  if (res[0]!=0) res[0][29]=a66;
  a64=(a0-a64);
  a66=casadi_sq(a64);
  a68=(a9-a68);
  a63=casadi_sq(a68);
  a66=(a66+a63);
  if (res[0]!=0) res[0][30]=a66;
  a0=(a0-a62);
  a62=casadi_sq(a0);
  a9=(a9-a67);
  a67=casadi_sq(a9);
  a62=(a62+a67);
  if (res[0]!=0) res[0][31]=a62;
  a62=(a7+a7);
  if (res[1]!=0) res[1][0]=a62;
  a67=(a13+a13);
  if (res[1]!=0) res[1][1]=a67;
  a66=sin(a2);
  a66=(a1*a66);
  a62=(a62*a66);
  a63=cos(a2);
  a63=(a1*a63);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][2]=a62;
  a62=(a8+a8);
  if (res[1]!=0) res[1][3]=a62;
  a67=(a5+a5);
  if (res[1]!=0) res[1][4]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][5]=a62;
  a62=(a6+a6);
  if (res[1]!=0) res[1][6]=a62;
  a67=(a12+a12);
  if (res[1]!=0) res[1][7]=a67;
  a70=cos(a2);
  a70=(a1*a70);
  a67=(a67*a70);
  a71=sin(a2);
  a71=(a1*a71);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][8]=a67;
  a67=(a4+a4);
  if (res[1]!=0) res[1][9]=a67;
  a62=(a11+a11);
  if (res[1]!=0) res[1][10]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][11]=a62;
  a62=(a17+a17);
  if (res[1]!=0) res[1][12]=a62;
  a67=(a21+a21);
  if (res[1]!=0) res[1][13]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][14]=a62;
  a62=(a18+a18);
  if (res[1]!=0) res[1][15]=a62;
  a67=(a15+a15);
  if (res[1]!=0) res[1][16]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][17]=a62;
  a62=(a16+a16);
  if (res[1]!=0) res[1][18]=a62;
  a67=(a20+a20);
  if (res[1]!=0) res[1][19]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][20]=a67;
  a67=(a14+a14);
  if (res[1]!=0) res[1][21]=a67;
  a62=(a19+a19);
  if (res[1]!=0) res[1][22]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][23]=a62;
  a62=(a25+a25);
  if (res[1]!=0) res[1][24]=a62;
  a67=(a29+a29);
  if (res[1]!=0) res[1][25]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][26]=a62;
  a62=(a26+a26);
  if (res[1]!=0) res[1][27]=a62;
  a67=(a23+a23);
  if (res[1]!=0) res[1][28]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][29]=a62;
  a62=(a24+a24);
  if (res[1]!=0) res[1][30]=a62;
  a67=(a28+a28);
  if (res[1]!=0) res[1][31]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][32]=a67;
  a67=(a22+a22);
  if (res[1]!=0) res[1][33]=a67;
  a62=(a27+a27);
  if (res[1]!=0) res[1][34]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][35]=a62;
  a62=(a33+a33);
  if (res[1]!=0) res[1][36]=a62;
  a67=(a37+a37);
  if (res[1]!=0) res[1][37]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][38]=a62;
  a62=(a34+a34);
  if (res[1]!=0) res[1][39]=a62;
  a67=(a31+a31);
  if (res[1]!=0) res[1][40]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][41]=a62;
  a62=(a32+a32);
  if (res[1]!=0) res[1][42]=a62;
  a67=(a36+a36);
  if (res[1]!=0) res[1][43]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][44]=a67;
  a67=(a30+a30);
  if (res[1]!=0) res[1][45]=a67;
  a62=(a35+a35);
  if (res[1]!=0) res[1][46]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][47]=a62;
  a62=(a41+a41);
  if (res[1]!=0) res[1][48]=a62;
  a67=(a45+a45);
  if (res[1]!=0) res[1][49]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][50]=a62;
  a62=(a42+a42);
  if (res[1]!=0) res[1][51]=a62;
  a67=(a39+a39);
  if (res[1]!=0) res[1][52]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][53]=a62;
  a62=(a40+a40);
  if (res[1]!=0) res[1][54]=a62;
  a67=(a44+a44);
  if (res[1]!=0) res[1][55]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][56]=a67;
  a67=(a38+a38);
  if (res[1]!=0) res[1][57]=a67;
  a62=(a43+a43);
  if (res[1]!=0) res[1][58]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][59]=a62;
  a62=(a49+a49);
  if (res[1]!=0) res[1][60]=a62;
  a67=(a53+a53);
  if (res[1]!=0) res[1][61]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][62]=a62;
  a62=(a50+a50);
  if (res[1]!=0) res[1][63]=a62;
  a67=(a47+a47);
  if (res[1]!=0) res[1][64]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][65]=a62;
  a62=(a48+a48);
  if (res[1]!=0) res[1][66]=a62;
  a67=(a52+a52);
  if (res[1]!=0) res[1][67]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][68]=a67;
  a67=(a46+a46);
  if (res[1]!=0) res[1][69]=a67;
  a62=(a51+a51);
  if (res[1]!=0) res[1][70]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][71]=a62;
  a62=(a57+a57);
  if (res[1]!=0) res[1][72]=a62;
  a67=(a61+a61);
  if (res[1]!=0) res[1][73]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][74]=a62;
  a62=(a58+a58);
  if (res[1]!=0) res[1][75]=a62;
  a67=(a55+a55);
  if (res[1]!=0) res[1][76]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][77]=a62;
  a62=(a56+a56);
  if (res[1]!=0) res[1][78]=a62;
  a67=(a60+a60);
  if (res[1]!=0) res[1][79]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][80]=a67;
  a67=(a54+a54);
  if (res[1]!=0) res[1][81]=a67;
  a62=(a59+a59);
  if (res[1]!=0) res[1][82]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][83]=a62;
  a62=(a65+a65);
  if (res[1]!=0) res[1][84]=a62;
  a67=(a69+a69);
  if (res[1]!=0) res[1][85]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][86]=a62;
  a62=(a3+a3);
  if (res[1]!=0) res[1][87]=a62;
  a67=(a10+a10);
  if (res[1]!=0) res[1][88]=a67;
  a62=(a62*a66);
  a67=(a67*a63);
  a62=(a62-a67);
  if (res[1]!=0) res[1][89]=a62;
  a62=(a64+a64);
  if (res[1]!=0) res[1][90]=a62;
  a67=(a68+a68);
  if (res[1]!=0) res[1][91]=a67;
  a67=(a67*a70);
  a62=(a62*a71);
  a67=(a67-a62);
  if (res[1]!=0) res[1][92]=a67;
  a67=(a0+a0);
  if (res[1]!=0) res[1][93]=a67;
  a62=(a9+a9);
  if (res[1]!=0) res[1][94]=a62;
  a62=(a62*a70);
  a67=(a67*a71);
  a62=(a62-a67);
  if (res[1]!=0) res[1][95]=a62;
  a62=2.;
  a67=arg[2]? arg[2][31] : 0;
  a71=(a62*a67);
  a70=arg[2]? arg[2][30] : 0;
  a63=(a62*a70);
  a71=(a71+a63);
  a63=arg[2]? arg[2][27] : 0;
  a66=(a62*a63);
  a71=(a71+a66);
  a66=arg[2]? arg[2][26] : 0;
  a72=(a62*a66);
  a71=(a71+a72);
  a72=arg[2]? arg[2][23] : 0;
  a73=(a62*a72);
  a71=(a71+a73);
  a73=arg[2]? arg[2][22] : 0;
  a74=(a62*a73);
  a71=(a71+a74);
  a74=arg[2]? arg[2][19] : 0;
  a75=(a62*a74);
  a71=(a71+a75);
  a75=arg[2]? arg[2][18] : 0;
  a76=(a62*a75);
  a71=(a71+a76);
  a76=arg[2]? arg[2][15] : 0;
  a77=(a62*a76);
  a71=(a71+a77);
  a77=arg[2]? arg[2][14] : 0;
  a78=(a62*a77);
  a71=(a71+a78);
  a78=arg[2]? arg[2][11] : 0;
  a79=(a62*a78);
  a71=(a71+a79);
  a79=arg[2]? arg[2][10] : 0;
  a80=(a62*a79);
  a71=(a71+a80);
  a80=arg[2]? arg[2][7] : 0;
  a81=(a62*a80);
  a71=(a71+a81);
  a81=arg[2]? arg[2][6] : 0;
  a82=(a62*a81);
  a71=(a71+a82);
  a82=arg[2]? arg[2][3] : 0;
  a83=(a62*a82);
  a71=(a71+a83);
  a83=arg[2]? arg[2][2] : 0;
  a84=(a62*a83);
  a71=(a71+a84);
  a84=arg[2]? arg[2][29] : 0;
  a85=(a62*a84);
  a86=arg[2]? arg[2][28] : 0;
  a87=(a62*a86);
  a85=(a85+a87);
  a87=arg[2]? arg[2][25] : 0;
  a88=(a62*a87);
  a85=(a85+a88);
  a88=arg[2]? arg[2][24] : 0;
  a89=(a62*a88);
  a85=(a85+a89);
  a89=arg[2]? arg[2][21] : 0;
  a90=(a62*a89);
  a85=(a85+a90);
  a90=arg[2]? arg[2][20] : 0;
  a91=(a62*a90);
  a85=(a85+a91);
  a91=arg[2]? arg[2][17] : 0;
  a92=(a62*a91);
  a85=(a85+a92);
  a92=arg[2]? arg[2][16] : 0;
  a93=(a62*a92);
  a85=(a85+a93);
  a93=arg[2]? arg[2][13] : 0;
  a94=(a62*a93);
  a85=(a85+a94);
  a94=arg[2]? arg[2][12] : 0;
  a95=(a62*a94);
  a85=(a85+a95);
  a95=arg[2]? arg[2][9] : 0;
  a96=(a62*a95);
  a85=(a85+a96);
  a96=arg[2]? arg[2][8] : 0;
  a97=(a62*a96);
  a85=(a85+a97);
  a97=arg[2]? arg[2][5] : 0;
  a98=(a62*a97);
  a85=(a85+a98);
  a98=arg[2]? arg[2][4] : 0;
  a99=(a62*a98);
  a85=(a85+a99);
  a99=arg[2]? arg[2][1] : 0;
  a100=(a62*a99);
  a85=(a85+a100);
  a100=arg[2]? arg[2][0] : 0;
  a101=(a62*a100);
  a85=(a85+a101);
  a101=(a71+a85);
  if (res[2]!=0) res[2][0]=a101;
  a101=sin(a2);
  a85=(a1*a85);
  a85=(a101*a85);
  a102=sin(a2);
  a71=(a1*a71);
  a71=(a102*a71);
  a85=(a85-a71);
  if (res[2]!=0) res[2][1]=a85;
  a85=(a62*a67);
  a71=(a62*a70);
  a85=(a85+a71);
  a71=(a62*a63);
  a85=(a85+a71);
  a71=(a62*a66);
  a85=(a85+a71);
  a71=(a62*a72);
  a85=(a85+a71);
  a71=(a62*a73);
  a85=(a85+a71);
  a71=(a62*a74);
  a85=(a85+a71);
  a71=(a62*a75);
  a85=(a85+a71);
  a71=(a62*a76);
  a85=(a85+a71);
  a71=(a62*a77);
  a85=(a85+a71);
  a71=(a62*a78);
  a85=(a85+a71);
  a71=(a62*a79);
  a85=(a85+a71);
  a71=(a62*a80);
  a85=(a85+a71);
  a71=(a62*a81);
  a85=(a85+a71);
  a71=(a62*a82);
  a85=(a85+a71);
  a71=(a62*a83);
  a85=(a85+a71);
  a71=(a62*a84);
  a103=(a62*a86);
  a71=(a71+a103);
  a103=(a62*a87);
  a71=(a71+a103);
  a103=(a62*a88);
  a71=(a71+a103);
  a103=(a62*a89);
  a71=(a71+a103);
  a103=(a62*a90);
  a71=(a71+a103);
  a103=(a62*a91);
  a71=(a71+a103);
  a103=(a62*a92);
  a71=(a71+a103);
  a103=(a62*a93);
  a71=(a71+a103);
  a103=(a62*a94);
  a71=(a71+a103);
  a103=(a62*a95);
  a71=(a71+a103);
  a103=(a62*a96);
  a71=(a71+a103);
  a103=(a62*a97);
  a71=(a71+a103);
  a103=(a62*a98);
  a71=(a71+a103);
  a103=(a62*a99);
  a71=(a71+a103);
  a62=(a62*a100);
  a71=(a71+a62);
  a62=(a85+a71);
  if (res[2]!=0) res[2][2]=a62;
  a62=cos(a2);
  a85=(a1*a85);
  a85=(a62*a85);
  a103=cos(a2);
  a71=(a1*a71);
  a71=(a103*a71);
  a85=(a85-a71);
  if (res[2]!=0) res[2][3]=a85;
  a85=sin(a2);
  a85=(a1*a85);
  a71=(a85+a85);
  a71=(a84*a71);
  a104=(a85+a85);
  a104=(a86*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a87*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a88*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a89*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a90*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a91*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a92*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a93*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a94*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a95*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a96*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a97*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a98*a104);
  a71=(a71+a104);
  a104=(a85+a85);
  a104=(a99*a104);
  a71=(a71+a104);
  a85=(a85+a85);
  a85=(a100*a85);
  a71=(a71+a85);
  a85=sin(a2);
  a85=(a1*a85);
  a104=(a85+a85);
  a104=(a67*a104);
  a105=(a85+a85);
  a105=(a70*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a63*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a66*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a72*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a73*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a74*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a75*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a76*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a77*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a78*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a79*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a80*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a81*a105);
  a104=(a104+a105);
  a105=(a85+a85);
  a105=(a82*a105);
  a104=(a104+a105);
  a85=(a85+a85);
  a85=(a83*a85);
  a104=(a104+a85);
  a85=(a71-a104);
  if (res[2]!=0) res[2][4]=a85;
  a85=cos(a2);
  a85=(a1*a85);
  a105=(a85+a85);
  a105=(a67*a105);
  a106=(a85+a85);
  a106=(a70*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a63*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a66*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a72*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a73*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a74*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a75*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a76*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a77*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a78*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a79*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a80*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a81*a106);
  a105=(a105+a106);
  a106=(a85+a85);
  a106=(a82*a106);
  a105=(a105+a106);
  a85=(a85+a85);
  a85=(a83*a85);
  a105=(a105+a85);
  a85=cos(a2);
  a85=(a1*a85);
  a106=(a85+a85);
  a106=(a84*a106);
  a107=(a85+a85);
  a107=(a86*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a87*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a88*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a89*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a90*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a91*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a92*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a93*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a94*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a95*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a96*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a97*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a98*a107);
  a106=(a106+a107);
  a107=(a85+a85);
  a107=(a99*a107);
  a106=(a106+a107);
  a85=(a85+a85);
  a85=(a100*a85);
  a106=(a106+a85);
  a85=(a105-a106);
  if (res[2]!=0) res[2][5]=a85;
  a105=(a1*a105);
  a62=(a62*a105);
  a9=(a9+a9);
  a9=(a9*a67);
  a68=(a68+a68);
  a68=(a68*a70);
  a9=(a9+a68);
  a59=(a59+a59);
  a59=(a59*a63);
  a9=(a9+a59);
  a60=(a60+a60);
  a60=(a60*a66);
  a9=(a9+a60);
  a51=(a51+a51);
  a51=(a51*a72);
  a9=(a9+a51);
  a52=(a52+a52);
  a52=(a52*a73);
  a9=(a9+a52);
  a43=(a43+a43);
  a43=(a43*a74);
  a9=(a9+a43);
  a44=(a44+a44);
  a44=(a44*a75);
  a9=(a9+a44);
  a35=(a35+a35);
  a35=(a35*a76);
  a9=(a9+a35);
  a36=(a36+a36);
  a36=(a36*a77);
  a9=(a9+a36);
  a27=(a27+a27);
  a27=(a27*a78);
  a9=(a9+a27);
  a28=(a28+a28);
  a28=(a28*a79);
  a9=(a9+a28);
  a19=(a19+a19);
  a19=(a19*a80);
  a9=(a9+a19);
  a20=(a20+a20);
  a20=(a20*a81);
  a9=(a9+a20);
  a11=(a11+a11);
  a11=(a11*a82);
  a9=(a9+a11);
  a12=(a12+a12);
  a12=(a12*a83);
  a9=(a9+a12);
  a9=(a1*a9);
  a12=sin(a2);
  a9=(a9*a12);
  a62=(a62-a9);
  a0=(a0+a0);
  a0=(a0*a67);
  a64=(a64+a64);
  a64=(a64*a70);
  a0=(a0+a64);
  a54=(a54+a54);
  a54=(a54*a63);
  a0=(a0+a54);
  a56=(a56+a56);
  a56=(a56*a66);
  a0=(a0+a56);
  a46=(a46+a46);
  a46=(a46*a72);
  a0=(a0+a46);
  a48=(a48+a48);
  a48=(a48*a73);
  a0=(a0+a48);
  a38=(a38+a38);
  a38=(a38*a74);
  a0=(a0+a38);
  a40=(a40+a40);
  a40=(a40*a75);
  a0=(a0+a40);
  a30=(a30+a30);
  a30=(a30*a76);
  a0=(a0+a30);
  a32=(a32+a32);
  a32=(a32*a77);
  a0=(a0+a32);
  a22=(a22+a22);
  a22=(a22*a78);
  a0=(a0+a22);
  a24=(a24+a24);
  a24=(a24*a79);
  a0=(a0+a24);
  a14=(a14+a14);
  a14=(a14*a80);
  a0=(a0+a14);
  a16=(a16+a16);
  a16=(a16*a81);
  a0=(a0+a16);
  a4=(a4+a4);
  a4=(a4*a82);
  a0=(a0+a4);
  a6=(a6+a6);
  a6=(a6*a83);
  a0=(a0+a6);
  a0=(a1*a0);
  a6=cos(a2);
  a0=(a0*a6);
  a104=(a1*a104);
  a102=(a102*a104);
  a0=(a0-a102);
  a62=(a62-a0);
  a10=(a10+a10);
  a10=(a10*a84);
  a69=(a69+a69);
  a69=(a69*a86);
  a10=(a10+a69);
  a55=(a55+a55);
  a55=(a55*a87);
  a10=(a10+a55);
  a61=(a61+a61);
  a61=(a61*a88);
  a10=(a10+a61);
  a47=(a47+a47);
  a47=(a47*a89);
  a10=(a10+a47);
  a53=(a53+a53);
  a53=(a53*a90);
  a10=(a10+a53);
  a39=(a39+a39);
  a39=(a39*a91);
  a10=(a10+a39);
  a45=(a45+a45);
  a45=(a45*a92);
  a10=(a10+a45);
  a31=(a31+a31);
  a31=(a31*a93);
  a10=(a10+a31);
  a37=(a37+a37);
  a37=(a37*a94);
  a10=(a10+a37);
  a23=(a23+a23);
  a23=(a23*a95);
  a10=(a10+a23);
  a29=(a29+a29);
  a29=(a29*a96);
  a10=(a10+a29);
  a15=(a15+a15);
  a15=(a15*a97);
  a10=(a10+a15);
  a21=(a21+a21);
  a21=(a21*a98);
  a10=(a10+a21);
  a5=(a5+a5);
  a5=(a5*a99);
  a10=(a10+a5);
  a13=(a13+a13);
  a13=(a13*a100);
  a10=(a10+a13);
  a10=(a1*a10);
  a13=sin(a2);
  a10=(a10*a13);
  a106=(a1*a106);
  a103=(a103*a106);
  a10=(a10+a103);
  a62=(a62+a10);
  a3=(a3+a3);
  a3=(a3*a84);
  a65=(a65+a65);
  a65=(a65*a86);
  a3=(a3+a65);
  a58=(a58+a58);
  a58=(a58*a87);
  a3=(a3+a58);
  a57=(a57+a57);
  a57=(a57*a88);
  a3=(a3+a57);
  a50=(a50+a50);
  a50=(a50*a89);
  a3=(a3+a50);
  a49=(a49+a49);
  a49=(a49*a90);
  a3=(a3+a49);
  a42=(a42+a42);
  a42=(a42*a91);
  a3=(a3+a42);
  a41=(a41+a41);
  a41=(a41*a92);
  a3=(a3+a41);
  a34=(a34+a34);
  a34=(a34*a93);
  a3=(a3+a34);
  a33=(a33+a33);
  a33=(a33*a94);
  a3=(a3+a33);
  a26=(a26+a26);
  a26=(a26*a95);
  a3=(a3+a26);
  a25=(a25+a25);
  a25=(a25*a96);
  a3=(a3+a25);
  a18=(a18+a18);
  a18=(a18*a97);
  a3=(a3+a18);
  a17=(a17+a17);
  a17=(a17*a98);
  a3=(a3+a17);
  a8=(a8+a8);
  a8=(a8*a99);
  a3=(a3+a8);
  a7=(a7+a7);
  a7=(a7*a100);
  a3=(a3+a7);
  a3=(a1*a3);
  a2=cos(a2);
  a3=(a3*a2);
  a1=(a1*a71);
  a101=(a101*a1);
  a3=(a3+a101);
  a62=(a62+a3);
  if (res[2]!=0) res[2][6]=a62;
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_model_constr_h_fun_jac_uxt_zt_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int vehicle_model_constr_h_fun_jac_uxt_zt_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_model_constr_h_fun_jac_uxt_zt_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_model_constr_h_fun_jac_uxt_zt_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int vehicle_model_constr_h_fun_jac_uxt_zt_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_model_constr_h_fun_jac_uxt_zt_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void vehicle_model_constr_h_fun_jac_uxt_zt_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void vehicle_model_constr_h_fun_jac_uxt_zt_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int vehicle_model_constr_h_fun_jac_uxt_zt_hess_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int vehicle_model_constr_h_fun_jac_uxt_zt_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real vehicle_model_constr_h_fun_jac_uxt_zt_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_model_constr_h_fun_jac_uxt_zt_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_model_constr_h_fun_jac_uxt_zt_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_model_constr_h_fun_jac_uxt_zt_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_model_constr_h_fun_jac_uxt_zt_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s7;
    case 4: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int vehicle_model_constr_h_fun_jac_uxt_zt_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
