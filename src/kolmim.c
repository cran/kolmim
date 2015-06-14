/*
 *  kolmim: a R package for improved evaluation of Kolmogorov's distribution
 *  Copyright (C) 2013   Luis Carvalho
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 */


#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/BLAS.h>

#define UNDERSCALE (1e-140)
#define OVERSCALE  (1e+140)

static int one = 1; /* for BLAS routines */

/* `buf` is at least max{2, m + m + (m - 2)} long */ 
static double K (int n, double d, double *buf) {
  int i, j, ns = 0; /* ns is number of exp shifts */
  double u, s;
  int k = (int) (n * d) + 1; /* ceiling(n * d) */
  int m = 2 * k - 1;
  double h = k - n * d;
  /* === initialize v, w, and q === */
  double *v = buf;
  double *q = v + m;
  double *w = q + m; /* 'w' is m - 2 long */
  for (i = 0; i < m; i++) q[i] = 0;
  q[k - 1] = u = s = 1.0; /* q = e_k */
  for (j = 0; j < m - 1; j++) {
    s /= j + 1; u *= h;
    if (j < m - 2) w[j] = s;
    v[j] = (1 - u) * s;
  }
  v[m - 1] = (1 - 2 * u * h + ((h > 0.5) ? pow(2 * h - 1, m) : 0)) * s / m;
  /* === iterate === */
  for (i = 1; i <= n; i++) {
    s = ((double) i) / n; u = q[0];
    q[0] = F77_NAME(ddot)(&m, v, &one, q, &one) * s; /* no shift */
    for (j = 1; j < m - 1; j++) {
      double a = u; /* q[j - 1] */
      int m1 = m - 1 - j;
      u = q[j];
      a += F77_NAME(ddot)(&m1, w, &one, q + j, &one) + v[m1] * q[m - 1];
      q[j] = a * s;
    }
    if (m > 1) /* shift? */
      q[m - 1] = (v[0] * q[m - 1] + u) * s;
    /* check for under/overflows */
    if (q[k - 1] > OVERSCALE) {
      double alpha = UNDERSCALE;
      F77_NAME(dscal)(&m, &alpha, q, &one);
      ns++;
    }
    if (q[k - 1] < UNDERSCALE) {
      double alpha = OVERSCALE;
      F77_NAME(dscal)(&m, &alpha, q, &one);
      ns--;
    }
  }
  /* === return result === */
  s = q[k - 1];
  if (ns != 0) s *= pow(OVERSCALE, ns); /* rescale if necessary */
  return s;
}


/* assume there are `nx` elements in `x` and `n` */
SEXP pkolmim (SEXP sx, SEXP sn, SEXP snx) {
  double *x = REAL(sx);
  int *n = INTEGER(sn);
  int nx = asInteger(snx);
  int i, k, m, mm;
  double *p, *buf;
  SEXP sp;
  /* find buffer size */
  mm = 0; /* max m */
  for (i = 0; i < nx; i++) {
    k = (int) (n[i] * x[i]) + 1; /* ceiling(n[i] * x[i]) */
    m = 2 * k - 1;
    if (mm < m) mm = m;
  }
  /* iterate */
  buf = (double *) Calloc((mm > 1) ? (3 * mm - 2) : 2, double);
  PROTECT(sp = allocVector(REALSXP, nx));
  p = REAL(sp);
  for (i = 0; i < nx; i++) p[i] = K(n[i], x[i], buf);
  UNPROTECT(1);
  Free(buf);
  return sp;
}

SEXP pkolmim2x (SEXP sx, SEXP sn) {
  double x = asReal(sx);
  int n = asInteger(sn);
  int k = (int) (n * x) + 1;
  int m = 2 * k - 1;
  double *buf = (double *) Calloc(3 * m - 2, double);
  double p = K(n, x, buf);
  Free(buf);
  return ScalarReal(p);
}

/* from ks.c */
double K0(int n, double d);
void m_multiply(double *A, double *B, double *C, int m);
void m_power(double *A, int eA, double *V, int *eV, int m, int n);

/* The two-sided one-sample 'exact' distribution */
SEXP pKolmogorov2x(SEXP statistic, SEXP sn) {
  int n = asInteger(sn);
  double st = asReal(statistic);
  double p = K0(n, st);
  return ScalarReal(p);
}

/* Interface */

static const R_CallMethodDef callMethods[] = {
  {"pkolmim", (DL_FUNC) &pkolmim, 3},
  {"pkolmim2x", (DL_FUNC) &pkolmim2x, 2},
  {"pKolmogorov2x", (DL_FUNC) &pKolmogorov2x, 2},
  {NULL, NULL, 0}
};

void R_init_kolmim (DllInfo *info) {
  R_registerRoutines(info, NULL, callMethods, NULL, NULL);
}

