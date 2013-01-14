#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/BLAS.h>

#define UNDERSCALE (1e-140)
#define OVERSCALE  (1e+140)

static int one = 1; /* for BLAS routines */

/* `buf` is at least 3*m-2 long */ 
static double K (int n, double d, double *buf) {
  int i, j, ns = 0; /* ns is number of exp shifts */
  double u, s;
  int k = (int) (n * d) + 1; /* ceiling(n * d) */
  int m = 2 * k - 1;
  double h = k - n * d;
  /* === initialize v, w, and q === */
  double *v = buf;
  double *w = v + m;
  double *q = w + m - 2;
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
    s = (double) i / n; u = q[0];
    q[0] = F77_NAME(ddot)(&m, v, &one, q, &one) * s; /* no shift */
    for (j = 1; j < m - 1; j++) {
      double a = u; /* q[j - 1] */
      int m1 = m - 1 - j;
      u = q[j];
      a += F77_NAME(ddot)(&m1, w, &one, q + j, &one) + v[m1] * q[m - 1];
      q[j] = a * s;
    }
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
  buf = (double *) Calloc(3 * mm - 2, double);
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


/* Interface */

static const R_CallMethodDef callMethods[] = {
  {"pkolmim", (DL_FUNC) &pkolmim, 3},
  {"pkolmim2x", (DL_FUNC) &pkolmim2x, 2},
  {NULL, NULL, 0}
};

void R_init_kolmim (DllInfo *info) {
  R_registerRoutines(info, NULL, callMethods, NULL, NULL);
}

