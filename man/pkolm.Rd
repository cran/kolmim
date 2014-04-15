\name{pkolm}
\alias{pkolm}
\title{Kolmogorov Dn Distribution}
\description{
  Cumulative distribution function for Kolmogorov's goodness-of-fit measure.
}
\usage{
pkolm(d, n)
}
\arguments{
  \item{d}{the argument for the cumulative distribution function of Dn.}
  \item{n}{the number of variates.}
}
\details{
  Given an ordered set of \code{n} standard uniform variates,
  \eqn{x_1 < \dots < x_n}, Kolmogorov suggested \eqn{D_n = \max[D_n^-, D_n^+]}
  as a goodness-of-fit measure, where:
  \eqn{D_n^- = \max_{i=1, \dots n} [x_i - (i - 1) / n]} and
  \eqn{D_n^+ = \max_{i=1, \dots n} [i / n - x_i]}.

  \code{pkolm} provides the original algorithm proposed by Wang, Tsang, and
  Marsaglia (2003) to compute the cumulative distribution function
  \eqn{K(n, d) = P(D_n < d)}. This routine is used by \code{\link{ks.test}}
  (package \code{stats}) for one-sample two-sided exact tests, and it is
  implemented in the C routine \code{pkolmogorov2x}. \code{pkolm} is a simple
  wrap around \code{pkolmogorov2x}.
}
\value{
  Returns \eqn{K(n, d) = P(D_n < d)}.
}
\source{
  The two-sided one-sample distribution comes \emph{via} Wang, Tsang, and
  Marsaglia (2003).
}
\references{
  George Marsaglia, Wai Wan Tsang and Jingbo Wang (2003),
  Evaluating Kolmogorov's distribution.
  \emph{Journal of Statistical Software}, \bold{8}/18.
  \url{http://www.jstatsoft.org/v08/i18/}.
}
\seealso{
  \code{pkolmim} for an improved routine to compute \eqn{K(n, d)}, and
  \code{\link{ks.test}} for the Kolmogorov-Smirnov test.
}
\examples{
n <- 100
x <- 1:100 / 500
plot(x, sapply(x, function (x) pkolm(x, n)), type='l')

# Wang et al. approximation
s <- x ^ 2 * n
ps <- pmax(0, 1 - 2 * exp(-(2.000071 + .331 / sqrt(n) + 1.409 / n) * s))
lines(x, ps, lty=2)
}
