\name{pkolmim}
\alias{pkolmim}
\title{Kolmogorov Dn Distribution}
\description{
  Cumulative distribution function for Kolmogorov's goodness-of-fit measure.
}
\usage{
pkolmim(d, n)
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

  Wang, Tsang, and Marsaglia (2003) have proposed an algorithm to compute the
  cumulative distribution function \eqn{K(n, d) = P(D_n < d)}.
  \code{pkolmim} offers an improved implementation that uses less memory and
  should be more efficient for a range of arguments that are common in
  practice, while keeping the same precision.

  The original algorithm of Wang, Tsang, and Marsaglia is implemented in the C
  routine \code{pkolmogorov2x} that is used by \code{\link{ks.test}} (package
  \code{stats}) for one-sample two-sided exact tests. Similarly,
  \code{pkolmim} is used by \code{\link{ks.test.imp}} in package
  \code{kolmim}.
}
\value{
  Returns \eqn{K(n, d) = P(D_n < d)}.
}
\source{
  The two-sided one-sample distribution comes \emph{via} Carvalho (2013).
}
\references{
  Luis Carvalho (2013),
  \emph{An improved evaluation of Kolmogorov's distribution}.

  George Marsaglia, Wai Wan Tsang and Jingbo Wang (2003),
  Evaluating Kolmogorov's distribution.
  \emph{Journal of Statistical Software}, \bold{8}/18.
  \url{http://www.jstatsoft.org/v08/i18/}.
}
\seealso{
  \code{\link{ks.test.imp}} for a Kolmogorov-Smirnov test similar to
  \code{\link{ks.test}} but that uses \code{pkolmim} for one-sample two-sided
  exact tests.
}
\examples{
n <- 100
x <- 1:100 / 500
plot(x, pkolmim(x, n), type='l')

# Wang et al. approximation
s <- x ^ 2 * n
ps <- pmax(0, 1 - 2 * exp(-(2.000071 + .331 / sqrt(n) + 1.409 / n) * s))
lines(x, ps, lty=2)
}
