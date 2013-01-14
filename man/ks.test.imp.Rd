\name{ks.test.imp}
\alias{ks.test.imp}
\title{Kolmogorov-Smirnov Tests}
\description{
  Perform a one-sample two-sided exact Kolmogorov-Smirnov test, similarly to
  \code{\link{ks.test}} from package \code{stats}, but using an improved
  routine.
}
\usage{
ks.test.imp(x, y, \dots)
}
\arguments{
  \item{x}{a numeric vector of data values.}
  \item{y}{either a numeric vector of data values, or a character string
    naming a cumulative distribution function or an actual cumulative
    distribution function such as \code{pnorm}.  Only continuous CDFs
    are valid.}
  \item{\dots}{parameters of the distribution specified (as a character
    string) by \code{y}.}
}
\details{
  This routine is equivalent to \code{ks.test(x, y, ..., exact=TRUE)} but uses
  an improved method based on \code{\link{pkolmim}}. For more details about
  the arguments, please refer to the documentation for \code{\link{ks.test}}.
}
\value{
  A list with class \code{"htest"} containing the following components:
  \item{statistic}{the value of the test statistic.}
  \item{p.value}{the p-value of the test.}
  \item{alternative}{"two-sided".}
  \item{method}{a character string indicating what type of test was
    performed.}
  \item{data.name}{a character string giving the name(s) of the data.}
}
\source{
  The two-sided one-sample distribution comes \emph{via}
  Carvalho (2013).
}
\references{
  Luis Carvalho (2013),
  \emph{An improved evaluation of Kolmogorov's distribution}.
}
\seealso{
  \code{\link{pkolmim}} for the cumulative distribution function of
  Kolmogorov's goodness-of-fit measure.
}
\examples{
x <- abs(rnorm(100))
p.kt <- ks.test(x, "pexp", exact = TRUE)$p
p.ktimp <- ks.test.imp(x, "pexp")$p
abs(p.kt - p.ktimp)

# compare execution times
x <- abs(rnorm(2000))
system.time(ks.test.imp(x, "pexp"))
system.time(ks.test(x, "pexp", exact = TRUE))
}
\keyword{htest}
