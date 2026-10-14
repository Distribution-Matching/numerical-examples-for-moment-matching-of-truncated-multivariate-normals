# tmvtnorm_loglik.R
#
# Check for the tmvtnorm comparison: evaluate the truncated-normal
# log-likelihood of every sample at the mle.tmvnorm estimate and at the
# fit_mvnormal(x̄, S) estimate, with tmvtnorm's own dtmvnorm, and the
# moment residuals of both with tmvtnorm's own mtmvnorm. Run after
# tmvtnorm_comparison.jl.
#
# Usage: Rscript experiments/tmvtnorm_loglik.R <results dir>

suppressMessages(library(tmvtnorm))
args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "experiments/results/tmvtnorm"
num <- function(s) as.numeric(strsplit(s, ";")[[1]])
panel <- read.csv(file.path(outdir, "panel.csv"), stringsAsFactors = FALSE)
rfits <- read.csv(file.path(outdir, "r_fits.csv"), stringsAsFactors = FALSE)
ours <- read.csv(file.path(outdir, "ours_fits.csv"), stringsAsFactors = FALSE)

loglik <- function(X, mu, Sig, lo, up)
  sum(dtmvnorm(X, mean = mu, sigma = Sig, lower = lo, upper = up, log = TRUE))
resid <- function(mu, Sig, lo, up, xb, S) {
  m <- mtmvnorm(mu, Sig, lo, up)
  c(max(abs(m$tmean - xb)), max(abs(m$tvar - S)))
}
cat(sprintf("%-5s %-13s %14s %10s %10s\n", "ex.", "estimate", "loglik", "max|dmu|", "max|dS|"))
for (r in seq_len(nrow(panel))) {
  id <- panel$id[r]; n <- panel$n[r]; lo <- num(panel$a[r]); up <- num(panel$b[r])
  X <- as.matrix(read.csv(file.path(outdir, sprintf("X_%s.csv", id)), header = FALSE))
  xb <- colMeans(X); S <- cov(X) * (nrow(X) - 1) / nrow(X)
  rr <- rfits[rfits$id == id & rfits$method == "mle", ]
  cf <- num(rr$coef); nm <- strsplit(rr$names, ";")[[1]]
  mu_r <- cf[1:n]; Sig_r <- matrix(0, n, n)
  for (k in (n + 1):length(cf)) {
    ij <- as.integer(strsplit(sub("sigma_", "", nm[k]), ".", fixed = TRUE)[[1]])
    Sig_r[ij[1], ij[2]] <- Sig_r[ij[2], ij[1]] <- cf[k]
  }
  oo <- ours[ours$id == id, ]
  mu_o <- num(oo$mu); Sig_o <- matrix(num(oo$Sigma), n, n, byrow = TRUE)
  set.seed(1)
  for (est in list(list("mle.tmvnorm", mu_r, Sig_r), list("fit_mvnormal", mu_o, Sig_o))) {
    res <- resid(est[[2]], est[[3]], lo, up, xb, S)
    cat(sprintf("%-5s %-13s %14.4f %10.1e %10.1e\n", id, est[[1]],
                loglik(X, est[[2]], est[[3]], lo, up), res[1], res[2]))
  }
}
