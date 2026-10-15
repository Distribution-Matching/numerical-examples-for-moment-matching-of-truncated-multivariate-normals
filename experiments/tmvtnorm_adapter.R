# tmvtnorm_adapter.R
#
# R half of the "tmvtnorm as a moment matcher" baseline. The truncated
# normal is an exponential family with sufficient statistics (x, x x'),
# so any data set with sample mean mu_hat and (1/N) sample covariance
# Sigma_hat has the moment-matching solution as its maximum-likelihood
# estimate. This script builds such a pseudo-data set for every target in
# targets.csv (written by tmvtnorm_adapter.jl), with every point inside the
# box, and runs mle.tmvnorm and gmm.tmvnorm on it.
#
# Design: x_k = mu_hat + L Q z_k with z_k the rows of a Hadamard design
# (entries +-1, columns orthogonal with mean 0, so mean(z) = 0 and
# (1/N) sum z z' = I), L a square root of Sigma_hat and Q orthogonal. N is
# the smallest power of two above n. We try the symmetric and Cholesky
# square roots first, then random rotations Q, then the 2n-point design
# +-sqrt(n) e_i, and take the first design whose points are all inside A.
#
# Usage: Rscript experiments/tmvtnorm_adapter.R <results dir>

suppressMessages(library(tmvtnorm))

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "experiments/results/tmvtnorm_adapter"
num <- function(s) as.numeric(strsplit(s, ";")[[1]])
targets <- read.csv(file.path(outdir, "targets.csv"), stringsAsFactors = FALSE)

hadamard <- function(N) {
  H <- matrix(1, 1, 1)
  while (nrow(H) < N) H <- rbind(cbind(H, H), cbind(H, -H))
  H
}

inside <- function(X, lo, up) all(sweep(X, 2, lo, ">") & sweep(X, 2, up, "<"))

pseudo_data <- function(mu, S, lo, up, n) {
  N <- 2^ceiling(log2(n + 1))
  Z <- hadamard(N)[, 2:(n + 1), drop = FALSE]          # drop the all-ones column
  e <- eigen(S, symmetric = TRUE)
  roots <- list(e$vectors %*% diag(sqrt(e$values), n) %*% t(e$vectors), t(chol(S)))
  build <- function(L, Q) sweep(Z %*% t(L %*% Q), 2, mu, "+")
  for (L in roots) {
    X <- build(L, diag(n)); if (inside(X, lo, up)) return(list(X = X, design = "hadamard"))
  }
  set.seed(1)
  for (k in 1:2000) {
    Q <- qr.Q(qr(matrix(rnorm(n * n), n)))
    for (L in roots) {
      X <- build(L, Q); if (inside(X, lo, up)) return(list(X = X, design = "hadamard-rotated"))
    }
  }
  Zs <- rbind(sqrt(n) * diag(n), -sqrt(n) * diag(n))
  for (L in roots) {
    X <- sweep(Zs %*% t(L), 2, mu, "+")
    if (inside(X, lo, up)) return(list(X = X, design = "axial"))
  }
  NULL
}

rows <- list()
for (r in seq_len(nrow(targets))) {
  id <- targets$id[r]; n <- targets$n[r]
  mu <- num(targets$mu_hat[r]); S <- matrix(num(targets$Sigma_hat[r]), n, n, byrow = TRUE)
  lo <- num(targets$a[r]); up <- num(targets$b[r])
  pd <- pseudo_data(mu, S, lo, up, n)
  if (is.null(pd)) {
    cat(sprintf("%-5s no in-box pseudo-data set found\n", id))
    for (method in c("mle", "gmm"))
      rows[[length(rows) + 1]] <- data.frame(id = id, method = method, N = NA,
        design = "none", time = NA, status = "no design", convergence = NA,
        names = "", coef = "")
    next
  }
  X <- pd$X; N <- nrow(X)
  # sanity: the pseudo-data reproduce the targets exactly
  stopifnot(max(abs(colMeans(X) - mu)) < 1e-10,
            max(abs(cov(X) * (N - 1) / N - S)) < 1e-10)
  start <- list(mu = mu, sigma = S)
  for (method in c("mle", "gmm")) {
    est <- rep(NA_real_, n + n * (n + 1) / 2); nm <- rep("", length(est))
    status <- "ok"; conv <- NA
    set.seed(2000 + r)
    t <- system.time({
      fit <- tryCatch(
        if (method == "mle") mle.tmvnorm(X, lo, up, start = start)
        # i.i.d. weighting: the default HAC weighting needs many more
        # observations, and for this just-identified system the weighting
        # matrix does not affect the solution.
        else gmm.tmvnorm(X, lo, up, start = start, vcov = "iid"),
        error = function(e) { status <<- paste("error:", conditionMessage(e)); NULL })
    })[["elapsed"]]
    if (!is.null(fit)) {
      est <- as.numeric(coef(fit)); nm <- names(coef(fit))
      conv <- if (method == "mle") fit@details$convergence else fit$algoInfo$convergence
    }
    cat(sprintf("%-5s %-4s N=%2d %-17s %8.2fs  %s  convergence=%s\n", id, method, N,
                pd$design, t, status, conv))
    rows[[length(rows) + 1]] <- data.frame(id = id, method = method, N = N,
      design = pd$design, time = t, status = status, convergence = conv,
      names = paste(nm, collapse = ";"), coef = paste(format(est, digits = 17), collapse = ";"))
  }
}
write.csv(do.call(rbind, rows), file.path(outdir, "r_fits.csv"), row.names = FALSE)
