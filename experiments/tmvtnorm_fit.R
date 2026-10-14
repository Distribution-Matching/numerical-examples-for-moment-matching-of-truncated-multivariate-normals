# tmvtnorm_fit.R
#
# R half of the tmvtnorm comparison (paper Section on the normal case,
# Table "tmvtnorm"). Called by tmvtnorm_comparison.jl, which writes the
# panel to results/tmvtnorm/panel.csv. For every panel row this script
#   1. draws N i.i.d. points from the box-truncated normal (rtmvnorm,
#      rejection, seeded per row) and writes them to X_<id>.csv;
#   2. fits the underlying (mu, Sigma) from the data with
#      tmvtnorm::mle.tmvnorm (BFGS on the likelihood, numerical gradient)
#      and tmvtnorm::gmm.tmvnorm (method of moments, "ManjunathWilhelm"),
#      both started from the sample mean and covariance;
#   3. writes wall-clock and estimates to r_fits.csv.
#
# Usage: Rscript experiments/tmvtnorm_fit.R <results dir> <N>

suppressMessages(library(tmvtnorm))

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "experiments/results/tmvtnorm"
N <- if (length(args) >= 2) as.integer(args[2]) else 2000L

panel <- read.csv(file.path(outdir, "panel.csv"), stringsAsFactors = FALSE)
num <- function(s) as.numeric(strsplit(s, ";")[[1]])

rows <- list()
for (r in seq_len(nrow(panel))) {
  id <- panel$id[r]; n <- panel$n[r]
  mu <- num(panel$mu[r]); Sigma <- matrix(num(panel$Sigma[r]), n, n, byrow = TRUE)
  lo <- num(panel$a[r]); up <- num(panel$b[r])
  set.seed(1000 + r)
  X <- rtmvnorm(N, mean = mu, sigma = Sigma, lower = lo, upper = up,
                algorithm = "rejection")
  write.table(X, file.path(outdir, sprintf("X_%s.csv", id)), sep = ",",
              row.names = FALSE, col.names = FALSE)
  start <- list(mu = colMeans(X), sigma = cov(X))
  for (method in c("mle", "gmm")) {
    est <- rep(NA_real_, n + n * (n + 1) / 2); nm <- rep("", length(est))
    status <- "ok"
    t <- system.time({
      fit <- tryCatch(
        if (method == "mle") mle.tmvnorm(X, lo, up, start = start)
        else gmm.tmvnorm(X, lo, up, start = start),
        error = function(e) { status <<- paste("error:", conditionMessage(e)); NULL })
    })[["elapsed"]]
    conv <- NA
    if (!is.null(fit)) {
      est <- as.numeric(coef(fit)); nm <- names(coef(fit))
      # optim convergence code (0 = success) for mle; gmm's algoInfo code
      conv <- if (method == "mle") fit@details$convergence else fit$algoInfo$convergence
    }
    cat(sprintf("%-5s %-4s %8.2fs  %s  convergence=%s\n", id, method, t, status, conv))
    rows[[length(rows) + 1]] <- data.frame(id = id, method = method, time = t,
      status = status, convergence = conv, names = paste(nm, collapse = ";"),
      coef = paste(format(est, digits = 17), collapse = ";"))
  }
}
write.csv(do.call(rbind, rows), file.path(outdir, "r_fits.csv"), row.names = FALSE)
