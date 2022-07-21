# Importing packages
library(microbenchmark)
library(purrr)
m <- matrix(abs(rnorm(n = 5*10^3)),nrow = 5000)
m <- tcrossprod(m)
m <- m + diag(nrow = nrow(m))
eigen <- M_solve(m)
inv_arma <- inverse_arma(m)
arma <- inverse_arma_sim(m)


micro_sum <- microbenchmark::microbenchmark(M_solve(m),
                               inverse_arma(m),
                               inverse_arma_sim(m),
                               chol2inv(chol(m)),times = 10)
micro_sum %>% summary
