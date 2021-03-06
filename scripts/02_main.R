library(tidyverse)
library(readxl)
source("src/functions-clustering.R")
source("src/functions-portfolio.R")


# random seed
set.seed(123)

# load data
stock_tbl <- read_csv("data/processed/stock.csv")
kospi <- read_csv("data/processed/kospi.csv")
risk_free <- read_csv("data/processed/risk_free.csv")

# models
with_list <- c("return", "market_residual", "factors", "factors_residual")
n_time_list <- c(6, 8, 10, 12)
method_list <- c("GMV", "Tangency")

# validation period
start_list <- str_c(c("2002", "2005", "2008", "2011"), "-4")
end_list <- str_c(c("2005", "2008", "2011", "2014"), "-3")
valid_res_list <- list()
for (i in 1:4) {
  start <- start_list[i]
  end <- end_list[i]
  valid_res_list[[i]] <-
    evaluate_portfolio(stock_tbl, kospi, risk_free, start, end,
                       with_list, n_time_list, method_list)
}

# test period
start <- "2014-4"
end <- "2017-3"
test_res <- evaluate_portfolio(stock_tbl, kospi, risk_free, start, end,
                                  with_list, n_time_list, method_list)

# save results
dir.create("outputs", showWarnings = FALSE)
save(valid_res_list, file = "outputs/valid_res_list.RData")
save(test_res, file = "outputs/test_res.RData")
