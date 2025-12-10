# R/data_preprocessing.R

library(tidyquant)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(ggplot2)

DATA_DIR <- file.path("data")
RAW_DIR  <- file.path(DATA_DIR, "raw")

if (!dir.exists(RAW_DIR)) {
  dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)
}

# Helper: check that we got exactly 480 monthly rows from 1985-01 to 2024-12
.check_480 <- function(df) {
  ok <- nrow(df) == 480 &&
    min(df$predicting_date) == as.Date("1985-01-01") &&
    max(df$predicting_date) == as.Date("2024-12-01")
  if (!ok) {
    warning("Total_data is not 480 rows from 1985-01-01 to 2024-12-01. Check cleaning.")
  }
}

# Main function used by the Rmd
build_total_data <- function() {

  # 1. S&P 500 daily OHLCV for Python and monthly returns for ECON370
  sp500_daily <- tq_get("^GSPC",
                        from = "1984-01-01",
                        to   = "2024-12-31",
                        get  = "stock.prices")

  # Write daily OHLCV in the format expected by Python's data_preprocessing.py
  sp500_daily_out <- sp500_daily %>%
    transmute(
      date   = as_date(date),
      Open   = open,
      High   = high,
      Low    = low,
      Close  = close,
      Volume = volume
    )

  write_csv(sp500_daily_out, file.path(RAW_DIR, "sp500_df.csv"))

  # Monthly simple returns for ECON work
  SP500_simple_returns <- sp500_daily %>%
    tq_transmute(
      select     = adjusted,
      mutate_fun = periodReturn,
      period     = "monthly",
      type       = "arithmetic",
      col_rename = "monthly_return"
    ) %>%
    mutate(
      monthly_return = monthly_return * 100,  # percentage
      date           = floor_date(date, unit = "month")
    )

  # 2. GDP growth (FRED, quarterly, will be lagged 4 months)
  gdp_data <- tq_get("A191RL1Q225SBEA",
                     get  = "economic.data",
                     from = "1984-01-01") %>%
    rename(GDP_g = price) %>%
    mutate(
      # shift to quarter end
      date = date %m+% months(2)
    )

  # 3. Personal saving rate (FRED, monthly) - for Python and ECON
  psavert <- tq_get("PSAVERT",
                    get  = "economic.data",
                    from = "1984-01-01") %>%
    transmute(date, Personal_Savings_Rate = price)

  write_csv(psavert, file.path(RAW_DIR, "psavert_df.csv"))

  # 4. Unemployment rate (FRED, monthly) - for Python and ECON
  unemploy <- tq_get("UNRATE",
                     get  = "economic.data",
                     from = "1984-01-01") %>%
    transmute(date, Unemployment = price)

  write_csv(unemploy, file.path(RAW_DIR, "unemploy_df.csv"))

  # Extra predictors to hit 10+ variables and 4+ sources

  # 5. CPI (FRED, monthly, we will use inflation rate)
  cpi <- tq_get("CPIAUCSL",
                get  = "economic.data",
                from = "1984-01-01") %>%
    transmute(date, CPI = price)

  # 6. Fed funds rate (FRED)
  fedfunds <- tq_get("FEDFUNDS",
                     get  = "economic.data",
                     from = "1984-01-01") %>%
    transmute(date, FedFunds = price)

  # 7. 10-year Treasury yield (FRED)
  gs10 <- tq_get("GS10",
                 get  = "economic.data",
                 from = "1984-01-01") %>%
    transmute(date, GS10 = price)

  # 8. 10y minus 2y term spread (FRED)
  term_spread <- tq_get("T10Y2Y",
                        get  = "economic.data",
                        from = "1984-01-01") %>%
    transmute(date, Term_Spread = price)

  # 9. BAA corporate bond yield (FRED) as credit risk proxy
  baa <- tq_get("BAA",
                get  = "economic.data",
                from = "1984-01-01") %>%
    transmute(date, BAA_Yield = price)

  # 10. VIX index (Yahoo via tidyquant)
  vix_daily <- tq_get("^VIX",
                      from = "1984-01-01",
                      to   = "2024-12-31",
                      get  = "stock.prices")

  vix_monthly <- vix_daily %>%
    tq_transmute(
      select     = adjusted,
      mutate_fun = periodReturn,
      period     = "monthly",
      type       = "arithmetic",
      col_rename = "VIX_ret"
    ) %>%
    mutate(
      VIX_ret = VIX_ret * 100,
      date    = floor_date(date, unit = "month")
    )

  # Align everything at monthly frequency by month begin
  base_monthly <- SP500_simple_returns %>%
    mutate(date = floor_date(date, unit = "month"))

  Total_data <- base_monthly %>%
    left_join(gdp_data,    by = "date") %>%
    left_join(psavert,     by = "date") %>%
    left_join(unemploy,    by = "date") %>%
    left_join(cpi,         by = "date") %>%
    left_join(fedfunds,    by = "date") %>%
    left_join(gs10,        by = "date") %>%
    left_join(term_spread, by = "date") %>%
    left_join(baa,         by = "date") %>%
    left_join(vix_monthly, by = "date")

  # Create lagged returns and inflation, and deal with mixed frequency / lags
  Total_data <- Total_data %>%
    arrange(date) %>%
    # fill quarterly / missing macro values forward
    tidyr::fill(
      GDP_g,
      Personal_Savings_Rate,
      Unemployment,
      CPI,
      FedFunds,
      GS10,
      Term_Spread,
      BAA_Yield,
      VIX_ret,
      .direction = "down"
    ) %>%
    # publication lags to avoid look ahead
    mutate(
      GDP_g             = lag(GDP_g, 4),  # 4 month lag for quarterly GDP
      CPI_inflation     = 100 * (log(CPI) - log(lag(CPI))),  # monthly inflation in percent
      CPI_inflation     = lag(CPI_inflation, 1),
      FedFunds_lag1     = lag(FedFunds, 1),
      Unemployment_lag1 = lag(Unemployment, 1),
      Term_Spread_lag1  = lag(Term_Spread, 1),
      BAA_Yield_lag1    = lag(BAA_Yield, 1),
      VIX_ret_lag1      = lag(VIX_ret, 1),
      lag_return        = lag(monthly_return)
    )

  # Build predicting_date and predicting_return
  Total_data <- Total_data %>%
    mutate(
      predicting_date   = lead(date),
      predicting_return = lead(monthly_return)
    ) %>%
    select(
      predicting_date,
      predicting_return,
      monthly_return,
      lag_return,
      GDP_g,
      Personal_Savings_Rate,
      Unemployment,
      CPI_inflation,
      FedFunds_lag1,
      Term_Spread_lag1,
      BAA_Yield_lag1,
      VIX_ret_lag1
    ) %>%
    filter(
      predicting_date >= as.Date("1985-01-01"),
      predicting_date <= as.Date("2024-12-01")
    )

  # Final NA cleanup: for any edge NAs, fill forward then back
  Total_data <- Total_data %>%
    arrange(predicting_date) %>%
    tidyr::fill(-predicting_date, .direction = "down") %>%
    tidyr::fill(-predicting_date, .direction = "up")

  .check_480(Total_data)

  # Tag each observation with the project split:
  # 40% training (1985–2000), 20% validation (2001–2008), 40% testing (2009–2024)
  Total_data <- Total_data %>%
    arrange(predicting_date) %>%
    dplyr::mutate(
      sample_set = dplyr::case_when(
        predicting_date < as.Date("2001-01-01") ~ "train",       # 1985–2000
        predicting_date < as.Date("2009-01-01") ~ "validation",  # 2001–2008
        TRUE                                     ~ "test"        # 2009–2024
      )
    )

  # Save processed data for Python EZ PPO
  processed_dir <- file.path(DATA_DIR, "processed")
  if (!dir.exists(processed_dir)) {
    dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
  }

  csv_path <- file.path(processed_dir, "Total_data_for_python.csv")
  readr::write_csv(Total_data, csv_path)
  message("R/data_preprocessing.R: wrote ", csv_path,
          " with ", nrow(Total_data), " rows and ", ncol(Total_data), " columns.")

  # Save for convenience (optional)
  saveRDS(Total_data, file.path(DATA_DIR, "Total_data.rds"))

  return(Total_data)
}
