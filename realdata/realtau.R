library(data.table)
library(car)
library(ggplot2)

all_tau <- load("realtau.RData")
tau_long <- list()

method_names <- c("DeepTL", "R-learner", "T-learner", "X-learner")

for (i in seq_along(all_tau)) {
  for (m in seq_along(all_tau[[i]])) {
    temp_df <- data.table(
      Bootstrap_ID = i,
      Method = method_names[m],
      Obs_ID = 1:14306, 
      tau_est = unlist(all_tau[[i]][[m]])
    )
    tau_long[[length(tau_long) + 1]] <- temp_df
  }
}

tau_df <- rbindlist(tau_long)
tau_summary <- tau_df[, .(
  tau_mean = mean(tau_est),
  tau_var = var(tau_est),
  tau_sd = sd(tau_est),
  tau_CI_lower = mean(tau_est) - 1.96 * (sd(tau_est) / sqrt(.N)),
  tau_CI_upper = mean(tau_est) + 1.96 * (sd(tau_est) / sqrt(.N))
), by = .(Obs_ID, Method)]

head(tau_summary)

ggplot(tau_summary, aes(x = tau_mean)) +
  geom_histogram(bins = 50, fill = "blue", alpha = 0.5, color = "black") +
  facet_wrap(~ Method, scales = "free") + 
  theme_minimal() +
  labs(title = "τ(X) Mean",
       x = "τ(X) Mean", y = "Count")

ggplot(tau_summary, aes(x = tau_var)) +
  geom_histogram(bins = 50, fill = "red", alpha = 0.5, color = "black") +
  facet_wrap(~ Method, scales = "free") +
  theme_minimal() +
  labs(title = "τ(X) Variance",
       x = "Variance of τ(X)", y = "Count")
