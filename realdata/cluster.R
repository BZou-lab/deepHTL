library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(cluster)
library(scales)

res_dnn_oof <- readRDS("unos_hte_eif_dnn_oof.rds")
load("TLD_Selected.RData")
keep <- with(TLD, trimws(as.character(DIABETES_DON)) %in% c("0","1") &
               trimws(as.character(CMV_DON)) %in% c("0","1"))
TLD <- TLD[keep, , drop = FALSE]

tau_hat <- if (is.numeric(res_dnn_oof$tau)) {
  as.numeric(res_dnn_oof$tau)
} else if (!is.null(res_dnn_oof$tau$revised)) {
  as.numeric(res_dnn_oof$tau$revised)
} else {
  as.numeric(tau_hat)
}

df_tau <- tibble(
  tau = tau_hat,
  LAS = as.numeric(TLD$CALC_LAS_LISTDATE),
  IschemicTime = as.numeric(TLD$ISCHTIME),
  SixMinWalk = as.numeric(TLD$SIX_MIN_WALK),
  BMI = as.numeric(TLD$BMI_CALC),
  RecipientAge = as.numeric(TLD$AGE),
  DonorAge = as.numeric(TLD$AGE_DON),
  DonorCMV = as.numeric(TLD$CMV_DON)
)

Xclust <- df_tau %>%
  select(tau, LAS, IschemicTime, SixMinWalk, BMI, RecipientAge, DonorAge, DonorCMV) %>%
  mutate(across(everything(), ~replace(., is.infinite(.) | is.nan(.), NA_real_)))
Xclust <- Xclust %>% mutate(across(everything(), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
Xsc <- scale(Xclust) %>% as.matrix()

set.seed(123)
km_final <- kmeans(Xsc, centers = 5, nstart = 50, iter.max = 200)
df_tau$cluster <- factor(km_final$cluster, labels = paste0("C", 1:5))

clust_summary <- df_tau %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    tau_mean = mean(tau), tau_sd = sd(tau),
    across(c(LAS, IschemicTime, SixMinWalk, BMI, RecipientAge, DonorAge, DonorCMV),
           list(mean = ~mean(.), sd = ~sd(.)), .names = "{.col}_{.fn}")
  ) %>% ungroup()

write.csv(clust_summary, "cluster_feature_summary.csv", row.names = FALSE)

p_tau_clust <- df_tau %>%
  group_by(cluster) %>%
  summarise(
    tau_mean = mean(tau), tau_sd = sd(tau), n = n(),
    lo = tau_mean - 1.96 * tau_sd / sqrt(n),
    hi = tau_mean + 1.96 * tau_sd / sqrt(n)
  ) %>%
  ggplot(aes(x = cluster, y = tau_mean)) +
  geom_hline(yintercept = res_dnn_oof$ATE$est, linetype = "dashed") +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.15, color = "grey40") +
  geom_point(size = 3) +
  labs(x = "Cluster", y = "Mean τ̂(X)", title = "Average individualized effect by cluster") +
  theme_bw(base_size = 10)

pca <- prcomp(Xsc, center = FALSE, scale. = FALSE)
p_pca <- data.frame(pca$x[, 1:2], cluster = df_tau$cluster) %>%
  ggplot(aes(PC1, PC2, color = cluster)) +
  geom_point(alpha = 0.6, size = 1) +
  theme_bw(base_size = 10)

cl_means <- df_tau %>%
  group_by(cluster) %>%
  summarise(across(c(tau, LAS, IschemicTime, SixMinWalk, BMI, RecipientAge, DonorAge, DonorCMV), mean)) %>%
  ungroup()
clm_long <- cl_means %>%
  pivot_longer(-cluster, names_to = "Feature", values_to = "Mean") %>%
  group_by(Feature) %>%
  mutate(Mean = scale(Mean)[,1]) %>%
  ungroup() %>%
mutate(Feature = ifelse(Feature == "tau", "hat(tau)(X)", Feature))

p_hm <- ggplot(clm_long, aes(x = Feature, y = cluster, fill = Mean)) +
  geom_tile() +
  scale_fill_gradient2(low = "steelblue", high = "firebrick", mid = "white", midpoint = 0) +
  scale_x_discrete(labels = label_parse()) +
  labs(x = "Feature", y = "Cluster", fill = "Mean (z-score)") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

write.csv(clust_summary, "clust_summary.csv")
ggsave("Cluster.png", p_hm, width = 10, height = 8, dpi = 300)
