library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(purrr)

cells <- tibble(p = c(20,40,20,40), sigma = c(1,1,3,3))
add_block <- function(scenario, n, method, means, sds) {
  stopifnot(length(means) == 4, length(sds) == 4)
  bind_cols(
    tibble(Scenario = scenario, n = n, method = method),
    cells,
    tibble(mean_logmse = means, sd_logmse = sds)
  )
}

# n = 1000
dat_I_1000 <- bind_rows(
  add_block("I", 1000, "Semi–lasso",        c( 0.252,  0.292,  0.534,  0.824), c(0.119, 0.130, 0.187, 0.202)),
  add_block("I", 1000, "Revised–lasso",     c( 0.251,  0.292,  0.532,  0.822), c(0.119, 0.130, 0.189, 0.202)),
  add_block("I", 1000, "Semi–DNN",          c(-0.550, -0.181,  0.355,  0.628), c(0.249, 0.207, 0.237, 0.183)),
  add_block("I", 1000, "Weighted–deepTL",   c(-0.711, -0.402,  0.333,  0.607), c(0.278, 0.246, 0.246, 0.185)),
  add_block("I", 1000, "Semi–XGBoost",      c( 0.294,  0.376,  1.019,  1.121), c(0.150, 0.145, 0.169, 0.168)),
  add_block("I", 1000, "Revised–XGBoost",   c( 0.283,  0.370,  1.014,  1.118), c(0.144, 0.133, 0.166, 0.160)),
  add_block("I", 1000, "Semi–kern",         c( 0.104,  0.458,  0.677,  0.958), c(0.121, 0.118, 0.187, 0.176)),
  add_block("I", 1000, "Revised–kern",      c( 0.046,  0.448,  0.669,  0.954), c(0.123, 0.121, 0.192, 0.171))
)

# n = 2000
dat_I_2000 <- bind_rows(
  add_block("I", 2000, "Semi–lasso",        c( 0.152,  0.184,  0.319,  0.387), c(0.074, 0.077, 0.118, 0.132)),
  add_block("I", 2000, "Revised–lasso",     c( 0.152,  0.183,  0.319,  0.387), c(0.074, 0.078, 0.117, 0.132)),
  add_block("I", 2000, "Semi–DNN",          c(-0.847, -0.769,  0.029,  0.154), c(0.201, 0.196, 0.243, 0.228)),
  add_block("I", 2000, "Weighted–deepTL",   c(-0.919, -0.874,  0.001,  0.113), c(0.208, 0.211, 0.247, 0.228)),
  add_block("I", 2000, "Semi–XGBoost",      c(-0.069,  0.006,  0.694,  0.790), c(0.131, 0.124, 0.155, 0.151)),
  add_block("I", 2000, "Revised–XGBoost",   c(-0.088, -0.018,  0.679,  0.776), c(0.123, 0.125, 0.150, 0.147)),
  add_block("I", 2000, "Semi–kern",         c(-0.382,  0.183,  0.356,  0.668), c(0.114, 0.094, 0.132, 0.133)),
  add_block("I", 2000, "Revised–kern",      c(-0.436,  0.150,  0.342,  0.676), c(0.103, 0.095, 0.138, 0.137))
)

# -------------------------------
# Enter DATA: Scenario II
# -------------------------------

# n = 1000
dat_II_1000 <- bind_rows(
  add_block("II", 1000, "Semi–lasso",       c( 0.410,  0.423,  0.459,  0.455), c(0.337, 0.355, 0.358, 0.351)),
  add_block("II", 1000, "Revised–lasso",    c( 0.410,  0.424,  0.463,  0.452), c(0.337, 0.352, 0.358, 0.352)),
  add_block("II", 1000, "Semi–DNN",         c(-0.010,  0.353,  0.382,  0.401), c(0.465, 0.379, 0.358, 0.345)),
  add_block("II", 1000, "Weighted–deepTL",  c(-0.068,  0.241,  0.306,  0.337), c(0.481, 0.414, 0.357, 0.345)),
  add_block("II", 1000, "Semi–XGBoost",     c( 0.389,  0.468,  0.817,  0.841), c(0.375, 0.365, 0.368, 0.353)),
  add_block("II", 1000, "Revised–XGBoost",  c( 0.319,  0.406,  0.546,  0.551), c(0.369, 0.372, 0.377, 0.368)),
  add_block("II", 1000, "Semi–kern",        c( 0.410,  0.439,  0.488,  0.460), c(0.356, 0.360, 0.371, 0.357)),
  add_block("II", 1000, "Revised–kern",     c( 0.272,  0.412,  0.446,  0.434), c(0.372, 0.357, 0.358, 0.349))
)

# n = 2000
dat_II_2000 <- bind_rows(
  add_block("II", 2000, "Semi–lasso",       c( 0.413,  0.391,  0.405,  0.383), c(0.337, 0.327, 0.325, 0.308)),
  add_block("II", 2000, "Revised–lasso",    c( 0.413,  0.392,  0.406,  0.382), c(0.337, 0.327, 0.326, 0.309)),
  add_block("II", 2000, "Semi–DNN",         c(-0.269, -0.189,  0.306,  0.309), c(0.568, 0.509, 0.334, 0.308)),
  add_block("II", 2000, "Weighted–deepTL",  c(-0.292, -0.218,  0.265,  0.269), c(0.575, 0.520, 0.333, 0.309)),
  add_block("II", 2000, "Semi–XGBoost",     c( 0.193,  0.299,  0.623,  0.550), c(0.428, 0.373, 0.351, 0.338)),
  add_block("II", 2000, "Revised–XGBoost",  c( 0.156,  0.285,  0.460,  0.409), c(0.422, 0.373, 0.348, 0.332)),
  add_block("II", 2000, "Semi–kern",        c( 0.081,  0.403,  0.427,  0.405), c(0.435, 0.328, 0.333, 0.321)),
  add_block("II", 2000, "Revised–kern",     c(-0.027,  0.367,  0.388,  0.377), c(0.467, 0.331, 0.326, 0.310))
)

df <- bind_rows(dat_I_1000, dat_I_2000, dat_II_1000, dat_II_2000) %>%
  mutate(
    Scenario = factor(Scenario, levels = c("I","II")),
    cluster  = factor(paste0("σ=", sigma, ", p=", p),
                      levels = c("σ=1, p=20","σ=1, p=40","σ=3, p=20","σ=3, p=40"))
  )

method_order <- df %>%
  group_by(Scenario, n, method) %>%
  summarise(overall_mean = mean(mean_logmse), .groups = "drop") %>%
  group_by(Scenario, n) %>%
  arrange(overall_mean, .by_group = TRUE) %>%
  mutate(order_rank = row_number())

plot_df <- df %>%
  left_join(method_order, by = c("Scenario","n","method")) %>%
  group_by(Scenario, n) %>%
  mutate(method_f = fct_reorder(method, order_rank, .desc = FALSE)) %>% # best first in legend
  ungroup()

method_colors <- c(
  "Weighted–deepTL"  = "#E75480",  # dark pink
  "Semi–DNN"         = "#FFA0C2",  # light pink
  "Semi–lasso"       = "#2E77BB",  # blue
  "Revised–lasso"    = "#6AA9E9",  # light blue
  "Semi–XGBoost"     = "#F28E2B",  # orange
  "Revised–XGBoost"  = "#FFBE7D",  # light orange
  "Semi–kern"        = "#59A14F",  # green
  "Revised–kern"     = "#8CD17D"   # light green
)

label_map <- c(
  "Weighted–deepTL" = "W-deepTL",
  "Semi–DNN"        = "R-deepTL",
  "Semi–lasso"      = "R-lasso",
  "Revised–lasso"   = "W-lasso",
  "Semi–XGBoost"    = "R-XGB",
  "Revised–XGBoost" = "W-XGB",
  "Semi–kern"       = "R-kern",
  "Revised–kern"    = "W-kern"
)

make_barplot <- function(scen, n_val, bar_width = 0.75, dodge_w = 0.85) {
  dat <- filter(plot_df, Scenario == scen, n == n_val)
  ggplot(dat, aes(x = cluster, y = mean_logmse, fill = method_f)) +
    geom_hline(yintercept = 0, linewidth = 0.25, color = "grey88") +
    geom_col(position = position_dodge(width = dodge_w),
             width = bar_width, alpha = 0.95, color = NA) +
    labs(
      x = NULL,
      y = "log mean-squared error"
    ) +
    theme_bw(base_size = 10) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      legend.position    = "bottom"
    )
}

scale_methods <- scale_fill_manual(
  values = method_colors,
  breaks = names(method_colors),
  labels = label_map,
  name   = "Method"
)

p_I_1000 <- make_barplot("I", 1000) + scale_methods +
  theme(
    legend.position = "none",
    axis.title.y    = element_blank(),   # we'll add one shared label later
    axis.text.y     = element_text(size = 10),
    axis.ticks.y    = element_line(),
    axis.title.x    = element_blank(),
    axis.text.x     = element_blank(),
    axis.ticks.x    = element_blank()
  )


p_I_2000 <- make_barplot("I", 2000) + scale_methods +
  theme(
    legend.position = "none",
    axis.title.y    = element_blank(),
    axis.text.y     = element_text(size = 10),
    axis.ticks.y    = element_line()
  )

p_I_1000_leg <- make_barplot("I", 1000) + scale_methods +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 8),
    legend.key.size = grid::unit(6, "pt")
  ) +
  guides(fill = guide_legend(
    ncol = 1, byrow = TRUE, title.position = "top",
    keyheight = grid::unit(15, "pt"),
    keywidth  = grid::unit(5, "pt")
  ))


leg_I <- cowplot::get_legend(p_I_1000_leg)

legend_panel <- cowplot::ggdraw() +
  cowplot::draw_plot(leg_I, x = 1, y = 0.5, width = 1, height = 1,
                     hjust = 1, vjust = 0.5)

stack_I <- (p_I_1000 / p_I_2000) + patchwork::plot_layout(heights = c(1, 1))

fig_I <- cowplot::plot_grid(
  stack_I, legend_panel,
  ncol = 2, rel_widths = c(1, 0.14), align = "h"
)

fig_I <- cowplot::ggdraw(fig_I) +
  cowplot::draw_label("log mse",
                      x = 0.02, y = 0.5, angle = 90, size = 10,
                      vjust = 0.5, hjust = 0.5)


p_II_1000 <- make_barplot("II", 1000) + scale_methods +
  theme(
    legend.position = "none",
    axis.title.y    = element_blank(),   
    axis.text.y     = element_text(size = 10),
    axis.ticks.y    = element_line(),
    axis.title.x    = element_blank(),
    axis.text.x     = element_blank(),
    axis.ticks.x    = element_blank()
  )


p_II_2000 <- make_barplot("II", 2000) + scale_methods +
  theme(
    legend.position = "none",
    axis.title.y    = element_blank(),
    axis.text.y     = element_text(size = 10),
    axis.ticks.y    = element_line()
  )

p_II_1000_leg <- make_barplot("II", 1000) + scale_methods +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 8),
    legend.key.size = grid::unit(6, "pt")
  ) +
  guides(fill = guide_legend(
    ncol = 1, byrow = TRUE, title.position = "top",
    keyheight = grid::unit(15, "pt"),
    keywidth  = grid::unit(5, "pt")
  ))


leg_II <- cowplot::get_legend(p_II_1000_leg)

legend_panel <- cowplot::ggdraw() +
  cowplot::draw_plot(leg_II, x = 1, y = 0.5, width = 1, height = 1,
                     hjust = 1, vjust = 0.5)

stack_II <- (p_II_1000 / p_II_2000) + patchwork::plot_layout(heights = c(1, 1))

fig_II <- cowplot::plot_grid(
  stack_II, legend_panel,
  ncol = 2, rel_widths = c(1, 0.14), align = "h"
)

fig_II <- cowplot::ggdraw(fig_II) +
  cowplot::draw_label("log mse",
                      x = 0.02, y = 0.5, angle = 90, size = 10,
                      vjust = 0.5, hjust = 0.5)
