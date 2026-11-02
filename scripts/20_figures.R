suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
})

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
root <- Sys.getenv("GLOF_ROOT", unset = normalizePath(file.path(script_dir, "..")))
models <- file.path(root, "models")
out_dir <- file.path(root, "paper", "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

c_glof <- "#c1272d"
c_non  <- "#3b6ea5"
c_gray <- "#5c6670"
c_light <- "#8fa8c9"
c_brown <- "#7a5c3e"

scheme_cols <- c(random = "#8C8C8C", group = "#0072B2", knndm = "#D55E00", region = "#E69F00",
                 forward = "#009E73")

FIG_W <- 6.85

theme_pub <- theme_minimal(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.25, colour = "grey86"),
    axis.line = element_line(linewidth = 0.3, colour = "grey30"),
    axis.ticks = element_line(linewidth = 0.3, colour = "grey30"),
    plot.title = element_text(face = "bold", size = 9.5, hjust = 0),
    plot.subtitle = element_text(size = 8, colour = "grey35", hjust = 0),
    axis.text = element_text(size = 8, colour = "grey25"),
    axis.title = element_text(size = 8.5),
    legend.text = element_text(size = 8),
    plot.margin = margin(4, 6, 4, 4),
    legend.position = "inside",
    legend.position.inside = c(0.98, 0.98),
    legend.justification = c(1, 1),
    legend.background = element_blank(),
    legend.key.size = unit(0.75, "lines"),
    legend.title = element_blank()
  )

theme_bottom <- theme(legend.position = "bottom", legend.justification = "center")

save_fig <- function(name, p, width, height) {
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = width, height = height,
         dpi = 600, bg = "white")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = width, height = height,
         device = if (capabilities("cairo")) cairo_pdf else "pdf", bg = "white")
  cat("  saved", name, "(png 600 dpi + pdf)\n")
}

read_opt <- function(name) {
  f <- file.path(models, name)
  if (!file.exists(f)) {
    cat("  [skip] ", name, " not found\n", sep = "")
    return(NULL)
  }
  d <- tryCatch(read.csv(f), error = function(e) NULL)
  if (is.null(d) || !nrow(d)) return(NULL)
  d
}

region_label <- c(
  cordillera_blanca = "C. Blanca", cordillera_huayhuash = "C. Huayhuash",
  cordillera_raura = "C. Raura", cordillera_central = "C. Central",
  cordillera_urubamba = "C. Urubamba", cordillera_vilcanota = "C. Vilcanota",
  cordillera_huanzo = "C. Huanzo", carabaya = "C. Carabaya",
  apolobamba = "Apolobamba", bolivia_cordillera_real = "C. Real",
  bolivia_norte = "Bolivia N.", ecuador_antisana = "Antisana",
  chile_andes_centrales = "Central Andes", patagonia_npi = "N. Patagonian Icefield",
  patagonia_norte = "Patagonia N.", patagonia_sur = "Patagonia S.")

nice_region <- function(x) {
  lab <- unname(region_label[x])
  ifelse(is.na(lab), gsub("_", " ", x), lab)
}

fcfg <- fromJSON(file.path(models, "feature_sets.json"))
run <- fromJSON(file.path(models, "run_summary.json"))
stats <- fromJSON(file.path(models, "paper_stats.json"))
oof <- read.csv(file.path(models, "oof_scores.csv"))
perf <- read_opt("performance_by_scheme.csv")

primary_model <- run$primary_model
primary_set <- run$primary_feature_set
headline <- stats$headline_scheme

scheme_label <- c(random = "Random (stratified)", group = "Sub-basin grouped",
                  knndm = "kNNDM", region = "Leave-one-region-out",
                  forward = sprintf("Forward (cut-off %d)", fcfg$forward_year))

y <- oof$glof
s <- oof$oof_score
t_youden <- stats$youden_threshold_raw_score
src <- oof[oof$glof == 1, ]

lab_pos <- sprintf("GLOF-source lakes (n = %s)", format(sum(y), big.mark = ","))
lab_neg <- sprintf("Non-source lakes (n = %s)", format(sum(y == 0), big.mark = ","))

p3a <- ggplot(oof[oof$glof == 0, ], aes(x = oof_score)) +
  geom_histogram(aes(y = after_stat(density), fill = lab_neg),
                 bins = 45, alpha = 0.75, colour = NA, na.rm = TRUE) +
  geom_segment(data = src, aes(x = oof_score, xend = oof_score, y = -0.28, yend = -0.05,
                               colour = lab_pos), linewidth = 0.55) +
  geom_vline(xintercept = t_youden, linetype = "dashed", linewidth = 0.5) +
  annotate("text", x = t_youden + 0.015, y = Inf, vjust = 1.6, hjust = 0,
           label = sprintf("Youden t = %.2f", t_youden), size = 3) +
  scale_fill_manual(values = setNames(c_non, lab_neg)) +
  scale_colour_manual(values = setNames(c_glof, lab_pos)) +
  scale_x_continuous(limits = c(0, 1), expand = expansion(mult = 0.01)) +
  labs(x = sprintf("Susceptibility score (%s out-of-fold)", scheme_label[[headline]]),
       y = "Density", title = "(a) Score distribution by class") +
  guides(fill = guide_legend(order = 1), colour = guide_legend(order = 2)) +
  theme_pub + theme_bottom

ord <- order(-s)
gain <- data.frame(frac = seq_along(y) / length(y) * 100,
                   recall = cumsum(y[ord]) / sum(y) * 100)
marks <- do.call(rbind, lapply(c(5, 10, 20, 50), function(p) {
  k <- max(1, round(length(y) * p / 100))
  data.frame(frac = p, recall = gain$recall[k])
}))

p3b <- ggplot(gain, aes(frac, recall)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = c_gray, linewidth = 0.45) +
  geom_step(colour = c_glof, linewidth = 0.8) +
  geom_point(data = marks, colour = c_glof, size = 2) +
  ggrepel::geom_text_repel(data = marks, aes(label = sprintf("%.0f%%", recall)),
                           size = 2.85, nudge_y = -7, direction = "x", segment.size = 0.25,
                           min.segment.length = 0, seed = 42) +
  scale_x_continuous(limits = c(0, 100), expand = expansion(mult = 0.01)) +
  scale_y_continuous(limits = c(0, 102), expand = expansion(mult = 0.01)) +
  labs(x = "Fraction of inventory inspected (%, ranked by score)",
       y = "GLOF-source lakes recovered (%)", title = "(b) Cumulative gain") +
  annotate("text", x = 84, y = 76, label = "Random", colour = c_gray, size = 3) +
  theme_pub

save_fig("FigS1_score_distribution", p3a + p3b, FIG_W, 3.4)

weighted_roc <- function(yy, ss, ww, grid) {
  o <- order(-ss)
  yy <- yy[o]; ww <- ww[o]
  tp <- cumsum(ww * yy) / sum(ww * yy)
  fp <- cumsum(ww * (1 - yy)) / sum(ww * (1 - yy))
  approx(c(0, fp), c(0, tp), xout = grid, ties = max, rule = 2)$y
}

roc_band <- function(yy, ss, gg, n_boot = 400, seed = 42) {
  ok <- is.finite(ss)
  yy <- yy[ok]; ss <- ss[ok]; gg <- as.character(gg[ok])
  grid <- seq(0, 1, length.out = 201)
  ug <- unique(gg)
  idx <- match(gg, ug)
  set.seed(seed)
  curves <- replicate(n_boot, {
    w <- tabulate(sample.int(length(ug), replace = TRUE), nbins = length(ug))[idx]
    if (sum(w * yy) == 0) rep(NA_real_, length(grid)) else weighted_roc(yy, ss, w, grid)
  })
  data.frame(fpr = grid, tpr = weighted_roc(yy, ss, rep(1, length(yy)), grid),
             lo = apply(curves, 1, quantile, 0.025, na.rm = TRUE),
             hi = apply(curves, 1, quantile, 0.975, na.rm = TRUE))
}

auc_of <- function(yy, ss) {
  ok <- is.finite(ss)
  r <- rank(ss[ok]); n1 <- sum(yy[ok]); n0 <- sum(!yy[ok])
  (sum(r[yy[ok] == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

if (!is.null(perf)) {
  prim <- perf[perf$cohort == "primary", ]
  curves <- list(knndm = roc_band(y, oof$oof_knndm, oof$group))
  curve_lab <- c(knndm = sprintf("kNNDM, AUC %.2f", auc_of(y, oof$oof_knndm)))
  if ("oof_region" %in% names(oof)) {
    curves$region <- roc_band(y, oof$oof_region, oof$group)
    curve_lab["region"] <- sprintf("Region held out, AUC %.2f", auc_of(y, oof$oof_region))
  }
  fw <- read_opt("forward_scores_primary.csv")
  if (!is.null(fw)) {
    curves$forward <- roc_band(fw$label, fw$score, fw$fold_group)
    curve_lab["forward"] <- sprintf("Forward 2011-2025, AUC %.2f", auc_of(fw$label, fw$score))
  }
  rc <- do.call(rbind, lapply(names(curves), function(k) cbind(curves[[k]], scheme = k)))
  rc$scheme <- factor(rc$scheme, levels = names(curves))
  ends <- do.call(rbind, lapply(names(curves), function(k) {
    d <- curves[[k]]
    i <- which.min(abs(d$fpr - c(knndm = 0.16, region = 0.40, forward = 0.62)[[k]]))
    data.frame(fpr = d$fpr[i], tpr = d$tpr[i], scheme = k, label = curve_lab[[k]])
  }))
  ops <- do.call(rbind, lapply(c(5, 10, 20), function(p) {
    k <- max(1, round(length(y) * p / 100))
    o <- order(-oof$oof_knndm)
    data.frame(fpr = sum(1 - y[o][1:k]) / sum(y == 0), tpr = sum(y[o][1:k]) / sum(y),
               label = sprintf("top %d%%: %d of %d", p, sum(y[o][1:k]), sum(y)))
  }))
  p4a <- ggplot(rc, aes(fpr, tpr, colour = scheme, fill = scheme)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60",
                linewidth = 0.4) +
    geom_ribbon(data = rc[rc$scheme == "knndm", ], aes(ymin = lo, ymax = hi),
                colour = NA, alpha = 0.18) +
    geom_line(linewidth = 0.8) +
    geom_point(data = ops, aes(fpr, tpr), inherit.aes = FALSE, shape = 21, size = 2.6,
               fill = "white", colour = scheme_cols[["knndm"]], stroke = 0.9) +
    geom_text(data = ops, aes(fpr, tpr, label = sub(":.*", "", label)), inherit.aes = FALSE,
              size = 2.85, hjust = -0.35, vjust = 1.3, colour = "grey20") +
    scale_colour_manual(values = scheme_cols, labels = curve_lab, name = NULL) +
    scale_fill_manual(values = scheme_cols, guide = "none") +
    coord_equal(expand = FALSE) +
    labs(x = "False positive rate", y = "True positive rate", title = "(a) ROC curves",
         subtitle = sprintf("Circles: top 5, 10, 20%% hold %s of %d sources",
                            paste(sub(".*: (\\d+) of.*", "\\1", ops$label), collapse = ", "), sum(y))) +
    theme_pub + theme(legend.position.inside = c(0.98, 0.03), legend.justification = c(1, 0),
                      legend.key.size = unit(0.6, "lines"),
                      legend.background = element_rect(fill = alpha("white", 0.9), colour = NA))

  runs <- read_opt("cv_runs.csv")
  pb <- list(p4a)
  if (!is.null(runs) && requireNamespace("ggdist", quietly = TRUE)) {
    rr <- runs[runs$cohort == "primary" & runs$scheme == headline, ]
    med <- tapply(rr$auc, rr$model, median)
    rr$model <- factor(rr$model, levels = names(sort(med)))
    rr$is_primary <- rr$model == primary_model
    p4b <- ggplot(rr, aes(auc, model, fill = is_primary)) +
      ggdist::stat_halfeye(adjust = 0.9, width = 0.65, .width = c(0.5, 0.95),
                           point_size = 1.8, interval_colour = "grey15",
                           point_colour = "grey15", slab_alpha = 0.85) +
      geom_point(aes(colour = is_primary), position = position_nudge(y = -0.12),
                 shape = "|", size = 2.4, alpha = 0.6) +
      geom_text(data = data.frame(model = names(med), auc = med,
                                  x = tapply(rr$auc, rr$model, max)[names(med)]),
                aes(x, model, label = sprintf("%.3f", auc)), inherit.aes = FALSE,
                hjust = -0.25, size = 2.85) +
      scale_x_continuous(expand = expansion(mult = c(0.03, 0.22))) +
      scale_fill_manual(values = c(`TRUE` = scheme_cols[["knndm"]], `FALSE` = "grey75"),
                        guide = "none") +
      scale_colour_manual(values = c(`TRUE` = scheme_cols[["knndm"]], `FALSE` = "grey45"),
                          guide = "none") +
      scale_y_discrete(labels = c(BalancedRF = "Balanced RF", EasyEnsemble = "Easy Ensemble",
                                  LogisticRegression = "Logistic (L2)", RandomForest = "Random forest",
                                  XGBoost = "XGBoost", LightGBM = "LightGBM")) +
      labs(x = sprintf("AUC over %d kNNDM repetitions", max(table(rr$model))), y = NULL,
           title = "(b) Spread across repetitions",
           subtitle = "Median; bars: 50 and 95% intervals") +
      theme_pub + theme(panel.grid.major.y = element_blank())
    pb <- c(pb, list(p4b))
  }

  tests <- read_opt("model_tests.csv")
  if (!is.null(tests)) {
    tests$half <- qt(0.975, pmax(tests$df, 1)) * tests$corrected_se
    lev <- if (length(pb) > 1) levels(pb[[2]]$data$model) else unique(tests$model_b)
    tests$model_b <- factor(tests$model_b, levels = intersect(lev, unique(tests$model_b)))
    tests$sig <- !is.na(tests$p_value) & tests$p_value < 0.05
    tests$plab <- ifelse(tests$p_value < 0.001, "p<0.001", sprintf("p=%.3f", tests$p_value))
    p4c <- ggplot(tests, aes(mean_diff, model_b, colour = scheme)) +
      annotate("rect", xmin = -Inf, xmax = 0, ymin = -Inf, ymax = Inf, fill = "grey95") +
      geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.4) +
      geom_errorbar(aes(xmin = mean_diff - half, xmax = mean_diff + half), width = 0,
                    linewidth = 0.6, orientation = "y", position = position_dodge(width = 0.55)) +
      geom_point(aes(shape = sig), size = 2.5, fill = "white",
                 position = position_dodge(width = 0.55)) +
      geom_text(aes(x = mean_diff + half, label = plab), hjust = -0.15, size = 2.85,
                position = position_dodge(width = 0.55), show.legend = FALSE) +
      scale_colour_manual(values = scheme_cols, labels = scheme_label, name = NULL) +
      scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
      scale_x_continuous(expand = expansion(mult = c(0.05, 0.3))) +
      scale_y_discrete(labels = c(EasyEnsemble = "Easy Ensemble", LogisticRegression = "Logistic (L2)",
                                  RandomForest = "Random forest", XGBoost = "XGBoost",
                                  LightGBM = "LightGBM")) +
      labs(x = "AUC gain of Balanced RF (corrected 95% CI)", y = NULL,
           title = "(c) Paired comparison",
           subtitle = "Filled: p < 0.05; open: not significant") +
      theme_pub + theme_bottom + theme(panel.grid.major.y = element_blank())
    pb <- c(pb, list(p4c))
  }
  lay <- if (length(pb) == 3) (pb[[1]] | pb[[2]]) / pb[[3]] + plot_layout(heights = c(1.15, 1))
         else wrap_plots(pb, nrow = 1)
  save_fig("Fig04_model_performance", lay, FIG_W, 7.2)
}

feat_label <- c(
  area_m2 = "Area", perimeter_m = "Perimeter", compactness = "Compactness",
  elongation = "Elongation", fractal_dim = "Shoreline fractal dim.",
  convexity = "Convexity", shore_dev_index = "Shoreline development",
  lake_elev_m = "Lake elevation", freeboard_m = "Dam freeboard",
  outlet_drop_m = "Outlet drop", ring_slope_mean = "Surrounding slope (mean)",
  ring_slope_p90 = "Surrounding slope (p90)", ring_relief_m = "Surrounding relief",
  ring_aspect_north = "Surrounding northness", ring_aspect_east = "Surrounding eastness",
  ring_tri_mean = "Terrain ruggedness (TRI)", ring_curvature_mean = "Surrounding curvature",
  ring_roughness_mean = "Surrounding roughness", ring_vrm_mean = "Terrain ruggedness (VRM)",
  dist_glacier_m = "Distance to glacier", area_trend = "Pre-cutoff area trend",
  area_cv = "Pre-cutoff area CV", presence_rate = "Pre-cutoff presence rate",
  area_last_m2 = "Last pre-cutoff area", years_since_first_seen = "Years since first seen")

nice_feature <- function(x) {
  lab <- unname(feat_label[x])
  ifelse(is.na(lab), x, lab)
}

sh <- read_opt("shap_values_long.csv")
if (!is.null(sh)) {
  sh$flab <- nice_feature(sh$feature)
  ordf <- unique(sh$flab[order(-sh$mean_abs)])
  sh$flab <- factor(sh$flab, levels = rev(ordf))
  set.seed(42)
  p5 <- ggplot(sh, aes(shap, flab, colour = value_scaled)) +
    geom_vline(xintercept = 0, colour = "grey55", linewidth = 0.4) +
    geom_jitter(height = 0.3, width = 0, size = 0.6, alpha = 0.55) +
    scale_colour_gradient(low = c_non, high = c_glof, breaks = c(0.02, 0.98),
                          labels = c("Low", "High"), name = "Feature\nvalue") +
    labs(x = "SHAP value (impact on the balanced random forest score)", y = NULL) +
    theme_pub +
    theme(legend.position = "right", legend.title = element_text(size = 9),
          panel.grid.major.y = element_line(linewidth = 0.2, colour = "grey90"))
  save_fig("Fig08_shap", p5, FIG_W, 5.4)
}

geo <- read_opt("geodist.csv")
if (!is.null(geo)) {
  w <- read_opt("knndm_W.csv")
  geo_lab <- c(`prediction-to-sample` = "Prediction to training",
               `sample-to-sample` = "Training to training (LOO)",
               `CV-random` = "CV: random", `CV-group` = "CV: sub-basin grouped",
               `CV-knndm` = "CV: kNNDM (rep 0)")
  geo_col <- c(`prediction-to-sample` = "black", `sample-to-sample` = c_light,
               `CV-random` = c_gray, `CV-group` = c_non, `CV-knndm` = c_glof)
  geo$what <- factor(geo$what, levels = names(geo_lab))
  geo$km <- pmax(geo$distance_m / 1000, 0.01)
  p6 <- ggplot(geo, aes(km, colour = what, linetype = what == "prediction-to-sample")) +
    stat_ecdf(linewidth = 0.75, pad = FALSE) +
    scale_x_log10(breaks = c(0.01, 0.1, 1, 10, 100, 1000),
                  labels = c("0.01", "0.1", "1", "10", "100", "1000")) +
    scale_colour_manual(values = geo_col, labels = geo_lab) +
    scale_linetype_manual(values = c(`TRUE` = "solid", `FALSE` = "22"), guide = "none") +
    labs(x = "Nearest-neighbour distance (km, log scale; distances below 10 m drawn at 10 m)",
         y = "Cumulative proportion") +
    guides(colour = guide_legend(nrow = 2)) +
    theme_pub + theme_bottom
  if (!is.null(w)) {
    wk <- w[grepl("^knndm", w$method), ]
    wl <- c(sprintf("W random = %.0f", w$W[w$method == "random"][1]),
            sprintf("W grouped = %.0f", w$W[w$method == "group"][1]),
            sprintf("W kNNDM = %.0f (median of %d)", median(wk$W), nrow(wk)))
    p6 <- p6 + annotate("label", x = Inf, y = 0.03, hjust = 1.02, vjust = 0, size = 2.85,
                        label = paste(wl, collapse = "\n"), linewidth = 0, fill = "white")
  }
  save_fig("Fig03_nn_distances", p6, FIG_W, 4.3)
}

decay <- read_opt("distance_decay_blocks.csv")
bins <- read_opt("distance_decay_bins.csv")
lc <- read_opt("learning_curve.csv")
if (!is.null(decay)) {
  bl <- decay[decay$scheme == "block" & !is.na(decay$auc), ]
  bm <- do.call(rbind, lapply(split(bl, bl$block_km), function(d)
    data.frame(block_km = d$block_km[1], auc = mean(d$auc), lo = min(d$auc), hi = max(d$auc),
               sep = median(d$median_separation_km))))
  ref <- aggregate(auc ~ scheme, decay[decay$scheme %in% c("random", "group"), ], mean)
  ref$auc <- ifelse(ref$scheme == "group", ref$auc, ref$auc)
  ref_lab <- data.frame(scheme = ref$scheme, auc = ref$auc,
                        label = sprintf("%s %.3f", c(random = "random folds", group = "sub-basin folds")[ref$scheme], ref$auc))
  p7a <- ggplot(bm, aes(block_km, auc)) +
    geom_hline(data = ref, aes(yintercept = auc, colour = scheme), linewidth = 0.45,
               linetype = "22") +
    geom_ribbon(aes(ymin = lo, ymax = hi), fill = scheme_cols[["group"]], alpha = 0.18) +
    geom_line(colour = scheme_cols[["group"]], linewidth = 0.9) +
    geom_point(data = bl, aes(block_km, auc), colour = scheme_cols[["group"]], size = 1.3,
               alpha = 0.5, position = position_jitter(width = 0.015, height = 0, seed = 42)) +
    geom_point(shape = 21, fill = "white", colour = scheme_cols[["group"]], size = 2.8,
               stroke = 1) +
    scale_x_log10(breaks = bm$block_km,
                  labels = sprintf("%g\n(%.0f km)", bm$block_km, bm$sep)) +
    scale_colour_manual(values = scheme_cols, guide = "none") +
    coord_cartesian(ylim = c(0.78, 0.9)) +
    annotate("text", x = 10, y = 0.785, hjust = 0, size = 2.85, colour = "grey30",
             label = sprintf("dashed: random folds %.3f (grey), sub-basin %.3f (blue)",
                             ref$auc[ref$scheme == "random"], ref$auc[ref$scheme == "group"])) +
    labs(x = "Block size in km (median test-training gap)", y = "Pooled AUC",
         title = "(a) Larger blocks", subtitle = "Band: range of 5 grid offsets") +
    theme_pub
  pl <- list(p7a)
  if (!is.null(bins)) {
    bins <- bins[bins$n_predictions > 0 & is.finite(bins$auc), ]
    bins$mid <- ifelse(is.finite(bins$dist_hi_km), sqrt(pmax(bins$dist_lo_km, 0.5) * bins$dist_hi_km),
                       bins$dist_lo_km * 1.5)
    bins$lab <- ifelse(is.finite(bins$dist_hi_km), sprintf("%g-%g", bins$dist_lo_km, bins$dist_hi_km),
                       sprintf(">%g", bins$dist_lo_km))
    n1 <- bins$n_pos
    n0 <- bins$n_predictions - bins$n_pos
    a <- bins$auc
    q1 <- a / (2 - a)
    q2 <- 2 * a^2 / (1 + a)
    se <- sqrt((a * (1 - a) + (n1 - 1) * (q1 - a^2) + (n0 - 1) * (q2 - a^2)) / (n1 * n0))
    bins$lo <- pmax(a - 1.96 * se, 0)
    bins$hi <- pmin(a + 1.96 * se, 1)
    p7b <- ggplot(bins, aes(mid, auc)) +
      geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
      geom_ribbon(aes(ymin = lo, ymax = hi), fill = scheme_cols[["knndm"]], alpha = 0.18) +
      geom_line(colour = scheme_cols[["knndm"]], linewidth = 0.9) +
      geom_point(aes(size = n_pos), shape = 21, fill = "white", colour = scheme_cols[["knndm"]],
                 stroke = 1) +
      scale_x_log10(breaks = bins$mid, labels = bins$lab) +
      scale_size_area(max_size = 3.6, guide = "none") +
      coord_cartesian(ylim = c(0.45, 1)) +
      labs(x = "Distance from test lake to nearest training lake (km)", y = "AUC within distance bin",
           title = "(b) Farther test lakes",
           subtitle = "Band: 95% CI (Hanley-McNeil); point size: source lakes") +
      theme_pub + theme(axis.text.x = element_text(angle = 30, hjust = 1))
    pl <- c(pl, list(p7b))
  }
  if (!is.null(lc)) {
    lc$npos <- ave(lc$n_pos_train_mean, lc$fraction, FUN = mean)
    lcl <- rbind(data.frame(npos = lc$npos, auc = lc$auc, what = "Pooled"),
                 data.frame(npos = lc$npos, auc = lc$auc_within_region, what = "Within region"))
    lcl <- lcl[is.finite(lcl$auc), ]
    lcm <- aggregate(auc ~ npos + what, lcl, median)
    lc_cols <- c(Pooled = scheme_cols[["group"]], `Within region` = c_glof)
    last <- lcm[lcm$npos == max(lcm$npos), ]
    p7c <- ggplot(lcl, aes(npos, auc, colour = what, fill = what)) +
      geom_boxplot(aes(group = interaction(npos, what)), width = 0.9, alpha = 0.25, outlier.size = 0.6,
                   linewidth = 0.35, position = "identity") +
      geom_line(data = lcm, linewidth = 0.9) +
      geom_point(data = lcm, size = 2) +
      geom_text(data = last, aes(x = npos + 0.8, label = what), hjust = 0, size = 2.85,
                show.legend = FALSE) +
      scale_colour_manual(values = lc_cols, guide = "none") +
      scale_fill_manual(values = lc_cols, guide = "none") +
      scale_x_continuous(breaks = round(sort(unique(lcm$npos)), 1),
                         expand = expansion(mult = c(0.04, 0.3))) +
      coord_cartesian(ylim = c(0.5, 0.95)) +
      labs(x = "Source lakes available for training (mean per fold)", y = "AUC (sub-basin folds)",
           title = "(c) More recorded events",
           subtitle = "Boxes: 20 repetitions per training size; line: median") +
      theme_pub
    pl <- c(pl, list(p7c))
  }
  lay <- if (length(pl) == 3) (pl[[1]] | pl[[2]]) / (pl[[3]] | plot_spacer()) + plot_layout(heights = c(1, 1))
         else wrap_plots(pl, nrow = 1)
  if (length(pl) == 3) lay <- (pl[[1]] | pl[[2]]) / pl[[3]]
  save_fig("Fig05_distance_learning", lay, FIG_W, 6.9)
}

if (all(c("oof_knndm", "oof_region") %in% names(oof))) {
  aoa <- read_opt("aoa_by_region.csv")
  regs <- names(which(tapply(oof$glof, oof$analysis_region, sum) >= 3))
  gain_curve <- function(d, col) {
    o <- order(-d[[col]])
    yy <- d$glof[o]
    data.frame(x = c(0, seq_along(yy) / length(yy) * 100), y = c(0, cumsum(yy) / sum(yy) * 100))
  }
  cond <- c(oof_knndm = "(a) Region included in training", oof_region = "(b) Region held out")
  gc <- do.call(rbind, lapply(names(cond), function(col) do.call(rbind, lapply(regs, function(r) {
    d <- oof[oof$analysis_region == r, ]
    g <- gain_curve(d, col)
    g$region <- r
    g$cond <- cond[[col]]
    g$auc <- auc_of(d$glof, d[[col]])
    g
  }))))
  gc$cond <- factor(gc$cond, levels = cond)
  reg_cols <- setNames(c("#0072B2", "#D55E00", "#009E73", "#CC79A7")[seq_along(regs)], regs)
  ends <- do.call(rbind, lapply(split(gc, list(gc$region, gc$cond), drop = TRUE), function(d) {
    k <- which.min(abs(d$x - 30))
    data.frame(x = d$x[k], y = d$y[k], region = d$region[1], cond = d$cond[1],
               label = sprintf("%s %.2f", nice_region(d$region[1]), d$auc[1]))
  }))
  p8 <- ggplot(gc, aes(x, y, colour = region)) +
    annotate("rect", xmin = 0, xmax = 10, ymin = -Inf, ymax = Inf, fill = "#fde0d2", alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey55", linewidth = 0.4) +
    geom_step(linewidth = 0.9, direction = "hv") +
    ggrepel::geom_label_repel(data = ends, aes(label = label), size = 2.85, label.size = 0,
                              fill = alpha("white", 0.85), nudge_x = 18, nudge_y = -8,
                              segment.size = 0.3, min.segment.length = 0, seed = 5,
                              direction = "y", show.legend = FALSE) +
    facet_wrap(~ cond, nrow = 1) +
    scale_colour_manual(values = reg_cols, guide = "none") +
    scale_x_continuous(breaks = c(0, 10, 25, 50, 75, 100), expand = expansion(mult = 0.01)) +
    scale_y_continuous(breaks = seq(0, 100, 25), expand = expansion(mult = 0.01)) +
    coord_equal() +
    labs(x = "Lakes of the region inspected, ranked by score (%)",
         y = "GLOF source lakes of the region found (%)") +
    theme_pub +
    theme(strip.text = element_text(size = 9, face = "bold", hjust = 0),
          panel.spacing.x = unit(1.2, "lines"))
  save_fig("Fig06_transfer_ranks", p8, FIG_W, 3.9)
}

co <- read_opt("spatial_autocorrelation.csv")
if (!is.null(co)) {
  co$band <- factor(sprintf("%g-%g", co$band_lo_km, co$band_hi_km),
                    levels = unique(sprintf("%g-%g", co$band_lo_km, co$band_hi_km)[order(co$band_lo_km)]))
  co$sig <- !is.na(co$p_value) & co$p_value < 0.05
  keep_model <- c(glof = "GLOF label", calibrated = "Model score", residual = "Residual (y - p)",
                  pearson_residual = "Pearson residual")
  co <- co[co$kind == "predictor" | co$variable %in% names(keep_model), ]
  co$grp <- ifelse(co$kind == "predictor", "Predictors", "Model")
  co$lab <- ifelse(co$kind == "predictor", nice_feature(co$variable), keep_model[co$variable])
  rng <- tapply(seq_len(nrow(co)), co$lab, function(i) {
    x <- co[i, ]
    x <- x[order(x$band_lo_km), ]
    k <- which(!x$sig | x$morans_I <= 0)
    if (length(k)) k[1] else nrow(x) + 1
  })
  co$lab <- factor(co$lab, levels = names(sort(rng)))
  co$grp <- factor(co$grp, levels = c("Predictors", "Model"))
  lim <- max(abs(co$morans_I), na.rm = TRUE)
  p10 <- ggplot(co, aes(band, lab, fill = morans_I)) +
    geom_tile(colour = "white", linewidth = 0.6) +
    geom_text(aes(label = ifelse(sig & abs(morans_I) >= 0.05, sprintf("%.2f", morans_I), ""),
                  colour = abs(morans_I) > 0.45 * lim), size = 2.85) +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15"), guide = "none") +
    scale_fill_gradientn(colours = colorspace::diverging_hcl(11, "Blue-Red 3"),
                         limits = c(-lim, lim), name = "Moran's I") +
    facet_grid(grp ~ ., scales = "free_y", space = "free_y") +
    labs(x = "Distance band (km)", y = NULL,
         subtitle = "Values printed where p < 0.05 and |I| >= 0.05") +
    theme_minimal(base_size = 9) +
    theme(panel.grid = element_blank(), strip.text.y = element_text(angle = 270, face = "bold", size = 8),
          plot.title = element_text(face = "bold", size = 8),
          plot.subtitle = element_text(colour = "grey35", size = 8),
          axis.text.x = element_text(angle = 30, hjust = 1, size = 8),
          axis.text.y = element_text(size = 8),
          legend.position = "bottom", legend.key.width = unit(1.2, "cm"),
          legend.key.height = unit(0.22, "cm"), legend.title = element_text(size = 8, vjust = 0.9))
  save_fig("FigA1_correlogram", p10, FIG_W, 7)
}

runs <- read_opt("cv_runs.csv")
if (!is.null(perf)) {
  cohort_label <- c(primary = "Primary cohort", all_events_no_dem = "All events, no elevation predictors",
                    unambiguous = "Unambiguous event matches", static_only = "Static predictors only",
                    bias_adjusted = "Access covariates fixed")
  sv <- perf[perf$model == primary_model & perf$cohort %in% names(cohort_label), ]
  npos <- tapply(sv$n_pos[sv$scheme != "forward"], sv$cohort[sv$scheme != "forward"], max)
  sv$cl <- factor(sprintf("%s\n(n+ = %d)", cohort_label[sv$cohort], npos[sv$cohort]),
                  levels = rev(sprintf("%s\n(n+ = %d)", names(cohort_label) |> (\(k) cohort_label[k])(),
                                       npos[names(cohort_label)])))
  sv$sl <- factor(scheme_label[sv$scheme], levels = scheme_label)
  sv$txt <- ifelse(is.na(sv$auc_rep_lo) | sv$auc_rep_lo == sv$auc_rep_hi, sprintf("%.3f", sv$auc),
                   sprintf("%.3f\n[%.3f-%.3f]", sv$auc, sv$auc_rep_lo, sv$auc_rep_hi))
  sv <- sv[!is.na(sv$cl), ]
  p11 <- ggplot(sv, aes(sl, cl, fill = auc)) +
    geom_tile(colour = "white", linewidth = 1.2) +
    geom_text(aes(label = txt, colour = auc > 0.8), size = 2.85, lineheight = 0.9) +
    scale_fill_gradientn(colours = colorspace::sequential_hcl(9, "Blues 3", rev = TRUE),
                         limits = c(0.7, 0.9), oob = scales::squish, name = "AUC") +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15"), guide = "none") +
    scale_x_discrete(position = "top", labels = function(x) gsub(" \\(", "\n(", x)) +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(panel.grid = element_blank(), axis.text = element_text(size = 8, colour = "grey15"),
          legend.position = "bottom", legend.key.width = unit(1.4, "cm"),
          legend.key.height = unit(0.25, "cm"), legend.title = element_text(size = 8.5, vjust = 0.9))
  save_fig("FigA2_sensitivity", p11, FIG_W, 4.6)
}

oof_all <- oof
reg <- aggregate(cbind(n = rep(1, nrow(oof_all)), pos = glof) ~ analysis_region, oof_all, sum)
reg$label <- nice_region(reg$analysis_region)
reg$rate <- 100 * reg$pos / reg$n

r1 <- reg[order(reg$n), ]
r1$label <- factor(r1$label, levels = r1$label)
p2a <- ggplot(r1, aes(n, label)) +
  geom_col(width = 0.65, fill = c_non, alpha = 0.85) +
  geom_text(aes(label = format(n, big.mark = ",")), hjust = -0.1, size = 2.85) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(x = "Modelled lakes (primary cohort)", y = NULL,
       title = "(a) Lakes per region") +
  theme_pub + theme(panel.grid.major.y = element_blank())

r2 <- reg[order(reg$rate), ]
r2$label <- factor(r2$label, levels = r2$label)
p2b <- ggplot(r2, aes(rate, label)) +
  geom_col(width = 0.65, fill = c_glof, alpha = 0.85) +
  geom_text(aes(label = ifelse(pos > 0, sprintf("%d", pos), "")), hjust = -0.25, size = 2.85) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "GLOF source lakes (%)", y = NULL,
       title = "(b) Source-lake rate") +
  theme_pub + theme(panel.grid.major.y = element_blank())

ev_file <- file.path(root, "data", "source", "historical_glofs.csv")
if (file.exists(ev_file)) {
  ev <- read.csv(ev_file)
  ev$trigger[is.na(ev$trigger) | ev$trigger == ""] <- "unknown"
  ev$lake_type[is.na(ev$lake_type) | ev$lake_type == ""] <- "unknown"
  top_tr <- names(sort(table(ev$trigger), decreasing = TRUE))[seq_len(min(8, length(unique(ev$trigger))))]
  ev$trigger_grp <- ifelse(ev$trigger %in% top_tr, ev$trigger, "other")
  ev$trigger_grp <- factor(ev$trigger_grp, levels = names(sort(table(ev$trigger_grp))))
  type_cols <- c(moraine = c_glof, bedrock = c_brown, ice = c_non, unknown = "grey70")
  ev$lake_type <- factor(ev$lake_type, levels = intersect(names(type_cols), unique(ev$lake_type)))

  p2c <- ggplot(ev, aes(y = trigger_grp, fill = lake_type)) +
    geom_bar(width = 0.65) +
    scale_fill_manual(values = type_cols, name = "Dam type") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
    labs(x = sprintf("Events (n = %d, GLOF database v4.2)", nrow(ev)), y = NULL,
         title = "(c) Trigger mechanism by dam type") +
    theme_pub +
    theme(panel.grid.major.y = element_blank(), legend.title = element_text(size = 9),
          legend.position.inside = c(0.98, 0.02), legend.justification = c(1, 0))

  p2d <- ggplot(ev[ev$year >= 1900, ], aes(year, fill = lake_type)) +
    geom_histogram(binwidth = 5, boundary = 1900, colour = "white", linewidth = 0.15) +
    geom_vline(xintercept = fcfg$dem_year + 0.5, linetype = "dashed", colour = c_gray,
               linewidth = 0.45) +
    annotate("text", x = fcfg$dem_year - 1, y = Inf, vjust = 1.5, hjust = 1, size = 2.85,
             label = sprintf("DEM (%d)", fcfg$dem_year)) +
    scale_fill_manual(values = type_cols, guide = "none") +
    scale_y_continuous(breaks = function(l) unique(floor(pretty(l))),
                       expand = expansion(mult = c(0, 0.12))) +
    labs(x = "Year (5-year bins)", y = "GLOF events",
         title = "(d) Events since 1900",
         subtitle = sprintf("%d events before 1900 not shown", sum(ev$year < 1900))) +
    theme_pub
  save_fig("Fig02_inventory", (p2a + p2b) / (p2c + p2d), FIG_W, 6.8)
} else {
  save_fig("Fig02_inventory", p2a + p2b, FIG_W, 2.8)
}

cat("saved figures to", out_dir, "\n")
