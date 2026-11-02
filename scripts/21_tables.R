suppressPackageStartupMessages(library(jsonlite))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
root <- Sys.getenv("GLOF_ROOT", unset = normalizePath(file.path(script_dir, "..")))
models <- file.path(root, "models")
lake_table <- file.path(root, "data", "processed", "labeled", "lake_table.csv")
out_dir <- file.path(root, "paper", "tables")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fmt <- function(x, d) ifelse(is.na(x), "--", sprintf(paste0("%.", d, "f"), x))
grp <- function(x) formatC(x, format = "d", big.mark = "{,}")
pct <- function(x, d = 1) paste0(fmt(x, d), "\\%")
is_true <- function(x) tolower(as.character(x)) %in% c("true", "1")
interval <- function(lo, hi, d = 3) ifelse(is.na(lo) | is.na(hi) | lo == hi, "--",
                                           sprintf("[%s, %s]", fmt(lo, d), fmt(hi, d)))
pval <- function(p) ifelse(is.na(p), "--", ifelse(p < 0.001, "$<$0.001", fmt(p, 3)))
emit <- function(rows, file) {
  writeLines(paste0(rows, " \\\\"), file.path(out_dir, file))
  cat(sprintf("  %-30s %d rows\n", file, length(rows)))
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

oof <- read.csv(file.path(models, "oof_scores.csv"))
stats <- fromJSON(file.path(models, "paper_stats.json"))
run <- fromJSON(file.path(models, "run_summary.json"))
fcfg <- fromJSON(file.path(models, "feature_sets.json"))
perf <- read_opt("performance_by_scheme.csv")

n_lakes <- nrow(oof)
n_pos <- sum(oof$glof)
primary_model <- run$primary_model
primary_set <- run$primary_feature_set
t_youden <- stats$youden_threshold

region_label <- c(
  cordillera_blanca = "Blanca", cordillera_huayhuash = "Huayhuash",
  cordillera_raura = "Raura", cordillera_central = "Central",
  cordillera_urubamba = "Urubamba", cordillera_vilcanota = "Vilcanota",
  cordillera_huanzo = "Huanzo", carabaya = "Carabaya",
  apolobamba = "Apolobamba", bolivia_cordillera_real = "Real",
  bolivia_norte = "Bolivia Norte", ecuador_antisana = "Antisana",
  chile_andes_centrales = "Central Andes",
  patagonia_npi = "N. Patagonian Icefield",
  patagonia_norte = "Patagonia Norte", patagonia_sur = "Patagonia Sur")
nice_region <- function(x) {
  lab <- unname(region_label[x])
  ifelse(is.na(lab), gsub("_", " ", x), lab)
}
scheme_disp <- c(random = "Random (stratified)", group = "Sub-basin grouped",
                 knndm = "kNNDM", region = "Leave-one-region-out",
                 forward = sprintf("Forward (cut-off %d)", fcfg$forward_year))
model_disp <- c(BalancedRF = "BalancedRF", EasyEnsemble = "Easy Ensemble",
                RandomForest = "Random Forest", XGBoost = "XGBoost",
                LightGBM = "LightGBM", LogisticRegression = "Logistic Regr.")
cohort_disp <- c(primary = "Primary (DEM-era events)", all_events_no_dem = "All events, no DEM",
                 unambiguous = "Unambiguous matches", static_only = "Static features only",
                 bias_adjusted = "Access covariates fixed")
set_disp <- c(full = "All (17)", no_dem = "No elevation (9)", static = "Static (13)")
disp <- function(map, x) ifelse(is.na(map[x]), x, unname(map[x]))

cat("generating tables from", n_lakes, "lakes /", n_pos, "positives\n")

MIN_LAKES_PCT <- 30
q90 <- quantile(oof$oof_score, 0.90, na.rm = TRUE)
oof$top <- as.integer(oof$oof_score >= q90)
agg <- aggregate(cbind(n = rep(1, nrow(oof)), pos = glof, top = top) ~ analysis_region,
                 data = oof, FUN = sum)
countries <- tapply(oof$country, oof$analysis_region, function(v) {
  paste(names(sort(table(v), decreasing = TRUE)), collapse = "/")
})
agg$country <- unname(countries[agg$analysis_region])
agg$label <- nice_region(agg$analysis_region)
agg$topdec <- 100 * agg$top / agg$n
agg <- agg[order(agg$country, -agg$n), ]
rows <- vapply(seq_len(nrow(agg)), function(i) {
  small <- agg$n[i] < MIN_LAKES_PCT
  paste(paste0(agg$label[i], if (small) "$^a$" else ""), agg$country[i], grp(as.integer(agg$n[i])),
        as.integer(agg$pos[i]), if (small) "--" else fmt(agg$topdec[i], 1), sep = " & ")
}, character(1))
rows <- c(rows, paste("\\midrule\n\\textbf{Total}", "",
                      paste0("\\textbf{", grp(n_lakes), "}"),
                      paste0("\\textbf{", n_pos, "}"),
                      paste0("\\textbf{", fmt(100 * sum(agg$top) / n_lakes, 1), "}"),
                      sep = " & "))
emit(rows, "tab_study_areas.tex")

if (!is.null(perf)) {
  p <- perf[perf$cohort == "primary" & perf$model == primary_model, ]
  p <- p[order(match(p$scheme, names(scheme_disp))), ]
  rows <- vapply(seq_len(nrow(p)), function(i) {
    r <- p[i, ]
    paste(disp(scheme_disp, r$scheme), r$n_pos, fmt(r$auc, 3), interval(r$auc_rep_lo, r$auc_rep_hi),
          fmt(r$auc_oof, 3), interval(r$auc_lo, r$auc_hi), fmt(r$pr_auc_oof, 3),
          interval(r$pr_auc_lo, r$pr_auc_hi), fmt(r$recall_top10, 2),
          fmt(r$median_nn_km, 2), sep = " & ")
  }, character(1))
  emit(rows, "tab_performance_by_scheme.tex")

  s <- perf[perf$model == primary_model, ]
  s <- s[order(match(s$cohort, names(cohort_disp)), match(s$scheme, names(scheme_disp))), ]
  rows <- vapply(seq_len(nrow(s)), function(i) {
    r <- s[i, ]
    paste(disp(cohort_disp, r$cohort), disp(set_disp, r$feature_set), disp(scheme_disp, r$scheme), r$n_pos,
          fmt(r$auc, 3), interval(r$auc_rep_lo, r$auc_rep_hi), fmt(r$pr_auc, 3),
          fmt(100 * r$prevalence, 2), sep = " & ")
  }, character(1))
  emit(rows, "tab_sensitivity.tex")
}

tr <- read_opt("transfer_by_region.csv")
if (!is.null(tr)) {
  aoa <- read_opt("aoa_by_region.csv")
  if (!is.null(aoa)) {
    tr <- merge(tr, aoa[, c("held_out", "frac_inside_aoa", "auc_inside_aoa", "di_median")],
                by = "held_out", all.x = TRUE)
  } else {
    tr$frac_inside_aoa <- NA
    tr$auc_inside_aoa <- NA
    tr$di_median <- NA
  }
  tr <- tr[order(-tr$n_pos, tr$held_out), ]
  rows <- vapply(seq_len(nrow(tr)), function(i) {
    r <- tr[i, ]
    star <- if (is_true(r$descriptive_only)) "$^*$" else ""
    paste(nice_region(r$held_out), grp(as.integer(r$n_test)), paste0(r$n_pos, star),
          r$n_train_pos, paste0(fmt(r$auc, 2), star),
          fmt(r$recall_top10, 2), fmt(r$di_median, 2), pct(100 * r$frac_inside_aoa, 0),
          fmt(r$auc_inside_aoa, 2), sep = " & ")
  }, character(1))
  emit(rows, "tab_transfer_region.tex")
}

tests <- read_opt("model_tests.csv")
if (!is.null(tests) && !is.null(perf)) {
  pp <- perf[perf$cohort == "primary", ]
  rows <- character(0)
  for (sch in intersect(names(scheme_disp), unique(tests$scheme))) {
    tt <- tests[tests$scheme == sch, ]
    base <- pp[pp$scheme == sch & pp$model == primary_model, ][1, ]
    rows <- c(rows, sprintf("\\multicolumn{7}{l}{\\textit{%s CV}}", disp(scheme_disp, sch)),
              paste(paste0("\\textbf{", disp(model_disp, primary_model), "}$^\\dagger$"),
                    fmt(base$auc, 3), interval(base$auc_rep_lo, base$auc_rep_hi), fmt(base$pr_auc, 3),
                    "--", "--", "--", sep = " & "))
    tt <- tt[order(tt$mean_diff), ]
    for (i in seq_len(nrow(tt))) {
      r <- tt[i, ]
      o <- pp[pp$scheme == sch & pp$model == r$model_b, ][1, ]
      rows <- c(rows, paste(disp(model_disp, r$model_b), fmt(o$auc, 3),
                            interval(o$auc_rep_lo, o$auc_rep_hi), fmt(o$pr_auc, 3),
                            sprintf("%+.4f", r$mean_diff), fmt(r$corrected_se, 4),
                            pval(r$p_value), sep = " & "))
    }
  }
  emit(rows, "tab_model_comparison.tex")
}

wl <- read_opt("watch_list.csv")
if (!is.null(wl)) {
  rows <- vapply(seq_len(nrow(wl)), function(i) {
    r <- wl[i, ]
    paste(pct(r$top_pct), grp(as.integer(r$n_lakes)), as.integer(r$pos_caught),
          pct(100 * r$recall, 0), paste0("$", fmt(r$lift, 1), "\\times$"), sep = " & ")
  }, character(1))
  emit(rows, "tab_watchlist.tex")
}

th <- read_opt("threshold_sensitivity.csv")
if (!is.null(th)) {
  near_youden <- which.min(abs(th$threshold - t_youden))
  rows <- vapply(seq_len(nrow(th)), function(i) {
    t <- th$threshold[i]
    tlab <- if (i == near_youden) paste0("$", fmt(t, 2), "^\\dagger$") else fmt(t, 2)
    paste(tlab, pct(th$pct_flagged[i]), pct(100 * th$recall[i]),
          paste0(th$pos_caught[i], " / ", n_pos), sep = " & ")
  }, character(1))
  emit(rows, "tab_threshold.tex")
}

feat_label <- c(
  area_m2 = "Area (m$^2$)", perimeter_m = "Perimeter (m)", compactness = "Compactness",
  elongation = "Elongation", fractal_dim = "Shoreline fractal dimension",
  convexity = "Convexity", shore_dev_index = "Shoreline development index",
  lake_elev_m = "Lake elevation (m a.s.l.)", freeboard_m = "Dam freeboard (m)",
  outlet_drop_m = "Outlet drop (m)", ring_slope_mean = "Surrounding slope, mean (deg)",
  ring_slope_p90 = "Surrounding slope, p90 (deg)", ring_relief_m = "Surrounding relief (m)",
  ring_aspect_north = "Surrounding northness", ring_aspect_east = "Surrounding eastness",
  ring_tri_mean = "Terrain ruggedness index", ring_curvature_mean = "Surrounding curvature",
  ring_roughness_mean = "Surrounding roughness", ring_vrm_mean = "Vector ruggedness measure",
  dist_glacier_m = "Distance to glacier (m)",
  area_trend = "Pre-cutoff area trend (m$^2$ yr$^{-1}$)",
  area_cv = "Pre-cutoff area CV", presence_rate = "Pre-cutoff presence rate",
  area_last_m2 = "Last pre-cutoff area (m$^2$)",
  years_since_first_seen = "Years since first detection")

mw <- read_opt("mann_whitney.csv")
if (!is.null(mw)) {
  mw$significant <- is_true(mw$significant)
  sig <- mw[mw$significant, ]
  shown <- if (nrow(sig)) sig[order(sig$p_value), ] else head(mw[order(mw$p_value), ], 5)
  big <- function(v) ifelse(abs(v) >= 1000, grp(round(v)),
                     ifelse(abs(v) >= 1, fmt(v, 2), fmt(v, 4)))
  rows <- vapply(seq_len(nrow(shown)), function(i) {
    r <- shown[i, ]
    dir <- if (r$rank_biserial < 0) "GLOF$<$" else "GLOF$>$"
    paste(disp(feat_label, r$feature), big(r$median_glof), big(r$median_nonglof),
          paste(fmt(abs(r$rank_biserial), 2), dir), pval(r$p_bonferroni), sep = " & ")
  }, character(1))
  emit(rows, "tab_mannwhitney.tex")
  cat(sprintf("  (%d of %d features significant after Bonferroni)\n", nrow(sig), nrow(mw)))
}

prosp_cols <- c("lake_elev_m", "freeboard_m", "dist_glacier_m", "ring_vrm_mean", "area_m2")
if (file.exists(lake_table)) {
  lt <- read.csv(lake_table)
  lt <- lt[, c("lake_phys", intersect(setdiff(prosp_cols, "area_m2"), names(lt)))]
  m <- merge(oof, lt, by.x = "lake_key", by.y = "lake_phys")
  prosp_cols <- intersect(prosp_cols, names(m))
  nonsrc <- m[m$glof == 0, ]
  top100 <- nonsrc[order(-nonsrc$oof_score), ][seq_len(min(100, nrow(nonsrc))), ]
  src <- m[m$glof == 1, ]
  rows <- vapply(prosp_cols, function(c) {
    f <- function(v) {
      md <- median(v, na.rm = TRUE)
      if (abs(md) >= 1000) grp(round(md)) else if (abs(md) >= 1) fmt(md, 1) else fmt(md, 4)
    }
    paste(disp(feat_label, c), f(src[[c]]), f(top100[[c]]), f(nonsrc[[c]]), sep = " & ")
  }, character(1))
  emit(rows, "tab_prospective.tex")
  tt <- sort(table(top100$analysis_region), decreasing = TRUE)
  k <- seq_len(min(5, length(tt)))
  cat("  top-100 uncatalogued lakes by region:",
      paste(sprintf("%s (%d)", nice_region(names(tt)[k]), as.integer(tt)[k]), collapse = ", "),
      "\n")
}

wr <- stats$within_region
if (!is.null(wr)) {
  rg <- wr$regions
  rg <- rg[order(-rg$n_pos), ]
  rows <- vapply(seq_len(nrow(rg)), function(i) {
    paste(nice_region(rg$region[i]), grp(as.integer(rg$n[i])), rg$n_pos[i],
          paste(fmt(rg$auc_within[i], 2), interval(rg$auc_within_lo[i], rg$auc_within_hi[i], 2)),
          if (is.null(rg$auc_heldout)) "--" else
            paste(fmt(rg$auc_heldout[i], 2), interval(rg$auc_heldout_lo[i], rg$auc_heldout_hi[i], 2)),
          sep = " & ")
  }, character(1))
  rows <- c(rows,
            paste("\\midrule\nMean of regions", "--", sum(rg$n_pos),
                  paste(fmt(wr$auc_within_mean, 2), interval(wr$auc_within_mean_lo,
                                                            wr$auc_within_mean_hi, 2)),
                  fmt(mean(rg$auc_heldout), 2), sep = " & "))
  for (k in c("tropical", "extratropical")) {
    b <- wr[[k]]
    lab <- if (k == "tropical") "Tropical Andes, pooled"
           else "Extratropical Andes, pooled"
    rows <- c(rows, paste(lab, grp(as.integer(b$n_scored)), b$n_pos_scored,
                          paste(fmt(b$auc_oof, 2), interval(b$auc_lo, b$auc_hi, 2)), "--",
                          sep = " & "))
  }
  emit(rows, "tab_within_region.tex")
}

iv <- read_opt("inventory_validation.csv")
if (!is.null(iv)) {
  ref_disp <- c(wood2021 = "Wood et al. (2021), Peru",
                macmanaway2025 = "MacManaway et al. (2025), Bolivia",
                zhang2024 = "Zhang et al. (2024), Andes")
  size_order <- c("all", "<2 ha", "2-10 ha", "10-100 ha", ">100 ha")
  rows <- character(0)
  for (ref in intersect(names(ref_disp), unique(iv$reference))) {
    rows <- c(rows, sprintf("\\multicolumn{7}{l}{\\textit{%s}}", ref_disp[[ref]]))
    for (sen in c("s2", "landsat")) {
      d <- iv[iv$reference == ref & iv$sensor == sen & iv$area_name == "ALL", ]
      if (!nrow(d)) next
      for (sz in intersect(size_order, unique(d$size_class))) {
        e <- d[d$size_class == sz, ][1, ]
        rows <- c(rows, paste(if (sen == "s2") "Sentinel-2" else "Landsat",
                              gsub("<", "$<$", gsub(">", "$>$", sz)), grp(as.integer(e$n_ref)),
                              fmt(e$recall, 2), fmt(e$precision, 2), fmt(e$f1, 2),
                              fmt(e$median_ape_pct, 0), sep = " & "))
      }
    }
  }
  emit(rows, "tab_detector_validation.tex")
}

cat("\nkey values for the manuscript text:\n")
cat(sprintf("  lakes=%s  positives=%d  imbalance=1:%s  grouping=%s\n", grp(n_lakes), n_pos,
            grp(round((n_lakes - n_pos) / n_pos)), run$group_level))
if (!is.null(perf)) {
  p <- perf[perf$cohort == "primary" & perf$model == primary_model, ]
  for (i in seq_len(nrow(p)))
    cat(sprintf("  %-22s AUC=%.3f %s  PR-AUC=%.3f  n+=%d  reps=%d\n",
                disp(scheme_disp, p$scheme[i]), p$auc[i], interval(p$auc_rep_lo[i], p$auc_rep_hi[i]),
                p$pr_auc[i], as.integer(p$n_pos[i]), as.integer(p$n_repeats[i])))
}
cat(sprintf("  Youden t=%.3f  KS D=%.3f p=%.4f\n", t_youden, stats$ks_test$ks_D,
            stats$ks_test$p_value))
if (!is.null(stats$spatial))
  cat(sprintf("  residual autocorrelation range ~%s km; median predictor range ~%s km; pooled AOA fraction %.2f\n",
              stats$spatial$residual_range_km, stats$spatial$predictor_range_km_median,
              stats$spatial$aoa_pooled_fraction_inside))
if (!is.null(tests))
  for (i in seq_len(nrow(tests)))
    cat(sprintf("  Nadeau-Bengio %s: %s vs %s dAUC=%+.4f p=%.3f\n", tests$scheme[i],
                tests$model_a[i], tests$model_b[i], tests$mean_diff[i], tests$p_value[i]))
cat("\nsaved to", out_dir, "\n")
