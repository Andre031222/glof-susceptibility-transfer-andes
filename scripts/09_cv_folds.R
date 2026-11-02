suppressPackageStartupMessages({
  library(sf)
  library(CAST)
  library(FNN)
})

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
root <- Sys.getenv("GLOF_ROOT", unset = normalizePath(file.path(script_dir, "..")))
models <- file.path(root, "models")
lake_table <- file.path(root, "data", "processed", "labeled", "lake_table.csv")
all_events_table <- file.path(root, "data", "processed", "labeled", "lake_table_all_events.csv")
phys_map <- file.path(root, "data", "processed", "physical", "lake_physical_map.csv")
dir.create(models, showWarnings = FALSE, recursive = TRUE)

outputs <- list(
  folds = file.path(models, "folds_knndm.csv"),
  w = file.path(models, "knndm_W.csv"),
  geodist = file.path(models, "geodist.csv")
)

earth_radius_m <- 6371008.8
equal_area_crs <- "ESRI:102033"

parse_args <- function(args) {
  opts <- list(reps = 50L, k = 5L, maxp = 0.5, max_pred = 5000L, clustering = "kmeans",
               seed = 42L, pred = "unlabelled")
  for (a in args) {
    kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    if (length(kv) != 2) next
    key <- gsub("-", "_", kv[1])
    if (!key %in% names(opts)) stop("unknown argument ", a)
    opts[[key]] <- if (is.character(opts[[key]])) kv[2] else as(kv[2], class(opts[[key]]))
  }
  opts
}

unit_xyz <- function(lon, lat) {
  lo <- lon * pi / 180
  la <- lat * pi / 180
  cbind(cos(la) * cos(lo), cos(la) * sin(lo), sin(la))
}

chord_to_m <- function(chord) 2 * earth_radius_m * asin(pmin(chord / 2, 1))

nn_dist <- function(query, data) chord_to_m(as.vector(knnx.dist(data, query, k = 1)))

cv_nn_dist <- function(xyz, fold) {
  out <- rep(NA_real_, nrow(xyz))
  for (f in unique(fold[!is.na(fold)])) {
    te <- which(fold == f)
    tr <- which(fold != f)
    if (length(tr)) out[te] <- nn_dist(xyz[te, , drop = FALSE], xyz[tr, , drop = FALSE])
  }
  out
}

random_folds <- function(y, k) {
  fold <- integer(length(y))
  for (cls in unique(y)) {
    idx <- which(y == cls)
    fold[idx] <- sample(rep_len(seq_len(k), length(idx)))
  }
  fold
}

group_folds <- function(group, k) {
  sizes <- table(group)
  order_g <- sample(names(sizes))
  load <- numeric(k)
  assign <- integer(length(order_g))
  names(assign) <- order_g
  for (g in order_g[order(-sizes[order_g])]) {
    f <- which.min(load)
    assign[g] <- f
    load[f] <- load[f] + sizes[[g]]
  }
  unname(assign[as.character(group)])
}

wasserstein <- function(a, b) twosamples::wass_stat(a[is.finite(a)], b[is.finite(b)])

read_points <- function() {
  for (f in c(lake_table, phys_map)) {
    if (!file.exists(f)) stop(f, " not found; run 06_physical_lakes.py and 08_lake_table.py first")
  }
  cols <- c("lake_phys", "label", "group", "lon", "lat")
  train <- read.csv(lake_table)[, cols]
  if (file.exists(all_events_table)) train <- rbind(train, read.csv(all_events_table)[, cols])
  train <- train[order(train$lake_phys, -train$label), ]
  train <- train[!duplicated(train$lake_phys), ]
  pred <- read.csv(phys_map)[, c("lake_phys", "lon", "lat")]
  pred <- pred[!duplicated(pred$lake_phys) & is.finite(pred$lon) & is.finite(pred$lat), ]
  list(train = train, pred = pred)
}

as_points <- function(d, clustering) {
  p <- st_as_sf(d, coords = c("lon", "lat"), crs = 4326)
  if (clustering == "kmeans") p <- st_transform(p, equal_area_crs)
  p
}

run_knndm <- function(tpts, pred_all, opts, seed) {
  set.seed(seed)
  pp <- pred_all
  if (nrow(pp) > opts$max_pred) pp <- pp[sample(nrow(pp), opts$max_pred), ]
  dist_fun <- if (opts$clustering == "kmeans") "euclidean" else "great_circle"
  set.seed(seed)
  kn <- suppressMessages(knndm(tpts, predpoints = pp, k = opts$k, maxp = opts$maxp,
                               clustering = opts$clustering, dist_fun = dist_fun))
  nk <- as.character(kn$q)
  method <- if (identical(nk, "random CV")) "knndm_random_cv" else paste0("knndm_", opts$clustering)
  list(fold = as.integer(kn$clusters), W = as.numeric(kn$W), method = method, nk = nk)
}

canonical <- function(fold) {
  first <- match(fold, unique(fold))
  paste(first, collapse = ",")
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  if (!opts$pred %in% c("all", "unlabelled")) stop("--pred must be 'all' or 'unlabelled'")
  pts <- read_points()
  train <- pts$train
  pred <- pts$pred
  unlabelled <- pred[!pred$lake_phys %in% train$lake_phys, ]
  if (opts$pred == "unlabelled" && nrow(unlabelled) < 10) {
    cat(sprintf("only %d unlabelled lakes; prediction points fall back to all lakes\n", nrow(unlabelled)))
  } else if (opts$pred == "unlabelled") {
    pred <- unlabelled
    cat("prediction points exclude the training lakes (--pred=unlabelled)\n")
  } else {
    cat(sprintf("prediction points include the %d training lakes (distance 0 to themselves); --pred=unlabelled drops them\n",
                sum(pred$lake_phys %in% train$lake_phys)))
  }
  cat(sprintf("training lakes %d (positives %d, groups %d); prediction lakes %d\n",
              nrow(train), sum(train$label), length(unique(train$group)), nrow(pred)))
  if (nrow(pred) > opts$max_pred) {
    cat(sprintf("prediction points subsampled to %d per repetition (seed = repetition seed) for knndm; G-functions use all %d\n",
                opts$max_pred, nrow(pred)))
  }
  cat(sprintf("knndm: k = %d, maxp = %.2f, clustering = %s, repetitions = %d (seeds %d-%d)\n",
              opts$k, opts$maxp, opts$clustering, opts$reps, opts$seed,
              opts$seed + opts$reps - 1L))

  tpts <- as_points(train, opts$clustering)
  ppts <- as_points(pred, opts$clustering)
  xyz_t <- unit_xyz(train$lon, train$lat)
  xyz_p <- unit_xyz(pred$lon, pred$lat)
  gij <- nn_dist(xyz_p, xyz_t)

  folds <- list()
  w_rows <- list()
  seen <- integer(0)
  t0 <- Sys.time()
  for (r in seq_len(opts$reps)) {
    seed <- opts$seed + r - 1L
    kn <- run_knndm(tpts, ppts, opts, seed)
    key <- canonical(kn$fold)
    dup <- key %in% names(seen)
    if (!dup) {
      seen[key] <- length(folds)
      folds[[length(folds) + 1]] <- data.frame(lake_phys = train$lake_phys,
                                               rep = length(folds), fold = kn$fold - 1L)
    }
    w_rows[[length(w_rows) + 1]] <- data.frame(rep = r - 1L, seed = seed, W = kn$W,
                                               method = kn$method, nk = kn$nk,
                                               duplicate = dup, fold_rep = seen[[key]])
    cat(sprintf("  rep %2d seed %d  W = %.1f  %s  nk = %s%s\n", r - 1L, seed, kn$W,
                kn$method, kn$nk, if (dup) "  (duplicate partition, not written)" else ""))
  }
  cat(sprintf("knndm done in %.1f min: %d distinct partitions out of %d repetitions\n",
              as.numeric(difftime(Sys.time(), t0, units = "mins")), length(folds), opts$reps))
  if (length(folds) < opts$reps) {
    cat("  identical partitions are written once so repeated CV does not count them twice;",
        "use --clustering=kmeans (projected coordinates) for more variation\n")
  }
  fold_df <- do.call(rbind, folds)

  set.seed(opts$seed)
  rf <- random_folds(train$label, opts$k)
  gf <- group_folds(train$group, opts$k)
  kf <- fold_df$fold[fold_df$rep == 0]
  d_random <- cv_nn_dist(xyz_t, rf)
  d_group <- cv_nn_dist(xyz_t, gf)
  d_knndm <- cv_nn_dist(xyz_t, kf)
  d_sample <- cv_nn_dist(xyz_t, seq_len(nrow(train)))

  w_df <- do.call(rbind, w_rows)
  w_ref <- data.frame(rep = 0L, seed = opts$seed,
                      W = c(wasserstein(d_random, gij), wasserstein(d_group, gij)),
                      method = c("random", "group"), nk = NA_character_, duplicate = FALSE,
                      fold_rep = NA_integer_)
  w_df <- rbind(w_df, w_ref)

  geo <- rbind(
    data.frame(what = "prediction-to-sample", distance_m = gij),
    data.frame(what = "sample-to-sample", distance_m = d_sample),
    data.frame(what = "CV-random", distance_m = d_random),
    data.frame(what = "CV-group", distance_m = d_group),
    data.frame(what = "CV-knndm", distance_m = d_knndm)
  )
  geo$distance_m <- round(geo$distance_m, 1)

  write.csv(fold_df, outputs$folds, row.names = FALSE)
  write.csv(w_df, outputs$w, row.names = FALSE)
  write.csv(geo, outputs$geodist, row.names = FALSE)

  med <- tapply(geo$distance_m, geo$what, median, na.rm = TRUE)
  cat("median nearest-neighbour distance (km):\n")
  for (nm in names(med)) cat(sprintf("  %-22s %8.2f\n", nm, med[[nm]] / 1000))
  cat(sprintf("W against prediction-to-sample: random %.1f, group %.1f, knndm (rep 0) %.1f\n",
              w_ref$W[1], w_ref$W[2], w_df$W[1]))
  cat("saved", paste(basename(unlist(outputs)), collapse = ", "), "\n")
}

main()
