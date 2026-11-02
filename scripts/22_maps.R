suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(ggplot2)
  library(tidyterra)
  library(ggspatial)
  library(patchwork)
})

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
root <- Sys.getenv("GLOF_ROOT", unset = normalizePath(file.path(script_dir, "..")))

models   <- file.path(root, "models")
dem_dir  <- file.path(root, "data", "interim", "dem")
rgi_path <- file.path(root, "data", "raw", "rgi7_andes.gpkg")
out_dir  <- file.path(root, "paper", "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

c_low  <- "#2c5f8a"
c_mid  <- "#e8c547"
c_high <- "#c1272d"
c_ice  <- "#cfe8f5"

theme_map <- theme_minimal(base_size = 9) +
  theme(
    panel.grid = element_line(linewidth = 0.2, colour = "grey80", linetype = "dotted"),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.title = element_text(face = "bold", size = 9.5),
    plot.subtitle = element_text(size = 8, colour = "grey30"),
    legend.position = "right",
    legend.key.height = unit(1.1, "cm"),
    legend.key.width = unit(0.35, "cm"),
    axis.title = element_blank()
  )

hillshade <- function(dem) {
  slope  <- terrain(dem, "slope", unit = "radians")
  aspect <- terrain(dem, "aspect", unit = "radians")
  shade(slope, aspect, angle = 40, direction = 315)
}

lake_points <- function() {
  oof_file <- file.path(models, "oof_scores.csv")
  if (!file.exists(oof_file)) return(NULL)
  m <- read.csv(oof_file)
  m <- m[is.finite(m$lon) & is.finite(m$lat) & is.finite(m$oof_score), ]
  if (!nrow(m)) return(NULL)
  st_as_sf(m, coords = c("lon", "lat"), crs = 4326)
}

region_plot <- function(area, lakes = NULL, glaciers = NULL, title = NULL, legend = TRUE) {
  dem_file <- file.path(dem_dir, paste0(area, "_dem_utm.tif"))
  if (!file.exists(dem_file)) return(NULL)

  dem <- rast(dem_file)
  if (ncell(dem) > 4e6) dem <- aggregate(dem, fact = ceiling(sqrt(ncell(dem) / 4e6)))
  hs <- hillshade(dem)
  target <- crs(dem)
  ext_ll <- st_bbox(st_transform(st_as_sfc(st_bbox(ext(dem), crs = target)), 4326))

  p <- ggplot() +
    geom_spatraster(data = hs, show.legend = FALSE) +
    scale_fill_gradient(low = "grey25", high = "white", na.value = "white") +
    ggnewscale::new_scale_fill()

  if (!is.null(glaciers)) {
    g <- suppressWarnings(st_crop(st_transform(glaciers, target), ext(dem)))
    if (nrow(g)) {
      p <- p + geom_sf(data = g, fill = c_ice, colour = "#7fb3d3",
                       linewidth = 0.15, alpha = 0.85)
    }
  }

  if (!is.null(lakes)) {
    l <- suppressWarnings(st_crop(st_transform(lakes, target), ext(dem)))
    if (nrow(l)) {
      p <- p +
        geom_sf(data = l, aes(colour = oof_score, size = log10(area_m2)),
                alpha = 0.85) +
        scale_colour_gradient2(low = c_low, mid = c_mid, high = c_high,
                               midpoint = 0.5, limits = c(0, 1),
                               name = "Susceptibility",
                               guide = if (legend) "colourbar" else "none") +
        scale_size_continuous(range = c(0.4, 3), guide = "none")
      src <- l[l$glof == 1, ]
      if (nrow(src)) {
        p <- p + geom_sf(data = src, shape = 24, fill = "black",
                         colour = "white", size = 2.4, stroke = 0.5)
      }
    }
  }

  p <- p +
    annotation_scale(location = "bl", width_hint = 0.25, height = unit(0.15, "cm"),
                     text_cex = 0.6) +
    annotation_north_arrow(location = "tr", height = unit(0.8, "cm"),
                           width = unit(0.8, "cm"),
                           style = north_arrow_minimal()) +
    coord_sf(crs = target, expand = FALSE) +
    labs(title = if (is.null(title)) gsub("_", " ", area) else title,
         subtitle = sprintf("%.2f to %.2f deg lon, %.2f to %.2f deg lat",
                            ext_ll["xmin"], ext_ll["xmax"],
                            ext_ll["ymin"], ext_ll["ymax"])) +
    theme_map
  p
}

region_map <- function(area, lakes = NULL, glaciers = NULL) {
  p <- region_plot(area, lakes, glaciers)
  if (is.null(p)) {
    message("  [skip] ", area, " - no DEM")
    return(invisible(NULL))
  }
  out <- file.path(out_dir, paste0("OR1_map_", area, ".png"))
  ggsave(out, p, width = 7.5, height = 6.5, dpi = 600, bg = "white")
  ggsave(sub("\\.png$", ".pdf", out), p, width = 7.5, height = 6.5, bg = "white",
         device = if (capabilities("cairo")) cairo_pdf else "pdf")
  message("  saved ", basename(out), " (png 600 dpi + pdf)")
  invisible(out)
}

nice_region <- function(x) gsub("_", " ", x)

area_short <- c(
  ecuador_antisana = "Antisana", cordillera_blanca = "C. Blanca",
  cordillera_huayhuash = "C. Huayhuash", cordillera_raura = "C. Raura",
  cordillera_central = "C. Central", cordillera_urubamba = "C. Urubamba",
  cordillera_vilcanota = "C. Vilcanota", carabaya = "Carabaya",
  apolobamba = "Apolobamba", bolivia_norte = "Bolivia Norte",
  bolivia_cordillera_real = "C. Real", chile_andes_centrales = "Central Andes",
  patagonia_norte = "N. Patagonian Icefield")

area_frames <- function() {
  files <- Sys.glob(file.path(dem_dir, "*_dem_utm.tif"))
  rows <- lapply(files, function(f) {
    r <- rast(f)
    b <- st_as_sfc(st_bbox(ext(r), crs = st_crs(crs(r))))
    st_sf(area = sub("_dem_utm\\.tif$", "", basename(f)), geometry = st_transform(b, 4326))
  })
  a <- do.call(rbind, rows)
  a <- a[order(-st_coordinates(st_centroid(a))[, 2]), ]
  a$id <- seq_len(nrow(a))
  a$label <- sprintf("%d %s", a$id, area_short[a$area])
  a
}

study_area_map <- function(glaciers, lakes) {
  ctry_zip <- file.path(root, "data", "raw", "boundaries", "ne_10m_admin_0_countries.zip")
  ev_file <- file.path(root, "data", "processed", "labeled", "event_match_log.csv")
  if (!file.exists(ctry_zip) || !file.exists(ev_file)) {
    message("  [skip] study area map: boundaries or event log missing")
    return(invisible(NULL))
  }
  ctry <- st_read(paste0("/vsizip/", ctry_zip), quiet = TRUE)
  ctry <- ctry[ctry$CONTINENT == "South America", ]
  frames <- area_frames()
  win <- c(xmin = -82, xmax = -60, ymin = -50, ymax = 2)
  sf_use_s2(FALSE)
  ctry <- suppressWarnings(suppressMessages(st_crop(st_make_valid(ctry), win)))
  ice <- suppressWarnings(st_centroid(st_geometry(glaciers)))
  ev <- read.csv(ev_file)
  ev <- ev[is.finite(ev$lon) & is.finite(ev$lat), ]
  ev$class <- ifelse(ev$status == "matched", "matched to a mapped lake",
                     "outside study areas or unmatched")
  ev <- st_as_sf(ev, coords = c("lon", "lat"), crs = 4326)
  lab <- cbind(st_drop_geometry(frames), st_coordinates(st_centroid(frames)))

  pa <- ggplot() +
    geom_sf(data = ctry, fill = "grey95", colour = "grey55", linewidth = 0.2) +
    geom_sf(data = ice, colour = "#7fb3d3", size = 0.05, alpha = 0.6) +
    geom_sf(data = ev, aes(shape = class, colour = class), size = 0.9, stroke = 0.35) +
    scale_shape_manual(values = c(17, 1), name = "GLOF record") +
    scale_colour_manual(values = c(c_high, "grey35"), name = "GLOF record") +
    geom_sf(data = frames, fill = NA, colour = "black", linewidth = 0.35) +
    ggrepel::geom_text_repel(data = lab, aes(X, Y, label = id), size = 2.85,
                             nudge_x = -2.2, segment.size = 0.2, min.segment.length = 0,
                             seed = 1) +
    annotation_scale(location = "bl", width_hint = 0.3, height = unit(0.12, "cm"),
                     text_cex = 0.55) +
    coord_sf(xlim = win[c("xmin", "xmax")], ylim = win[c("ymin", "ymax")], expand = FALSE) +
    labs(title = "(a) Study areas and GLOF records") +
    theme_map +
    theme(legend.position = "inside", legend.position.inside = c(0.98, 0.02),
          legend.justification = c(1, 0), legend.key.height = unit(0.4, "cm"),
          legend.background = element_rect(fill = "white", colour = "grey70",
                                            linewidth = 0.2),
          legend.text = element_text(size = 7), legend.title = element_text(size = 7.5))

  key <- paste(lab$label, collapse = "\n")
  pa <- pa + annotate("label", x = -81.5, y = -36, label = key, hjust = 0, vjust = 1,
                      size = 2.85, label.size = 0.2, fill = "white", lineheight = 0.95)

  pb <- region_plot("cordillera_blanca", lakes, glaciers,
                    title = "(b) Cordillera Blanca, Peru", legend = TRUE)
  pc <- region_plot("apolobamba", lakes, glaciers,
                    title = "(c) Apolobamba, Peru-Bolivia", legend = FALSE)
  right <- (pb / pc) & theme(plot.subtitle = element_blank())
  p <- pa + right + plot_layout(widths = c(1, 1.15))
  out <- file.path(out_dir, "Fig01_study_area.png")
  ggsave(out, p, width = 10.5, height = 11, dpi = 600, bg = "white")
  ggsave(sub("\\.png$", ".pdf", out), p, width = 10.5, height = 11, bg = "white",
         device = if (capabilities("cairo")) cairo_pdf else "pdf")
  message("  saved ", basename(out), " (png 600 dpi + pdf)")
  invisible(out)
}

aoa_map <- function() {
  di_file <- file.path(models, "di_lakes.csv")
  reg_file <- file.path(models, "aoa_by_region.csv")
  if (!file.exists(di_file) || !file.exists(reg_file)) {
    message("  [skip] AOA map: run scripts/11_spatial_diagnostics.py first")
    return(invisible(NULL))
  }
  di <- read.csv(di_file)
  reg <- read.csv(reg_file)
  di$inside <- tolower(as.character(di$inside_aoa)) %in% c("true", "1")
  di$di_rel <- pmax(di$di / di$di_threshold, 1e-3)
  pts <- st_as_sf(di[order(di$di_rel), ], coords = c("lon", "lat"), crs = 4326)
  outside <- pts[!pts$inside, ]
  src <- pts[pts$glof == 1, ]

  ctry_zip <- file.path(root, "data", "raw", "boundaries", "ne_10m_admin_0_countries.zip")
  pa <- ggplot()
  if (file.exists(ctry_zip)) {
    sf_use_s2(FALSE)
    ctry <- st_read(paste0("/vsizip/", ctry_zip), quiet = TRUE)
    ctry <- suppressWarnings(suppressMessages(st_crop(st_make_valid(ctry[ctry$CONTINENT == "South America", ]),
                                                      c(xmin = -82, xmax = -60, ymin = -50, ymax = 2))))
    pa <- pa + geom_sf(data = ctry, fill = "grey96", colour = "grey70", linewidth = 0.15)
  }
  pa <- pa +
    geom_sf(data = pts, aes(colour = di_rel), size = 0.55, alpha = 0.85) +
    scale_colour_gradient2(low = c_low, mid = c_mid, high = c_high, midpoint = 0,
                           trans = "log10", name = "DI / threshold",
                           breaks = c(0.1, 0.3, 1, 3, 10), oob = scales::squish,
                           limits = c(0.05, 20))
  if (nrow(outside)) {
    pa <- pa + geom_sf(data = outside, shape = 4, size = 0.4, colour = "black", stroke = 0.12,
                       alpha = 0.5)
  }
  if (nrow(src)) {
    pa <- pa + geom_sf(data = src, shape = 24, fill = "#ff2a6d", colour = "black", size = 1.6,
                       stroke = 0.3)
  }
  pa <- pa +
    coord_sf(xlim = c(-82, -64), ylim = c(-49, 1), expand = FALSE) +
    labs(title = "(a) Dissimilarity to training data",
         subtitle = "x: outside the AOA; triangles: GLOF sources") +
    theme_map + theme(axis.title = element_blank(), legend.position = "left",
                      legend.key.height = unit(0.9, "cm"))

  reg$label <- ifelse(reg$held_out == "patagonia_npi", "N. Patagonian Icefield",
                      ifelse(is.na(area_short[reg$held_out]), nice_region(reg$held_out),
                             area_short[reg$held_out]))
  reg <- reg[order(reg$frac_inside_aoa), ]
  reg$label <- factor(reg$label, levels = reg$label)
  pb <- ggplot(reg, aes(100 * frac_inside_aoa, label)) +
    geom_col(width = 0.6, fill = c_low) +
    geom_text(aes(label = sprintf("%.0f%% (n+ = %d)", 100 * frac_inside_aoa, n_pos)),
              hjust = -0.08, size = 2.85) +
    scale_x_continuous(limits = c(0, 170), breaks = seq(0, 100, 25),
                       expand = expansion(mult = c(0, 0))) +
    labs(x = "Lakes inside the AOA (%)", y = NULL,
         title = "(b) Share inside the AOA") +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          axis.text = element_text(size = 8),
          plot.title = element_text(face = "bold", size = 9.5))

  p <- pa + pb + plot_layout(widths = c(1, 1))
  out <- file.path(out_dir, "Fig07_applicability.png")
  ggsave(out, p, width = 6.85, height = 6.6, dpi = 600, bg = "white")
  ggsave(sub("\\.png$", ".pdf", out), p, width = 6.85, height = 6.6, bg = "white",
         device = if (capabilities("cairo")) cairo_pdf else "pdf")
  message("  saved ", basename(out), " (png 600 dpi + pdf)")
  invisible(out)
}

zones <- data.frame(
  zone = c("peru_north", "bolivia", "peru_south", "patagonia"),
  tag = c("b", "c", "d", "e"),
  title = c("(b) Blanca to Raura", "(c) Apolobamba to C. Real", "(d) Urubamba and Vilcanota",
            "(e) N. Patagonian Icefield"),
  step = c(0.5, 1, 1, 1),
  xmin = c(-77.95, -69.75, -72.75, -74.25), xmax = c(-76.40, -67.55, -70.30, -72.35),
  ymin = c(-10.70, -16.75, -14.30, -48.00), ymax = c(-8.70, -13.70, -12.90, -46.25))

zone_panel <- function(z, lakes, glaciers, frames, legend = FALSE) {
  f <- file.path(root, "data", "interim", "basemap", paste0(z$zone, "_s2_rgb.tif"))
  if (!file.exists(f)) return(NULL)
  img <- rast(f)
  target <- crs(img)
  ex <- ext(img)
  p <- ggplot() + geom_spatraster_rgb(data = img, maxcell = 4e6, interpolate = TRUE)
  if (!is.null(glaciers)) {
    g <- suppressWarnings(st_crop(st_transform(glaciers, target), ex))
    if (nrow(g)) p <- p + geom_sf(data = g, fill = NA, colour = "#bfe6ff", linewidth = 0.12)
  }
  fr <- suppressWarnings(st_crop(st_transform(frames, target), ex))
  if (nrow(fr)) p <- p + geom_sf(data = fr, fill = NA, colour = "white", linewidth = 0.35,
                                 linetype = "22")
  if (!is.null(lakes)) {
    l <- suppressWarnings(st_crop(st_transform(lakes, target), ex))
    l <- l[order(l$oof_score), ]
    if (nrow(l)) {
      p <- p +
        geom_sf(data = l, aes(fill = oof_score), shape = 21, colour = "black", stroke = 0.1,
                size = 0.9) +
        scale_fill_gradientn(colours = c("#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"),
                             limits = c(0, 1), name = "Susceptibility\nscore",
                             guide = if (legend) guide_colourbar(barwidth = unit(3.5, "cm"),
                                                                 barheight = unit(0.25, "cm"),
                                                                 title.vjust = 0.9)
                                     else "none")
      src <- l[l$glof == 1, ]
      if (nrow(src)) p <- p + geom_sf(data = src, shape = 24, fill = "#ff2a6d", colour = "white",
                                      size = 1.9, stroke = 0.35)
    }
  }
  p + annotation_scale(location = "bl", width_hint = 0.28, height = unit(0.12, "cm"),
                       text_cex = 0.7, text_col = "white", bar_cols = c("white", "grey20"),
                       line_col = "white") +
    annotation_north_arrow(location = "tr", height = unit(0.4, "cm"), width = unit(0.3, "cm"),
                           style = north_arrow_orienteering(text_col = "white", fill = c("white", "grey20"),
                                                            line_col = "white", text_size = 6)) +
    scale_x_continuous(breaks = seq(ceiling(z$xmin / z$step) * z$step, z$xmax, by = z$step)) +
    scale_y_continuous(breaks = seq(ceiling(z$ymin / z$step) * z$step, z$ymax, by = z$step)) +
    coord_sf(crs = target, expand = FALSE) +
    labs(title = z$title) +
    theme_minimal(base_size = 8.5) +
    theme(plot.title = element_text(face = "bold", size = 9),
          panel.background = element_rect(fill = "grey85", colour = NA),
          axis.title = element_blank(), axis.text = element_text(size = 8),
          panel.grid = element_line(colour = alpha("white", 0.35), linewidth = 0.15),
          legend.title = element_text(size = 8.5), legend.text = element_text(size = 8))
}

satellite_figure <- function(glaciers, lakes) {
  bm_file <- file.path(root, "data", "interim", "basemap", "bluemarble_andes.tif")
  ev_file <- file.path(root, "data", "processed", "labeled", "event_match_log.csv")
  if (!file.exists(bm_file)) {
    message("  [skip] satellite figure: run scripts/18_basemap_rgb.py first")
    return(invisible(NULL))
  }
  bm <- rast(bm_file)
  frames <- area_frames()
  ev <- read.csv(ev_file)
  ev <- st_as_sf(ev[is.finite(ev$lon) & is.finite(ev$lat), ], coords = c("lon", "lat"), crs = 4326)
  zb <- do.call(rbind, lapply(seq_len(nrow(zones)), function(i) {
    st_sf(tag = zones$tag[i], geometry = st_as_sfc(st_bbox(c(xmin = zones$xmin[i], ymin = zones$ymin[i],
                                                              xmax = zones$xmax[i], ymax = zones$ymax[i]),
                                                            crs = 4326)))
  }))
  zl <- cbind(st_drop_geometry(zb), st_coordinates(st_centroid(zb)))
  pa <- ggplot() +
    geom_spatraster_rgb(data = bm, maxcell = 6e6, interpolate = TRUE) +
    geom_sf(data = frames, fill = NA, colour = "white", linewidth = 0.3) +
    geom_sf(data = ev[ev$status == "matched", ], shape = 24, fill = "#ff2a6d", colour = "white",
            size = 1.3, stroke = 0.25) +
    geom_sf(data = zb, fill = NA, colour = "#ffd400", linewidth = 0.7) +
    geom_label(data = zl, aes(X + 2.2, Y, label = tag), size = 2.85, fontface = "bold",
               fill = "#ffd400", colour = "black", label.size = 0, label.padding = unit(0.12, "lines")) +
    annotation_scale(location = "tl", width_hint = 0.4, height = unit(0.1, "cm"), text_cex = 0.7,
                     text_col = "white", bar_cols = c("white", "grey20"), line_col = "white") +
    scale_x_continuous(breaks = c(-80, -70)) +
    coord_sf(xlim = c(-82, -60), ylim = c(-50, 0), expand = FALSE) +
    labs(title = "(a) Study areas") +
    theme_minimal(base_size = 8.5) +
    theme(plot.title = element_text(face = "bold", size = 9), axis.title = element_blank(),
          axis.text = element_text(size = 8),
          panel.grid = element_line(colour = alpha("white", 0.3), linewidth = 0.15))
  pz <- lapply(seq_len(nrow(zones)), function(i) zone_panel(zones[i, ], lakes, glaciers, frames,
                                                            legend = i == 2))
  pz <- pz[!vapply(pz, is.null, logical(1))]
  if (length(pz) < 4) {
    message("  [skip] satellite figure: ", 4 - length(pz), " zone basemap(s) missing")
    return(invisible(NULL))
  }
  right <- wrap_plots(pz, ncol = 2) + plot_layout(guides = "collect") &
    theme(legend.position = "bottom", legend.direction = "horizontal")
  p <- (pa | right) + plot_layout(widths = c(0.78, 2))
  out <- file.path(out_dir, "Fig01_study_area.png")
  ggsave(out, p, width = 6.85, height = 6.2, dpi = 600, bg = "white")
  ggsave(sub("\\.png$", ".pdf", out), p, width = 6.85, height = 6.2, bg = "white",
         device = if (capabilities("cairo")) cairo_pdf else "pdf")
  message("  saved ", basename(out), " (satellite version, png 600 dpi + pdf)")
  invisible(out)
}

main <- function() {
  aoa_map()
  areas <- sub("_dem_utm\\.tif$", "", basename(Sys.glob(file.path(dem_dir, "*_dem_utm.tif"))))
  if (!length(areas)) {
    message("no DEMs found; run scripts/03_process_dem.py first")
    return(invisible(NULL))
  }

  lakes <- lake_points()
  if (is.null(lakes)) message("no oof_scores.csv yet: drawing terrain and ice only")

  glaciers <- NULL
  if (file.exists(rgi_path)) glaciers <- st_read(rgi_path, quiet = TRUE)

  if (is.null(satellite_figure(glaciers, lakes))) study_area_map(glaciers, lakes)
  message("regions: ", length(areas))
  for (a in areas) region_map(a, lakes, glaciers)
  message("done -> ", out_dir)
}

if (!requireNamespace("ggnewscale", quietly = TRUE)) {
  install.packages("ggnewscale", repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages(library(ggnewscale))

main()
