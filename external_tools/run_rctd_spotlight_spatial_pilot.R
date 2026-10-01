suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
  library(SPOTlight)
  library(ggplot2)
  library(data.table)
})

analysis_ready <- Sys.getenv("SC_HGNN_ANALYSIS_READY", unset = "")
if (!nzchar(analysis_ready)) stop("Set SC_HGNN_ANALYSIS_READY to the prepared bundle root")
input_dir <- file.path(analysis_ready, "spatial_mapping", "pilot_inputs")
output_dir <- file.path(analysis_ready, "spatial_mapping", "pilot_results", "rctd_spotlight")
manifest_path <- file.path(input_dir, "spatial_mapping_pilot_manifest.csv")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

main_celltypes <- c(
  "B_cell",
  "Endothelial",
  "Fibroblast_CAF",
  "Myeloid",
  "Plasma_cell",
  "T_NK_ILC",
  "Tumor_Lineage"
)

run_samples_env <- Sys.getenv("SPATIAL_PILOT_SAMPLES", unset = "")
run_samples <- if (nzchar(run_samples_env)) trimws(strsplit(run_samples_env, ",", fixed = TRUE)[[1]]) else character(0)
max_cores <- as.integer(Sys.getenv("SPATIAL_PILOT_CORES", unset = "8"))
if (is.na(max_cores) || max_cores < 1) max_cores <- 4
rctd_mode <- Sys.getenv("SPATIAL_RCTD_MODE", unset = "full")
if (!rctd_mode %in% c("full", "multi", "doublet")) rctd_mode <- "full"

read_bundle <- function(dir_path, item_label) {
  counts <- Matrix::readMM(file.path(dir_path, "counts.mtx"))
  genes <- readLines(file.path(dir_path, "genes.tsv"), warn = FALSE, encoding = "UTF-8")
  items <- readLines(file.path(dir_path, paste0(item_label, "s.tsv")), warn = FALSE, encoding = "UTF-8")
  genes <- make.unique(trimws(genes), sep = "__dup")
  items <- trimws(items)
  counts <- as(counts, "CsparseMatrix")
  rownames(counts) <- genes
  colnames(counts) <- items
  counts
}

normalize_rows <- function(mat) {
  mat <- as.matrix(mat)
  mat[is.na(mat)] <- 0
  mat[mat < 0] <- 0
  rs <- rowSums(mat)
  keep <- rs > 0
  mat[keep, ] <- mat[keep, , drop = FALSE] / rs[keep]
  mat
}

align_gene_space <- function(ref_counts, spatial_counts) {
  common <- intersect(rownames(ref_counts), rownames(spatial_counts))
  if (length(common) < 1000) {
    stop("Too few common genes between reference and spatial matrices: ", length(common))
  }
  ref_counts <- ref_counts[common, , drop = FALSE]
  spatial_counts <- spatial_counts[common, , drop = FALSE]
  keep <- Matrix::rowSums(ref_counts) >= 20 & Matrix::rowSums(spatial_counts) >= 20
  ref_counts <- ref_counts[keep, , drop = FALSE]
  spatial_counts <- spatial_counts[keep, , drop = FALSE]
  list(ref = ref_counts, spatial = spatial_counts, n_common_before_filter = length(common))
}

make_marker_table <- function(ref_counts, labels, spatial_counts, top_n = 100) {
  candidate_genes <- intersect(rownames(ref_counts), rownames(spatial_counts))
  ref_counts <- ref_counts[candidate_genes, , drop = FALSE]
  out <- list()
  for (ct in sort(unique(labels))) {
    in_cells <- labels == ct
    out_cells <- !in_cells
    if (sum(in_cells) < 25 || sum(out_cells) < 25) next
    mean_in <- Matrix::rowMeans(ref_counts[, in_cells, drop = FALSE])
    mean_out <- Matrix::rowMeans(ref_counts[, out_cells, drop = FALSE])
    pct_in <- Matrix::rowMeans(ref_counts[, in_cells, drop = FALSE] > 0)
    score <- log1p(mean_in) - log1p(mean_out)
    keep <- is.finite(score) & score > 0 & pct_in >= 0.05 & mean_in > 0
    genes <- names(sort(score[keep], decreasing = TRUE))
    genes <- head(genes, top_n)
    if (length(genes)) {
      out[[ct]] <- data.frame(
        gene = genes,
        cluster = ct,
        weight = as.numeric(score[genes]),
        pct_in = as.numeric(pct_in[genes]),
        mean_in = as.numeric(mean_in[genes]),
        mean_out = as.numeric(mean_out[genes]),
        stringsAsFactors = FALSE
      )
    }
  }
  markers <- data.table::rbindlist(out, fill = TRUE)
  markers <- as.data.frame(markers)
  if (!nrow(markers)) stop("No marker genes generated for SPOTlight")
  markers
}

extract_rctd_weights <- function(rctd, spot_ids, celltypes) {
  res <- rctd@results
  actual_spot_ids <- colnames(rctd@spatialRNA@counts)
  if (!is.null(actual_spot_ids) && length(actual_spot_ids)) {
    spot_ids <- actual_spot_ids
  }
  candidates <- list()
  if (!is.null(res$weights)) candidates[["weights"]] <- res$weights
  if (!is.null(res$weights_doublet)) candidates[["weights_doublet"]] <- res$weights_doublet
  if (length(candidates)) {
    w <- candidates[[1]]
  } else if (is.list(res) && length(res) == length(spot_ids) && all(vapply(res, is.list, logical(1)))) {
    w <- matrix(0, nrow = length(spot_ids), ncol = length(celltypes), dimnames = list(spot_ids, celltypes))
    for (i in seq_along(res)) {
      entry <- res[[i]]
      vec <- entry$sub_weights
      if (is.null(vec)) vec <- entry$all_weights
      if (is.null(vec)) next
      vec <- as.numeric(vec)
      vec_names <- names(entry$sub_weights)
      if (is.null(vec_names) && !is.null(entry$cell_type_list) && length(entry$cell_type_list) == length(vec)) {
        vec_names <- as.character(entry$cell_type_list)
      }
      if (is.null(vec_names) && length(vec) == length(celltypes)) {
        vec_names <- celltypes
      }
      if (is.null(vec_names)) next
      names(vec) <- vec_names
      overlap <- intersect(names(vec), celltypes)
      w[i, overlap] <- vec[overlap]
    }
  } else {
    stop("RCTD object does not contain recognizable weights")
  }
  w <- as.matrix(w)
  if (nrow(w) == length(celltypes) && ncol(w) == length(spot_ids)) {
    w <- t(w)
  }
  if (nrow(w) != length(spot_ids) && ncol(w) == length(spot_ids)) {
    w <- t(w)
  }
  rownames(w) <- spot_ids
  missing <- setdiff(celltypes, colnames(w))
  if (length(missing)) {
    add <- matrix(0, nrow = nrow(w), ncol = length(missing), dimnames = list(rownames(w), missing))
    w <- cbind(w, add)
  }
  w <- w[, celltypes, drop = FALSE]
  normalize_rows(w)
}

write_weights <- function(weights, path) {
  dt <- data.frame(spot_id = rownames(weights), weights, check.names = FALSE)
  data.table::fwrite(dt, path)
}

read_weights <- function(path) {
  dt <- data.table::fread(path)
  rn <- dt$spot_id
  dt$spot_id <- NULL
  mat <- as.matrix(dt)
  rownames(mat) <- rn
  mat
}

plot_celltype_maps <- function(weights, coords, out_png, title_prefix) {
  plot_df <- merge(
    coords[, c("spot_id", "x", "y"), drop = FALSE],
    data.frame(spot_id = rownames(weights), weights, check.names = FALSE),
    by = "spot_id",
    all = FALSE
  )
  long <- data.table::melt(
    data.table::as.data.table(plot_df),
    id.vars = c("spot_id", "x", "y"),
    variable.name = "celltype",
    value.name = "proportion"
  )
  p <- ggplot(long, aes(x = x, y = y, color = proportion)) +
    geom_point(size = 0.45, alpha = 0.9) +
    facet_wrap(~celltype, ncol = 4) +
    scale_color_viridis_c(option = "magma", limits = c(0, max(long$proportion, na.rm = TRUE))) +
    scale_y_reverse() +
    coord_fixed() +
    labs(title = title_prefix, x = NULL, y = NULL, color = "Weight") +
    theme_bw(base_size = 9) +
    theme(
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      strip.background = element_rect(fill = "grey92", color = NA),
      plot.title = element_text(face = "bold", size = 11)
    )
  ggsave(out_png, p, width = 10, height = 7, dpi = 300)
}

plot_dominant_map <- function(weights, coords, out_png, title_prefix) {
  dominant <- colnames(weights)[max.col(weights, ties.method = "first")]
  score <- apply(weights, 1, max)
  plot_df <- merge(
    coords[, c("spot_id", "x", "y"), drop = FALSE],
    data.frame(spot_id = rownames(weights), dominant_celltype = dominant, max_weight = score, check.names = FALSE),
    by = "spot_id",
    all = FALSE
  )
  p <- ggplot(plot_df, aes(x = x, y = y, color = dominant_celltype, alpha = max_weight)) +
    geom_point(size = 0.75) +
    scale_y_reverse() +
    coord_fixed() +
    labs(title = title_prefix, x = NULL, y = NULL, color = "Dominant", alpha = "Weight") +
    theme_bw(base_size = 9) +
    theme(panel.grid = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), plot.title = element_text(face = "bold", size = 11))
  ggsave(out_png, p, width = 7, height = 6, dpi = 300)
}

calculate_agreement <- function(rctd_w, spotlight_w) {
  common_spots <- intersect(rownames(rctd_w), rownames(spotlight_w))
  celltypes <- intersect(colnames(rctd_w), colnames(spotlight_w))
  out <- lapply(celltypes, function(ct) {
    a <- rctd_w[common_spots, ct]
    b <- spotlight_w[common_spots, ct]
    cutoff_a <- stats::quantile(a, 0.8, na.rm = TRUE)
    cutoff_b <- stats::quantile(b, 0.8, na.rm = TRUE)
    hot_a <- a >= cutoff_a
    hot_b <- b >= cutoff_b
    jaccard <- sum(hot_a & hot_b) / max(1, sum(hot_a | hot_b))
    data.frame(
      celltype = ct,
      n_spots = length(common_spots),
      spearman = suppressWarnings(stats::cor(a, b, method = "spearman", use = "pairwise.complete.obs")),
      pearson = suppressWarnings(stats::cor(a, b, method = "pearson", use = "pairwise.complete.obs")),
      top20_jaccard = jaccard,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

run_one_sample <- function(row) {
  sample_id <- row[["sample_id"]]
  cancer <- row[["cancer"]]
  message("[sample] ", sample_id)
  sample_out <- file.path(output_dir, sample_id)
  dir.create(sample_out, recursive = TRUE, showWarnings = FALSE)

  ref_counts <- read_bundle(row[["reference_dir"]], "cell")
  spatial_counts <- read_bundle(row[["spatial_dir"]], "spot")
  labels <- data.table::fread(row[["reference_labels_csv"]])
  coords <- data.table::fread(row[["spatial_coords_csv"]])
  labels <- as.data.frame(labels)
  coords <- as.data.frame(coords)
  labels <- labels[match(colnames(ref_counts), labels$reference_cell_id), , drop = FALSE]
  if (anyNA(labels$reference_celltype)) stop("Missing reference cell type labels for ", sample_id)
  celltypes <- intersect(main_celltypes, sort(unique(labels$reference_celltype)))

  aligned <- align_gene_space(ref_counts, spatial_counts)
  ref_counts <- aligned$ref
  spatial_counts <- aligned$spatial
  labels_vec <- factor(labels$reference_celltype, levels = celltypes)
  keep_cells <- !is.na(labels_vec)
  ref_counts <- ref_counts[, keep_cells, drop = FALSE]
  labels_vec <- droplevels(labels_vec[keep_cells])
  names(labels_vec) <- colnames(ref_counts)

  coords <- coords[match(colnames(spatial_counts), coords$spot_id), , drop = FALSE]
  if (anyNA(coords$spot_id)) stop("Missing spatial coordinates for ", sample_id)
  spatial_coords <- data.frame(
    x = as.numeric(coords$x),
    y = as.numeric(coords$y),
    row.names = coords$spot_id
  )

  rctd_weight_path <- file.path(sample_out, "rctd_weights.csv.gz")
  force_rctd <- Sys.getenv("SPATIAL_FORCE_RCTD", unset = "0") == "1"
  if (file.exists(rctd_weight_path) && !force_rctd) {
    message("[RCTD] ", sample_id, " using existing weights")
    rctd_weights <- read_weights(rctd_weight_path)
  } else {
    message("[RCTD] ", sample_id, " genes=", nrow(ref_counts), " cells=", ncol(ref_counts), " spots=", ncol(spatial_counts))
    reference <- Reference(ref_counts, labels_vec, nUMI = Matrix::colSums(ref_counts))
    spatial <- SpatialRNA(spatial_coords, spatial_counts, nUMI = Matrix::colSums(spatial_counts))
    rctd <- create.RCTD(
      spatial,
      reference,
      max_cores = max_cores,
      CELL_MIN_INSTANCE = 25,
      MAX_MULTI_TYPES = 4,
      UMI_min = 100,
      counts_MIN = 10
    )
    rctd <- run.RCTD(rctd, doublet_mode = rctd_mode)
    saveRDS(rctd, file.path(sample_out, "rctd_object.rds"))
    saveRDS(rctd, file.path(sample_out, paste0("rctd_object_", rctd_mode, ".rds")))
    rctd_weights <- extract_rctd_weights(rctd, colnames(spatial_counts), celltypes)
    write_weights(rctd_weights, rctd_weight_path)
    plot_celltype_maps(rctd_weights, coords, file.path(sample_out, "rctd_celltype_maps.png"), paste0(sample_id, " RCTD"))
    plot_dominant_map(rctd_weights, coords, file.path(sample_out, "rctd_dominant_celltype.png"), paste0(sample_id, " RCTD dominant"))
  }

  message("[SPOTlight] ", sample_id)
  spotlight_weight_path <- file.path(sample_out, "spotlight_weights.csv.gz")
  force_spotlight <- Sys.getenv("SPATIAL_FORCE_SPOTLIGHT", unset = "0") == "1"
  if (file.exists(spotlight_weight_path) && !force_spotlight) {
    message("[SPOTlight] ", sample_id, " using existing weights")
    spotlight_weights <- read_weights(spotlight_weight_path)
  } else {
    markers <- make_marker_table(ref_counts, as.character(labels_vec), spatial_counts, top_n = 100)
    data.table::fwrite(markers, file.path(sample_out, "spotlight_marker_genes.csv"))
    marker_genes <- intersect(unique(markers$gene), intersect(rownames(ref_counts), rownames(spatial_counts)))
    sc_mat <- as.matrix(ref_counts[marker_genes, , drop = FALSE])
    sp_mat <- as.matrix(spatial_counts[marker_genes, , drop = FALSE])
    set.seed(20260503)
    spotlight <- SPOTlight(
      x = sc_mat,
      y = sp_mat,
      groups = as.character(labels_vec),
      mgs = markers,
      gene_id = "gene",
      group_id = "cluster",
      weight_id = "weight",
      model = "ns",
      min_prop = 0.01,
      verbose = TRUE,
      nrun = 1
    )
    spotlight_weights <- as.matrix(spotlight$mat)
    if (nrow(spotlight_weights) != ncol(spatial_counts) && ncol(spotlight_weights) == ncol(spatial_counts)) {
      spotlight_weights <- t(spotlight_weights)
    }
    rownames(spotlight_weights) <- colnames(spatial_counts)
    missing_spotlight <- setdiff(celltypes, colnames(spotlight_weights))
    if (length(missing_spotlight)) {
      add <- matrix(0, nrow = nrow(spotlight_weights), ncol = length(missing_spotlight), dimnames = list(rownames(spotlight_weights), missing_spotlight))
      spotlight_weights <- cbind(spotlight_weights, add)
    }
    spotlight_weights <- spotlight_weights[, celltypes, drop = FALSE]
    spotlight_weights <- normalize_rows(spotlight_weights)
    write_weights(spotlight_weights, spotlight_weight_path)
    saveRDS(spotlight$NMF, file.path(sample_out, "spotlight_nmf_model.rds"))
    plot_celltype_maps(spotlight_weights, coords, file.path(sample_out, "spotlight_celltype_maps.png"), paste0(sample_id, " SPOTlight"))
    plot_dominant_map(spotlight_weights, coords, file.path(sample_out, "spotlight_dominant_celltype.png"), paste0(sample_id, " SPOTlight dominant"))
  }

  agreement <- calculate_agreement(rctd_weights, spotlight_weights)
  data.table::fwrite(agreement, file.path(sample_out, "rctd_spotlight_agreement.csv"))
  summary <- data.frame(
    sample_id = sample_id,
    cancer = cancer,
    platform = row[["platform"]],
    n_reference_cells = ncol(ref_counts),
    n_spots = ncol(spatial_counts),
    n_genes_after_intersection_filter = nrow(ref_counts),
    n_common_genes_before_filter = aligned$n_common_before_filter,
    n_celltypes = length(celltypes),
    rctd_mode = rctd_mode,
    median_spearman = stats::median(agreement$spearman, na.rm = TRUE),
    median_top20_jaccard = stats::median(agreement$top20_jaccard, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  data.table::fwrite(summary, file.path(sample_out, "sample_summary.csv"))
  summary
}

manifest <- data.table::fread(manifest_path)
manifest <- as.data.frame(manifest)
if (length(run_samples)) {
  manifest <- manifest[manifest$cancer %in% run_samples | manifest$sample_id %in% run_samples, , drop = FALSE]
}
if (!nrow(manifest)) stop("No samples selected")

summaries <- list()
for (i in seq_len(nrow(manifest))) {
  summaries[[i]] <- run_one_sample(manifest[i, , drop = FALSE])
}
summary_df <- data.table::rbindlist(summaries, fill = TRUE)
data.table::fwrite(summary_df, file.path(output_dir, "rctd_spotlight_pilot_summary.csv"))
message("[done] wrote ", file.path(output_dir, "rctd_spotlight_pilot_summary.csv"))
