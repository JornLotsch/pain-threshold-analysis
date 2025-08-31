###############################################################################
# Pain Thresholds VIF Analysis with Extended Status Including NA Results
#
# This script loads *prepared pain threshold data* (already renamed,
# includes target and noise column), performs VIF analysis accounting for
# colinearity, aliasing, and additional NA results in coefficient p-values,
# and visualizes the results as a clustered heatmap with detailed statuses.
###############################################################################

# --- Libraries ---
library(reshape2)
library(dplyr)
library(corrplot)
library(ComplexHeatmap)
library(circlize)
library(grid)
library(ggthemes)
library(viridis)
library(car)

###############################################################################
# Configuration Parameters 
###############################################################################

# --- Config ---
PREPARED_DATA_FILE_PATH <- "/home/joern/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/Pheno_125_prepared_data.csv"
SEED                <- 42
VIF_LIMIT           <- 10
significance_Level  <- 0.05

DATASET_NAME <- "Atom"
EXPERIMENTS_DIR <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/"
# External functions
FUNCTIONS_FILE_PATH <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/feature_selection_and_classification_functions.R"

stimulus_types <- list(
  "Mechanical"          = c("Pressure", "vonFrey", "vonFrey_Capsaicin", "Pressure2"),
  "Thermal"             = c("Heat", "Heat_Capsaicin", "Cold", "Cold_Menthol"),
  "Electrical"          = c("Current"),
  "SensitizationEffect" = c("Capsaicin_Effect_Heat", "Menthol_Effect_Cold", "Capsaicin_Effect_vonFrey")
)

###############################################################################
# Load External Functions and set actual path
###############################################################################
if (file.exists(FUNCTIONS_FILE_PATH)) {
  source(FUNCTIONS_FILE_PATH)
} else {
  stop(paste("Functions file not found:", FUNCTIONS_FILE_PATH))
}

set_working_directory(EXPERIMENTS_DIR)

###############################################################################
# Load prepared data and further process them
###############################################################################

prepared_data <- read.csv(PREPARED_DATA_FILE_PATH, row.names = 1)

# Split into pain data (features) and target
target_data <- prepared_data$Target
pain_data   <- prepared_data[, !(colnames(prepared_data) %in% "Target")]

# --- VIF Candidates and Subset Preparation ---
VIF_candidate_var_names <- c("Capsaicin_Effect_Heat", "Menthol_Effect_Cold", 
                             "Capsaicin_Effect_vonFrey", "Pressure2")

subsets_to_remove <- lapply(1:length(VIF_candidate_var_names), function(n) 
  combn(VIF_candidate_var_names, n, simplify=FALSE))
subsets_to_remove <- unlist(subsets_to_remove, recursive=FALSE)
subsets_to_remove <- append(subsets_to_remove, "")  # baseline with no vars removed

df_orig_4_lr <- prepared_data

df_orig_4_lr_cleaned <- df_orig_4_lr[, !names(df_orig_4_lr) %in% VIF_candidate_var_names]
lr_res_cleaned <- glm(Target ~ ., data = df_orig_4_lr_cleaned, family = binomial)
p_values_cleaned <- summary(lr_res_cleaned)$coefficients[, 4]
significant_vars_cleaned <- names(p_values_cleaned)[p_values_cleaned < significance_Level]

df_orig_4_lr_alternative <- df_orig_4_lr[, names(df_orig_4_lr) %in% c(VIF_candidate_var_names, "Target")]
lr_res_alternative <- glm(Target ~ ., data = df_orig_4_lr_alternative, family = binomial)
p_values_alternative <- summary(lr_res_alternative)$coefficients[, 4]
significant_vars_alternative <- names(p_values_alternative)[p_values_alternative < significance_Level]
significant_vars_cleaned <- append(significant_vars_cleaned, significant_vars_alternative)
significant_vars_cleaned <- significant_vars_cleaned[!significant_vars_cleaned %in% "(Intercept)"]

###############################################################################
# Main analysis of aliased variables and VIF
###############################################################################

# --- Extended VIF and Aliasing Analysis Including NA Results ---
aliased <- lapply(seq_along(subsets_to_remove), function(i) {
  
  aliased_vars <- c()
  colinear_vars <- c()
  colinear_not_checked <- c()
  wrong_result <- c()
  vars_in <- c()
  
  df_orig_4_lr_i <- df_orig_4_lr[, !names(df_orig_4_lr) %in% subsets_to_remove[[i]]]
  vars_in <- setdiff(names(df_orig_4_lr_i), "Target")
  
  lr_res_i <- glm(Target ~ ., data = df_orig_4_lr_i, family = binomial)
  print(summary(lr_res_i))
  
  p_values <- summary(lr_res_i)$coefficients[, 4]
  significant_vars <- names(p_values)[p_values < significance_Level]
  
  wrong_result <- unique(c(
    setdiff(significant_vars, significant_vars_cleaned),
    setdiff(significant_vars_cleaned, significant_vars)
  ))
  
  aliased_check <- try(alias(lr_res_i)$Complete, silent = TRUE)
  if (!inherits(aliased_check, "try-error")) aliased_vars <- rownames(aliased_check)
  
  if (!inherits(aliased_check, "try-error") && is.null(aliased_check)) {
    vif <- car::vif(lr_res_i)
    colinear_vars <- names(which(vif > VIF_LIMIT))
  } else {
    colinear_not_checked <- subsets_to_remove[[i]]
  }
  
  # Capture the printed summary output as text lines
  summary_text <- capture.output(summary(lr_res_i))
  
  # Locate the lines showing coefficients with NA (e.g the NA rows)
  na_lines <- grep("NA +NA +NA +NA", summary_text, value = TRUE)
  
  # Extract variable names from those lines (assuming variable name is first word in line)
  NA_results <- as.vector(unlist(sapply(strsplit(na_lines, " "), function(x) x[1])))
  
  list(
    colinear_vars = colinear_vars,
    colinear_not_checked = if (identical(colinear_not_checked, "")) NULL else colinear_not_checked,
    aliased_vars = aliased_vars,
    wrong_result = wrong_result,
    vars_in = vars_in,
    vars_removed = if (identical(subsets_to_remove[[i]], "")) NULL else subsets_to_remove[[i]],
    NA_results = NA_results
  )
  
})

###############################################################################
# --- Plot Results ---
###############################################################################

# --- Construct Matrix for Heatmap Visualization ---
all_vars <- colnames(pain_data)
n_subsets <- length(aliased)

mat <- matrix(0, nrow = n_subsets, ncol = length(all_vars))
rownames(mat) <- sapply(subsets_to_remove, function(x) 
  if(length(x) == 0) "All variables" else paste(x, collapse = "\n"))
colnames(mat) <- all_vars

# Priority order for matrix values:
# 6 = NA result, 5 = wrong result, 4 = aliased, 3 = colinear, 
# 2 = colinear not checked, 1 = included & ok, 0 = removed

for (i in seq_len(n_subsets)) {
  print(i)
  res <- aliased[[i]]
  mat[i,res$vars_in] <- 1
  mat[i,res$colinear_not_checked] <- 2
  mat[i,res$colinear_vars] <- 3
  mat[i,res$aliased_vars] <- 4
  mat[i,res$wrong_result] <- 5
  #mat[i,res$NA_results] <- 6
  mat[i,res$vars_removed] <- 0
}

# --- Visualization Setup and Plotting ---

column_group <- ifelse(colnames(mat) %in% VIF_candidate_var_names, "VIF_candidate", "Other")
column_group <- factor(column_group, levels = c("VIF_candidate", "Other"))

col_annotation_df <- data.frame(
  VIF_candidate = factor(ifelse(colnames(mat) %in% VIF_candidate_var_names, "yes", "no"),
                         levels = c("yes", "no"))
)
col_annot_colors <- list(VIF_candidate = c(yes = "forestgreen", no = "lightgray"))
column_ha <- HeatmapAnnotation(
  df = col_annotation_df,
  col = col_annot_colors,
  show_legend = FALSE,
  show_annotation_name = FALSE
)

colorblind <- ggthemes::colorblind_pal()(8)

my_colors <- c("ghostwhite", "lightskyblue1", colorblind[5], colorblind[2], "grey77", "chartreuse3", "grey7")
#my_colors <- ggthemes::colorblind_pal()(8)[2:8]
col_fun <- circlize::colorRamp2(
  c(0, 1, 2, 3, 4, 5, 6), my_colors
)

status_legend <- Legend(
  labels = c("removed", "OK", "colinear not checked", "colinear", "aliased", "wrong results", "NA results"),
  title = "Variable status",
  legend_gp = gpar(fill = my_colors),
  direction = "horizontal"
)

create_clustered_heatmap <- function() {
  rownames(mat) <- gsub(",", "\n", rownames(mat))
  
  ht <- Heatmap(
    mat,
    name = "Variable Status",
    col = col_fun,
    cluster_rows = FALSE,
    clustering_method_rows = "ward.D2",
    cluster_columns = FALSE,
    clustering_method_columns = "ward.D2",
    column_split = column_group,
    cluster_column_slices = TRUE,
    show_row_dend = TRUE,
    show_column_dend = TRUE,
    row_dend_width = unit(6, "cm"),
    row_dend_side = "left",
    column_dend_height = unit(4, "cm"),
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 9),
    column_names_side = "bottom",
    border = F,
    column_gap = unit(3, "mm"),
    rect_gp = gpar(col = "grey60", lwd = 0.5),
    top_annotation = NULL,
    bottom_annotation = column_ha,
    cell_fun = NULL,
    show_heatmap_legend = FALSE
  )
  
  grid.newpage()
  draw(ht)
  
  grid::pushViewport(grid::viewport())
  draw(status_legend,
       x = unit(0, "npc") + unit(7, "mm"),
       y = unit(0, "npc") + unit(10, "mm"),
       just = c("left", "bottom"))
  
  grid.text(
    "Variable status heatmap",
    x = unit(0, "npc") + unit(4, "mm"),
    y = unit(1.01, "npc") - unit(4, "mm"),
    just = c("left", "top"),
    gp = gpar(fontsize = 14, fontface = "plain", fontfamily = "Arial", col = "#222222")
  )
}


###############################################################################
# --- Export Plot as SVG ---
###############################################################################

# Capture the heatmap graphic output as a grid object for export
gp_VIF <- grid.grabExpr(create_clustered_heatmap())
grid.draw(gp_VIF)

# Export the plot to an SVG file with specified dimensions
svg("Pheno_125_LR_VIF.svg", width = 13, height = 12)
grid.draw(gp_VIF)
dev.off()
