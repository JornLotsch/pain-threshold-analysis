###############################################################################
# Pain Thresholds Correlation Analysis
# 
# This script loads prepared pain threshold data (already renamed,
# includes Target and Pressure2), then analyzes and visualizes correlations 
# using ComplexHeatmap with grouped variable annotations.
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

###############################################################################
# Configuration Parameters 
###############################################################################

PREPARED_DATA_FILE_PATH <- "/home/joern/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/Pheno_125_prepared_data.csv"
SEED                <- 42              # Seed for reproducibility
CORRELATION_METHOD  <- "pearson"       # Correlation calculation method
CORRELATION_LIMIT   <- 0.9             # Threshold for correlation strength
stepwise_corr_colors <- TRUE           # Use stepwise correlation coloring flag

DATASET_NAME <- "Pheno_125"
EXPERIMENTS_DIR <- "/home/joern/.Datenplatte/Joerns Dateien/Aktuell/ABCPython/08AnalyseProgramme/R/ABC2way/"

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
# Load data and modify files when needed
###############################################################################

# --- Stimulus Types Grouping ---
stimulus_types <- list(
  "Mechanical"        = c("Pressure", "vonFrey", "vonFrey_Capsaicin", "Pressure2"),
  "Thermal"           = c("Heat", "Heat_Capsaicin", "Cold", "Cold_Menthol"),
  "Electrical"        = c("Current"),
  "SensitizationEffect" = c("Capsaicin_Effect_Heat", "Menthol_Effect_Cold", "Capsaicin_Effect_vonFrey")
)

# --- Load Prepared Data ---
prepared_data <- read.csv(PREPARED_DATA_FILE_PATH, row.names = 1)

# Split into predictors and target if needed
target_data <- prepared_data$Target
pain_data   <- prepared_data[, !(colnames(prepared_data) %in% "Target")]

###############################################################################
# --- Correlation Calculation ---
###############################################################################

corr_mat          <- cor(pain_data[, !names(pain_data) %in% c("Pressure2")], method = CORRELATION_METHOD)
corr_mat_modifed  <- cor(pain_data, method = CORRELATION_METHOD)

###############################################################################
# --- Color Mapping Setup for Heatmap ---
###############################################################################

# Function to determine text color based on background fill brightness
text_color_fun <- function(fill_color) {
  rgb_val <- col2rgb(fill_color) / 255
  brightness <- 0.299 * rgb_val[1,] + 0.587 * rgb_val[2,] + 0.114 * rgb_val[3,]
  ifelse(brightness > 0.6, "#111111", "#FFFFFF")  # dark text on light bg, white text on dark bg
}

if (stepwise_corr_colors) {
  # Define breakpoints and stepwise range colors for correlation
  breaks <- c(0, 0.05, 0.1, 0.2, 0.3, 0.4, 1)
  # range_colors <- c(
  #   "ghostwhite",
  #   "#eaeef6",  # Tiny
  #   "#cfd8e7",  # Very small
  #   "#b3c3d6",  # Small
  #   "#97adc8",  # Medium
  #   "#7b98ba",  # Large
  #   "#1a428c"   # Very large
  # )
  range_colors <- c(
    colorRampPalette(c("ghostwhite", "dodgerblue2"))(length(breaks)-1),
    "dodgerblue4")
  # range_colors <- c(
  #   colorRampPalette(c("ghostwhite", "gold"))(length(breaks)-1),
  #   "darkorange2")
  col_fun <- circlize::colorRamp2(breaks, range_colors)
} else {
  # More granular breaks with NYT-inspired color palette (soft blues)
  breaks <- c(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9)
  nyt_colors <- c(
    "ghostwhite",
    "#fbfbfb",
    "#e6f0fa",
    "#c9def9",
    "#add0fa",
    "#7bb8fa",
    "dodgerblue2",
    "#041a58"
  )
  color_vec <- colorRampPalette(nyt_colors)(length(breaks))
  col_fun <- colorRamp2(breaks, color_vec)
}

###############################################################################
# --- Row Annotation Setup ---
###############################################################################

# Assign group membership to each variable for annotation
stim_category <- rep(NA, nrow(corr_mat_modifed))
names(stim_category) <- rownames(corr_mat_modifed)
for (type in names(stimulus_types)) {
  stim_category[stimulus_types[[type]]] <- type
}

# Color palette options for stimulus groups

# Colorblind-friendly palette (ggthemes)
cb_colors <- ggthemes::colorblind_pal()(8)
cb_category_names <- c("Mechanical", "Thermal", "Electrical", "SensitizationEffect")
cb_group_colors <- setNames(cb_colors[1:4], cb_category_names)

# Viridis palette alternative
vir_group_colors <- setNames(viridis::viridis(4, option = "D"), cb_category_names)

# Define row annotation with stimulus groups; disable default legend drawing
row_ha <- rowAnnotation(
  StimulusType = stim_category,
  col          = list(StimulusType = vir_group_colors),
  show_legend  = FALSE
)
# Define row annotation with stimulus groups; disable default legend drawing
column_ha <- HeatmapAnnotation(
  StimulusType = stim_category,
  col          = list(StimulusType = vir_group_colors),
  show_legend  = FALSE, 
  show_annotation_name = FALSE
)
# Create a stand-alone legend for stimulus groups to draw manually
ann_legend <- Legend(
  labels     = names(cb_group_colors),
  legend_gp  = gpar(fill = vir_group_colors),
  title      = "Stimulus Type"
)

###############################################################################
# --- Heatmap Creation and Plotting Function ---
###############################################################################

create_heatmap <- function() {
  
  ht <- Heatmap(
    as.matrix(abs(corr_mat_modifed)),
    col = col_fun,
    cluster_rows = TRUE,
    clustering_method_rows = "ward.D2",
    show_row_dend = TRUE,
    cluster_columns = TRUE,
    clustering_method_columns = "ward.D2",
    show_column_dend = FALSE,
    row_dend_width = unit(4, "cm"),
    column_dend_height = unit(4, "cm"),
    row_names_side = "left",
    
    # Add numeric correlation values inside heatmap cells with dynamic text color
    cell_fun = function(j, i, x, y, w, h, fill_col) {
      val <- round(corr_mat_modifed[i, j], 2)
      col_text <- text_color_fun(fill_col)
      grid.text(val, x, y, gp = gpar(fontsize = 10, fontfamily = "Arial", col = col_text))
    },
    
    left_annotation = row_ha,               # Add colored annotation bar for groups
    bottom_annotation = column_ha, # Add colored annotation bar for groups
    show_heatmap_legend = FALSE,            # Suppress default heatmap correlation legend
    border = FALSE,
    rect_gp = gpar(col = NA)                 # Remove heatmap cell borders
  )
  
  grid.newpage()                           # Start with a fresh graphics page
  
  draw(ht)                                # Draw heatmap with annotation
  
  # Draw annotation legend manually at lower-left corner with padding
  grid::pushViewport(grid::viewport())
  draw(ann_legend,
       x = unit(0, "npc") + unit(7, "mm"),   # 7mm from left edge
       y = unit(0, "npc") + unit(10, "mm"),  # 10mm from bottom edge
       just = c("left", "bottom"))
  
  # Add a left-aligned title relative to the entire plotting device
  grid.text(
    "Correlation matrix",
    x = unit(0, "npc") + unit(4, "mm"),
    y = unit(1, "npc") - unit(4, "mm"),
    just = c("left", "top"),
    gp = gpar(fontsize = 13, fontface = "plain", fontfamily = "Arial", col = "#222222")
  )
}

###############################################################################
# --- Export Plot as SVG ---
###############################################################################

# Capture the heatmap graphic output as a grid object for export
gp <- grid.grabExpr(create_heatmap())
grid.draw(gp)

# Export the plot to an SVG file with specified dimensions
svg(paste0(DATASET_NAME, "_correaltion_heatmap.svg"), width = 13, height = 12)
grid.draw(gp)
dev.off()
