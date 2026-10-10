# Phase 1 — Data foundation (combined script)
import pandas as pd
import numpy as np
import re
from pathlib import Path
# Visualization libraries
import matplotlib.pyplot as plt 
import seaborn as sns 
import io
from typing import Optional, Dict

# ---------- CONFIG ----------
file_path = "/Users/LANKM9/Downloads/MajorProject-main/MajorProject-main/combined_year_dataset_cleaned_ST.xlsx"
output_path = "/Users/LANKM9/Downloads/MajorProject-main/MajorProject-main/dataset_phase1_ready.xlsx"
image_output_dir = "/Users/LANKM9/Downloads/MajorProject-main/MajorProject-main/phase1_visualizations_LLM" # Directory for saving plots

# HARDCODED COLUMN MAPPING 
found = {
    "campus": "Campus",        
    "district": "District",
    "region": "Region",
    "year": "year",
    "sat_erw": "ERW_SAT",
    "sat_math": "Math_SAT",
    "tsi": "Above_TSI_Both_Rate_SAT", 
    "participation": "Part_Rate_SAT",
    "group": "Group",
    "Campus_Type": "Campus_Type",
    "Campus_Type_Description": "Campus_Type_Description",
    "NCES_Campus_Type": "NCES_Campus_Type",
    "NCES_Description": "NCES_Description",
    "Campus_Dropout_2122": "Campus_Dropout_2122",
    "Campus_Dropout_2223": "Campus_Dropout_2223",
    "Campus_Retention_2223": "Campus_Retention_2223"
}

# Masked values to consider (TEA style)
masked_values = ["<25", "*", "-", "NA", "N/A", "na", "n/a", "PrivacySuppressed"]

# ---------- HELPERS ----------'
def sanitize_numeric_column(series):
    """Strip commas, percent signs, parentheses, convert to numeric if possible."""
    def clean_val(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.number)): return x
        s = str(x).strip()
        if s in masked_values: return np.nan
        s = re.sub(r"[,\%]|[\(\)]|\*|†|‡", "", s)
        s = re.sub(r"^[^\d\.-]+", "", s)
        s = re.sub(r"[^\d\.-]+$", "", s)
        if s == "": return np.nan
        try:
            if "." in s or re.search(r"\d+\.\d+", s): return float(s)
            return int(float(s))
        except:
            try: return float(re.sub(r"[^\d\.-]", "", s))
            except: return np.nan
    return series.map(clean_val)

def safe_diff(df, group_col, value_col, out_col):
    """Helper to compute year-over-year difference."""
    if value_col in df.columns:
        if group_col and group_col in df.columns:
            df[out_col] = df.groupby(group_col)[value_col].diff()
        else:
            df[out_col] = df[value_col].diff() 
        return True
    return False


def df_to_image_bytes(df: pd.DataFrame, title: Optional[str] = None, fontsize: int = 10) -> io.BytesIO:
    """Render a DataFrame to a PNG image (BytesIO) using matplotlib table."""
    rows, cols = df.shape
    fig_w = min(14, max(6, 0.8 * max(1, cols)))
    fig_h = min(20, max(2, 0.35 * (rows + 3)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    if title:
        ax.text(0.5, 1.02, title, ha='center', va='bottom', transform=ax.transAxes, fontsize=fontsize+2, weight='bold')
    try:
        table = ax.table(cellText=df.fillna('').values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.2)
    except Exception:
        ax.text(0, 0.5, df.to_string(), fontfamily='monospace', fontsize=fontsize)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def save_df_image(df: pd.DataFrame, out_dir: Path, prefix: str, title: Optional[str] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{prefix}_{ts}.png"
    path = out_dir / fname
    buf = df_to_image_bytes(df, title=title)
    with open(path, 'wb') as f:
        f.write(buf.getvalue())
    return path


def save_df_excel(df: pd.DataFrame, out_dir: Path, prefix: str, sheet_name: Optional[str] = None) -> Path:
    """Save a single DataFrame to an Excel file and return the Path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{prefix}_{ts}.xlsx"
    path = out_dir / fname
    try:
        df.to_excel(path, index=False, sheet_name=(sheet_name or 'Sheet1'))
    except Exception:
        # fallback when sheet_name not supported by to_excel directly
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=(sheet_name or 'Sheet1'))
    return path


def save_multiple_dfs_to_excel(dfs: Dict[str, pd.DataFrame] , out_dir: Path, prefix: str) -> Path:


    """Save multiple DataFrames to a single Excel workbook, each as its own sheet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{prefix}_{ts}.xlsx"
    path = out_dir / fname
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df_sheet in dfs.items():
            try:
                df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            except Exception:
                # coerce to string if cannot write directly
                df_sheet.astype(str).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return path


def show_image(path: Path):
    try:
        img = plt.imread(str(path))
        plt.figure(figsize=(10, max(4, img.shape[0] / img.shape[1] * 10)))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image {path}: {e}")

# ---------- 1) LOAD & INITIAL CHECK ----------
print("Loading:", file_path)
df = pd.read_excel(file_path, engine="openpyxl")
print("Loaded. Shape:", df.shape)
# Save a sample of the loaded input as an Excel sheet
try:
    out_dir = Path(image_output_dir)
    sample_in = df.head(20)[[c for c in df.columns[:min(30, len(df.columns))]]]
    in_xlsx = save_df_excel(sample_in, out_dir, "input_sample", sheet_name="input_sample")
    print(f"Saved input sample Excel: {in_xlsx}")
except Exception as e:
    print(f"Could not save input sample Excel: {e}")

# Define column variables
year_col = found.get("year")
campus_col = found.get("campus")
district_col = found.get("district")
region_col = found.get("region")
group_col = found.get("group")
sat_erw_col = found.get("sat_erw")
sat_math_col = found.get("sat_math")
tsi_col = found.get("tsi")
part_col = found.get("participation")
sat_total_col = "Total_SAT"

# ---------- 2) CLEAN & COERCE NUMERICS ----------
print("\nCleaning and coercing numeric columns...")
df.replace(masked_values, np.nan, inplace=True)
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in obj_cols:
    if col in [found.get(k) for k in ["sat_erw", "sat_math", "tsi", "participation"]]:
        df[col] = sanitize_numeric_column(df[col])
        continue
    non_null = df[col].dropna()
    if len(non_null) > 0 and non_null.astype(str).str.contains(r"\d").mean() >= 0.30:
        df[col] = sanitize_numeric_column(df[col])

for col in ["Campus_Dropout_2122","Campus_Dropout_2223","Campus_Retention_2223"]:
    if col in df.columns:
        df[col] =  sanitize_numeric_column(df[col])
    

# Final coercion for key metrics
for col in [sat_erw_col, sat_math_col, tsi_col, part_col]:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"Coerced {col} -> numeric. Nulls: {df[col].isna().sum()}")

for col in ["Campus_Dropout_2122","Campus_Dropout_2223","Campus_Retention_2223"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"Coerced {col} -> numeric. Nulls: {df[col].isna().sum()}")

# NORMALIZE LOCATION COLUMNS (before grouping)
for col in [campus_col, district_col, region_col, group_col]:
    if col and col in df.columns:
        df[col] = df[col].astype(str).str.strip()

#for col in ["Campus_Type","Campus_Type_Description","NCES_Campus_Type","NCES_Description"]:
#    if col and col in df.columns:
#        df[col] = df[col].astype(str).str.strip()         

# ---------- 3) SORT and determine grouping column for time series ----------
if campus_col and campus_col in df.columns:
    used_group = campus_col
elif district_col and district_col in df.columns:
    used_group = district_col
else:
    used_group = None
sort_cols = [used_group, year_col] if used_group else [year_col]
df = df.sort_values(sort_cols).reset_index(drop=True)

# ---------- 4) CREATE TIME-SERIES FEATURES (for plotting) ----------
print("\nCreating time-series features...")
diffs_created = []
if safe_diff(df, used_group, sat_erw_col, "ERW_change"): diffs_created.append("ERW_change")
# Note: Creating rolling mean columns for use in Section 11, even though the feature columns themselves aren't the final output.
rolling_cols = []
if used_group and used_group in df.columns:
    for col, name in [(sat_erw_col, "ERW_3yr"), (sat_math_col, "Math_3yr"), (tsi_col, "TSI_3yr")]:
        if col and col in df.columns:
            df[name] = df.groupby(used_group)[col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_cols.append(name)
print(f"Time-series features created: {diffs_created + rolling_cols}")

# ---------- 5) PREPARE AGGREGATION DATAFRAMES (for plotting) ----------
print("\nPreparing aggregation dataframes for comparative plots...")
metrics = [sat_erw_col, tsi_col, part_col] # Focus on key metrics for comparisons
agg_dfs = {}

# A. Group-level average scores
if group_col and group_col in df.columns:
    # Average performance for each student group across all years
    agg_dfs['group_avg'] = df.groupby(group_col, dropna=True)[metrics].mean().reset_index()

# B. Campus vs District Comparison (Yearly)
if campus_col and district_col and campus_col in df.columns and district_col in df.columns:
    # Calculate District Average for each year
    district_avg = df.groupby([district_col, year_col], dropna=True)[metrics].mean().reset_index()
    district_avg.columns = [district_col, year_col] + [f'{c}_Avg' for c in metrics]
    agg_dfs['district_avg'] = district_avg

# C. District vs Region Comparison (Yearly)
if district_col and region_col and district_col in df.columns and region_col in df.columns:
    # Calculate Regional Average for each year
    region_avg = df.groupby([region_col, year_col], dropna=True)[metrics].mean().reset_index()
    region_avg.columns = [region_col, year_col] + [f'{c}_Avg' for c in metrics]
    agg_dfs['region_avg'] = region_avg

print(f"Prepared {len(agg_dfs)} aggregation tables for visualization.")

# Save aggregation tables into a single Excel workbook (one sheet per aggregation)
try:
    agg_out = Path(image_output_dir)
    dfs_to_save = {k: (v.reset_index(drop=True) if isinstance(v, pd.DataFrame) else pd.DataFrame()) for k, v in agg_dfs.items()}
    if dfs_to_save:
        agg_xlsx = save_multiple_dfs_to_excel(dfs_to_save, agg_out, "aggregations")
        print(f"Saved aggregations workbook: {agg_xlsx}")
except Exception as e:
    print(f"Could not save aggregation tables to Excel: {e}")

# --- 6) VISUALIZATION AND SAVING ---
Path(image_output_dir).mkdir(parents=True, exist_ok=True)
print(f"\nGenerating and saving visualizations to: {image_output_dir}")

sns.set_style("whitegrid")
saved_plots = []

# --- PLOT 1: 3-YEAR ROLLING TRENDS (Line Plots) ---
for col in rolling_cols:
    plt.figure(figsize=(10, 6))
    
    # Calculate overall trend
    avg_df = df.groupby(year_col, dropna=True)[col].mean().reset_index()
    
    # Plot individual group trends (sampled for clarity)
    if used_group and used_group in df.columns:
        sample_groups = np.random.choice(df[used_group].dropna().unique(), min(10, len(df[used_group].dropna().unique())), replace=False)
        sample_df = df[df[used_group].isin(sample_groups)]
        sns.lineplot(x=year_col, y=col, hue=used_group, data=sample_df, 
                     linestyle='--', alpha=0.3, legend=False, palette='tab10')
    
    # Plot the overall trend
    sns.lineplot(x=year_col, y=col, data=avg_df, 
                 marker='o', linestyle='-', color='red', linewidth=3, label='Overall Average Trend')
    
    plt.title(f'3-Year Rolling Trend: {col}')
    plt.xlabel(year_col)
    plt.ylabel(col)
    file_name = f'Trend_1_{col}.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)


# --- PLOT 2: GROUP PERFORMANCE COMPARISON (Bar Plot/Distribution) ---
if 'group_avg' in agg_dfs:
    plt.figure(figsize=(12, 6))
    # Focus on ERW for simplicity
    sns.barplot(x=group_col, y=sat_erw_col, data=agg_dfs['group_avg'], palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Overall Average {sat_erw_col} Score by Student Group')
    plt.ylabel(f'Average {sat_erw_col} Score')
    plt.xlabel(group_col)
    plt.tight_layout()
    file_name = f'Comparison_2_Group_Avg_{sat_erw_col}.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)

# --- PLOT 3: CAMPUS vs DISTRICT PERFORMANCE (Box Plot of Deviation) ---
if campus_col and 'district_avg' in agg_dfs:
    # Merge district average back to calculate deviation
    temp_df = df.merge(agg_dfs['district_avg'], on=[district_col, year_col], how='left')
    deviation_col = f'{sat_erw_col}_Dev_Campus_Dist'
    temp_df[deviation_col] = temp_df[sat_erw_col] - temp_df[f'{sat_erw_col}_Avg']

    plt.figure(figsize=(12, 6))
    # Plot the distribution of Campus scores relative to their District average
    sns.histplot(temp_df[deviation_col].dropna(), kde=True, bins=50, color='skyblue')
    plt.axvline(0, color='red', linestyle='--', label='District Average')
    plt.title(f'Distribution of {sat_erw_col} Deviation (Campus Score - District Average)')
    plt.xlabel('Deviation Score')
    plt.ylabel('Count of Campuses')
    plt.legend()
    file_name = f'Comparison_3_Campus_Dist_Deviation.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)

# --- PLOT 4: DISTRICT vs REGION PERFORMANCE (Scatter Plot) ---
if district_col and 'region_avg' in agg_dfs:
    # Flatten regional data to plot district average vs region average for one year
    plot_metric = sat_erw_col
    plot_year = df[year_col].max() # Use the most recent year

    # Filter District Averages for the latest year
    dist_latest = agg_dfs['district_avg'][agg_dfs['district_avg'][year_col] == plot_year][[district_col, f'{plot_metric}_Avg']]
    
    # Get Region Average for the latest year
    region_latest = agg_dfs['region_avg'][agg_dfs['region_avg'][year_col] == plot_year][[region_col, f'{plot_metric}_Avg']]
    
    # Merge District Avg with Region
    plot_df = dist_latest.merge(df[[district_col, region_col]].drop_duplicates(), on=district_col, how='left')
    plot_df = plot_df.merge(region_latest, on=region_col, how='left', suffixes=('_Dist', '_Reg'))
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=f'{plot_metric}_Avg_Dist', y=f'{plot_metric}_Avg_Reg', hue=region_col, data=plot_df, alpha=0.7)
    
    # Add diagonal line (where District Avg = Region Avg)
    max_val = plot_df[[f'{plot_metric}_Avg_Dist', f'{plot_metric}_Avg_Reg']].max().max()
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='District = Region')

    plt.title(f'District Average vs. Region Average {plot_metric} ({plot_year})')
    plt.xlabel(f'District Average {plot_metric}')
    plt.ylabel(f'Region Average {plot_metric}')
    plt.legend(title=region_col, bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    file_name = f'Comparison_4_Dist_Region_Scatter.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)

print(f"Successfully generated and saved {len(saved_plots)} visualization files.")

# (Keep plots saved as PNG files in the image output dir; optionally create an index workbook)
try:
    # Create a simple index workbook listing saved plot filenames
    plot_index = pd.DataFrame({'plot_file': saved_plots})
    plot_index_path = save_df_excel(plot_index, Path(image_output_dir), 'plot_index', sheet_name='plots')
    print(f"Saved plot index workbook: {plot_index_path}")
except Exception as e:
    print(f"Could not save plot index workbook: {e}")

# --- 7) SAVE PHASE 1 DATASET (with time-series features only) ---
# Saving the raw data + time-series features (Rolling Mean and Diff) as requested
print("\nSaving Phase 1 dataset to:", output_path)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
df.to_excel(output_path, index=False)
print("Saved. Final shape:", df.shape)

# Save a sample of the final dataset as an Excel sheet
try:
    out_dir = Path(image_output_dir)
    sample_out = df.head(30)[[c for c in df.columns[:min(40, len(df.columns))]]]
    out_xlsx = save_df_excel(sample_out, out_dir, "final_sample", sheet_name="final_sample")
    print(f"Saved final dataset sample Excel: {out_xlsx}")
except Exception as e:
    print(f"Could not save final dataset sample Excel: {e}")

# --- 8) BRIEF SAMPLE OUTPUT ---
print("\nPhase 1 complete. Data ready:", output_path)
print(f"Visualizations saved to: {image_output_dir}")