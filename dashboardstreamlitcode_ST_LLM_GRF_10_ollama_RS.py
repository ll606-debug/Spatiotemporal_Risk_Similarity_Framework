# app_phase4_chat_final.py  (fixed subgroup handling + graceful fallback)
import os
import io
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# peer similarity utilities imported from existing module
from campus_similarity_dtw_dashboard_RS import (
    df as sim_df,
    find_similar_campuses,
    ACT_COLS,
    SAT_COLS,
)

# ==========================
# RISK SCORE VISUAL UTILITIES
# ==========================
RISK_BANDS = [
    (0, 20, "Excellent readiness", "#2ECC71"),
    (20, 40, "Caution", "#F1C40F"),
    (40, 60, "Moderate risk", "#E67E22"),
    (60, 80, "High risk", "#D35400"),
    (80, 100, "Severe risk", "#C0392B"),
]

def band_color(score):
    for lo, hi, label, color in RISK_BANDS:
        if lo <= score < hi:
            return color
    return RISK_BANDS[-1][3]

def band_label(score):
    for lo, hi, label, color in RISK_BANDS:
        if lo <= score < hi:
            return label
    return RISK_BANDS[-1][2]

# ==========================
# TRY LOADING PHASE 4 AGENTS
# ==========================
try:
    from phase4code_ST_LLM import (
        WhatIfAgent, InterventionAgent, LLMClient,
        ScenarioConfig, ScenarioChange
    )
    PHASE4_OK = True
except Exception as e:
    PHASE4_OK = False
    _phase4_err = e

st.set_page_config(layout="wide", page_title="EIP – What-If & Intervention Engine")
st.title("🎓 Educational Intelligence Platform ")

if not PHASE4_OK:
    st.error(f"Cannot import phase4code_ST_LLM.py: {_phase4_err}")
    st.stop()

# ==========================
# LOAD ANALYTICS DATA
# ==========================
candidate_paths = [
    "/Users/LANKM9/Downloads/MajorProject-main/MajorProject-main/phase3_predictions_ST_LLM.xlsx",
    "/Users/LANKM9/Downloads/MajorProject-main/MajorProject-main/phase2_analytics_with_ACT_equity_ST_LLM.xlsx",
    "/Users/LANKM9/Downloads/MajorProject-main/MajorProject-main/combined_year_dataset_cleaned_ST.xlsx",
]

df_master = pd.DataFrame()
loaded_path = None

for p in candidate_paths:
    if Path(p).exists():
        try:
            st.info(f"Trying to load analytics file from: {p}")
            df_master = pd.read_excel(p)
            loaded_path = p
            break
        except:
            pass

if df_master.empty:
    st.error("No analytics file found. Update candidate_paths or place Phase2/3 file in same directory.")
    st.stop()

st.success(f"Loaded analytics file: {loaded_path}   |   Shape = {df_master.shape}")

# ==========================
# AUTO-DETECT ID & LOC COLUMNS
# ==========================
def find_first(cols, pats):
    for pat in pats:
        for c in cols:
            if pat in c.lower():
                return c
    return None

cols = list(df_master.columns)
campus_col = find_first(cols, ["aicode", "campus"]) or "Campus"
group_col  = find_first(cols, ["group"]) or "Group"
year_col   = find_first(cols, ["year"]) or "year"
district_col = find_first(cols, ["district", "distname", "dist"]) or "District"
region_col = find_first(cols, ["region", "regnname", "reg"]) or "Region"

id_cols = {"campus": campus_col, "group": group_col, "year": year_col}
st.info(f"Auto-detected ID columns: {id_cols}  —  District: '{district_col}', Region: '{region_col}'")

# basic sanity
if campus_col not in df_master.columns:
    st.error(f"Campus column '{campus_col}' not found in dataset.")
    st.stop()
if year_col not in df_master.columns:
    st.error(f"Year column '{year_col}' not found in dataset.")
    st.stop()

# ==========================
# INITIALIZE AGENTS + LLM PROVIDERS
# ==========================
api_key_env = os.getenv("GEMINI_API_KEY", "AIzaSyBL0BypOJNiWQLth0khJ4d1hJdTfvUm14Y")
# Primary Gemini client (may be disabled if no key/sdk)
llm_client = LLMClient(api_key=api_key_env)
# Instantiate agents without a bound LLM; we'll attach the selected provider at runtime
what_if_agent = WhatIfAgent(df=df_master, id_cols=id_cols, llm_client=None)
intervention_agent = InterventionAgent(None, id_cols=id_cols)

# Build available LLM providers (only add if the client appears configured)
llm_providers = {}
try:
    if llm_client and getattr(llm_client, 'client', None):
        llm_providers['Gemini'] = llm_client
except Exception:
    pass

# Import other provider classes from the phase4 module if available
try:
    from phase4code_ST_LLM import OpenAILLMClient, ClaudeLLMClient, AzureOpenAIClient, GroqLLMClient, MockLLMClient, OllamaLLMClient
    try:
        openai_client = OpenAILLMClient()
        if getattr(openai_client, 'client', None):
            llm_providers['ChatGPT'] = openai_client
    except Exception:
        pass

    try:
        claude_client = ClaudeLLMClient()
        if getattr(claude_client, 'client', None):
            llm_providers['Claude'] = claude_client
    except Exception:
        pass

    try:
        azure_client = AzureOpenAIClient()
        if getattr(azure_client, 'client', None):
            llm_providers['MS Copilot'] = azure_client
    except Exception:
        pass

    try:
        groq_api_key = os.getenv("GROQ_API_KEY","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")
        if groq_api_key:
            groq_client = GroqLLMClient(api_key=groq_api_key)
            if getattr(groq_client, 'client', None):
                llm_providers['Groq'] = groq_client
    except Exception:
        pass

    # Try local Ollama (gemma2 and gemma3)
    try:
        try:
            ollama_client = OllamaLLMClient(model="gemma2")
            # Quick availability check: ensure session exists (server may be offline)
            if getattr(ollama_client, 'session', None):
                llm_providers['Ollama (gemma2)'] = ollama_client
        except Exception:
            pass

        try:
            ollama_client_3 = OllamaLLMClient(model="gemma3")
            if getattr(ollama_client_3, 'session', None):
                llm_providers['Ollama (gemma3)'] = ollama_client_3
        except Exception:
            pass
    except Exception:
        pass

    # Always include Mock provider for demo / fallback
    try:
        mock_client = MockLLMClient()
        llm_providers['Mock'] = mock_client
    except Exception:
        pass
except Exception:
    # If import fails, at least try to add a Mock client by name (phase4 module import error handled above)
    try:
        from phase4code_ST_LLM import MockLLMClient as MockLLMClientAlt
        llm_providers['Mock'] = MockLLMClientAlt()
    except Exception:
        pass

# UI: choose which LLM provider to use for narrative & recommendations
provider_names = sorted(list(llm_providers.keys())) if llm_providers else []
if provider_names:
    selected_provider = st.selectbox("Choose LLM provider for narrative / recommendations", provider_names, index=provider_names.index('Mock') if 'Mock' in provider_names else 0, key="llm_provider_select")
    st.info(f"Selected LLM provider: {selected_provider}")
else:
    selected_provider = None
    st.warning("No LLM providers detected; narrative and LLM recommendations will use fallbacks.")

# ==========================
# TOP: Campus Overview & Subgroup breakdowns
# ==========================
st.header("Campus overview & subgroup breakdowns")

overview_cols = [campus_col]
campus_vals = sorted(df_master[campus_col].dropna().unique().tolist())
selected_overview_campus = st.selectbox("Select campus to preview", campus_vals, index=0, key="overview_campus")

# show subgroup breakdown table for that campus
campus_df = df_master[df_master[campus_col] == selected_overview_campus].copy()
st.markdown(f"**Rows for {selected_overview_campus} (all years & subgroups)** — total: {len(campus_df)}")
preview_cols = [year_col, group_col, "ERW_SAT", "Math_SAT", "Total_SAT", "Above_TSI_Both_Rate", "Part_Rate", "risk_score", "Max_Equity_Gap", "Campus_Dropout_2223", "Campus_Retention_2223", "Campus_Type"]
preview_cols = [c for c in preview_cols if c in campus_df.columns]
if not campus_df.empty:
    st.dataframe(campus_df[preview_cols].sort_values(year_col, ascending=False).reset_index(drop=True).head(200))
else:
    st.write("No rows for this campus in dataset.")

st.markdown("---")

# ==========================
# Academic Risk Index explorer (campus risk colors included)
# ==========================
st.header("Academic Risk Index explorer")

explore_campus = st.selectbox(
    "Choose campus to inspect risk", campus_vals,
    index=campus_vals.index(selected_overview_campus) if selected_overview_campus in campus_vals else 0,
    key="explore_campus"
)
campus_rows = df_master[df_master[campus_col] == explore_campus]
if campus_rows.empty:
    st.write("No data for this campus.")
else:   
    latest_year_for_campus = int(campus_rows[year_col].max())
    row_latest = campus_rows[campus_rows[year_col] == latest_year_for_campus].iloc[0]
    st.markdown(f"**Latest year available:** {latest_year_for_campus}")
    if "risk_score" in row_latest.index and pd.notna(row_latest["risk_score"]):
        rs = float(row_latest["risk_score"])
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:{band_color(rs)};color:white;text-align:center'><b>{explore_campus}</b><br/>Risk: {rs:.1f} — {band_label(rs)}</div>", unsafe_allow_html=True)
    else:
        st.info("Risk score not available for this campus (latest row).")

    comp_list = []
    for c in ["English_ACT", "Math_ACT", "Reading_ACT", "Science_ACT", "Compos_ACT", "Part_Rate_ACT", "Above_Crit_Rate_ACT", "Above_TSI_Both_Rate_ACT", "ERW_SAT", "Math_SAT", "Total_SAT", "Part_Rate_SAT", "Above_Crit_Rate_SAT", "Above_TSI_Both_Rate_SAT", "year", "Campus_Type", "Campus_Type_Description", "Campus_Dropout_2223", "Campus_Retention_2223"]:
        if c in row_latest.index:
            comp_list.append((c, row_latest.get(c)))
    if comp_list:
        st.markdown("**Key components (latest row)**")
        comp_display = {k: (f"{v:.1f}" if pd.notna(v) else "NA") for k, v in comp_list}
        st.write(comp_display)
    st.markdown("---")

# ==========================
# Benchmarks: Campus vs District vs Region
# ==========================
st.header("Benchmarks (Campus vs District vs Region)")

bench_metrics = [m for m in ["English_ACT", "Math_ACT", "Reading_ACT", "Science_ACT", "Compos_ACT", "Part_Rate_ACT", "Above_Crit_Rate_ACT", "Above_TSI_Both_Rate_ACT", "ERW_SAT", "Math_SAT", "Total_SAT", "Part_Rate_SAT", "Above_Crit_Rate_SAT", "Above_TSI_Both_Rate_SAT", "year", "Campus_Type", "Campus_Type_Description", "Campus_Dropout_2223", "Campus_Retention_2223"] if m in df_master.columns]
bench_years = sorted(df_master[year_col].dropna().astype(int).unique().tolist())
bench_year_choice = st.selectbox("Choose year for benchmarks", ["latest"] + bench_years, index=0, key="bench_year")

bench_year_true = None if bench_year_choice == "latest" else int(bench_year_choice)
if bench_year_true is None:
    bench_year_true = int(df_master[year_col].max())

bench_sub = df_master[df_master[year_col] == bench_year_true].copy()
if bench_sub.empty:
    st.write(f"No data for year {bench_year_true}")
else:
    bench_campus = st.selectbox("Choose campus for benchmark comparison", campus_vals, index=campus_vals.index(selected_overview_campus) if selected_overview_campus in campus_vals else 0, key="bench_campus")
    campus_avg = bench_sub[bench_sub[campus_col] == bench_campus][bench_metrics].mean().to_frame().T

    if district_col in bench_sub.columns:
        district_name = bench_sub[bench_sub[campus_col] == bench_campus][district_col].dropna().unique()
        district_name = district_name[0] if len(district_name) > 0 else None
        if district_name is not None:
            district_avg = bench_sub[bench_sub[district_col] == district_name][bench_metrics].mean().to_frame().T
        else:
            district_avg = pd.DataFrame(columns=bench_metrics)
    else:
        district_avg = pd.DataFrame(columns=bench_metrics)

    if region_col in bench_sub.columns:
        region_name = bench_sub[bench_sub[campus_col] == bench_campus][region_col].dropna().unique()
        region_name = region_name[0] if len(region_name) > 0 else None
        if region_name is not None:
            region_avg = bench_sub[bench_sub[region_col] == region_name][bench_metrics].mean().to_frame().T
        else:
            region_avg = pd.DataFrame(columns=bench_metrics)
    else:
        region_avg = pd.DataFrame(columns=bench_metrics)

    rows = []
    campus_row = {"level": bench_campus}
    if not campus_avg.empty:
        for m in bench_metrics:
            campus_row[m] = campus_avg.iloc[0].get(m, np.nan)
    else:
        for m in bench_metrics:
            campus_row[m] = np.nan
    rows.append(campus_row)

    if not district_avg.empty:
        drow = {"level": district_name or "District Avg"}
        for m in bench_metrics:
            drow[m] = district_avg.iloc[0].get(m, np.nan)
        rows.append(drow)

    if not region_avg.empty:
        rrow = {"level": region_name or "Region Avg"}
        for m in bench_metrics:
            rrow[m] = region_avg.iloc[0].get(m, np.nan)
        rows.append(rrow)

    bench_df = pd.DataFrame(rows)
    if not bench_df.empty:
        st.dataframe(bench_df.set_index("level").T.style.format("{:.1f}"))
    else:
        st.write("Benchmarks could not be computed for this campus/year.")

# -----------------------------
# peer campus similarity section
# -----------------------------
st.header("Peer Campus Similarity")
st.write("The similarity analysis uses the campus chosen above in the overview section.")

sim_top_k = st.slider("Number of similar campuses", 5, 20, 10, key="sim_k")
if st.button("Find Similar Campuses", key="sim_button"):
    with st.spinner("Running similarity engine..."):
        #row_latest = campus_rows[campus_rows[year_col] == latest_year_for_campus].iloc[0]
        Campus_Aicode = row_latest["Campus"] 
        sim_results = find_similar_campuses(sim_df, Campus_Aicode, top_k=sim_top_k)
        st.subheader(f"📌 Similar Campuses for: {selected_overview_campus} - {Campus_Aicode}")
        st.dataframe(sim_results, use_container_width=True)

        # trend comparison plot (same as original dashboard_app)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ACT trends
        target_act = (
            sim_df[sim_df["Campus"] == Campus_Aicode]
            .sort_values("year")[ACT_COLS]
            .mean(axis=1)
        )

        axes[0].plot(
            sim_df[sim_df["Campus"] == Campus_Aicode]["year"],
            target_act,
            marker="o",
            label=f"{Campus_Aicode} (Target)"
        )
        for peer in sim_results["Campus"][1:10]:
            peer_act = (
                #sim_df[sim_df["Campus"] == peer]
                sim_df[sim_df["Campus"] == int(peer.split("-")[1].strip())]
                .sort_values("year")[ACT_COLS]
                .mean(axis=1)
            )
            axes[0].plot(
                sim_df[sim_df["Campus"] == int(peer.split("-")[1].strip())]["year"],
                peer_act,
                linestyle="--",
                alpha=0.7,
                label=int(peer.split("-")[1].strip())
            )
        axes[0].set_title("ACT Trends")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("ACT Score")
        #axes[0].legend()
        # Move legend outside
        #axes[0].legend(bbox_to_anchor=(1, 1), loc='upper left')

        # SAT trends
        target_sat = (
            sim_df[sim_df["Campus"] == Campus_Aicode]
            .sort_values("year")[SAT_COLS]
            .mean(axis=1)
        )

        axes[1].plot(
            sim_df[sim_df["Campus"] == Campus_Aicode]["year"],
            target_sat,
            marker="o",
            label=f"{Campus_Aicode} (Target)"
        )
        for peer in sim_results["Campus"][1:10]:
            peer_sat = (
                #sim_df[sim_df["Campus"] == peer]
                #sim_df[sim_df["Campus"] == peer]
                sim_df[sim_df["Campus"] == int(peer.split("-")[1].strip())]                
                .sort_values("year")[SAT_COLS]
                .mean(axis=1)
            )
            axes[1].plot(
                sim_df[sim_df["Campus"] == int(peer.split("-")[1].strip())]["year"],
                peer_sat,
                linestyle="--",
                alpha=0.7,
                label=peer
            )
        axes[1].set_title("SAT Trends")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("SAT Score")
        #axes[1].legend()
        # Move legend outside
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')        
        st.pyplot(fig)
        # -----------------------------
        # Streamlit comparison graphs for selected similar campus
        # -----------------------------
        peers = sim_results["Campus"].tolist() if "Campus" in sim_results.columns else []
        # assume the first entry is the target itself; offer the others for comparison
        peer_options = peers[1:] if len(peers) > 1 else []

        if peer_options:
            chosen_peer = st.selectbox("Choose a similar campus to compare with target", peer_options, index=0, key="peer_compare")

            # build timeseries for target and chosen peer
            target_df = sim_df[sim_df["Campus"] == Campus_Aicode].sort_values("year")
            peer_df = sim_df[sim_df["Campus"] == chosen_peer].sort_values("year")

            # choose numeric metrics to compare (ACT/SAT sets plus other numeric fields)
            metric_candidates = []
            try:
                metric_candidates = list(ACT_COLS) + list(SAT_COLS) + ["Total_SAT", "Part_Rate", "Above_TSI_Both_Rate", "risk_score"]
            except Exception:
                metric_candidates = []

            # also include any other numeric columns present in sim_df (excluding year)
            numeric_cols = [c for c in sim_df.columns if c != "year" and pd.api.types.is_numeric_dtype(sim_df[c])]
            for c in numeric_cols:
                if c not in metric_candidates:
                    metric_candidates.append(c)

            # filter to columns that actually exist
            metrics = [m for m in metric_candidates if m in sim_df.columns]

            st.markdown("**Comparison: target vs selected peer (per-metric timeseries)**")
            if not metrics:
                st.info("No comparable numeric metrics found for plotting.")
            else:
                for metric in metrics:
                    s_target = target_df.set_index("year")[metric] if metric in target_df.columns else pd.Series(dtype=float)
                    s_peer = peer_df.set_index("year")[metric] if metric in peer_df.columns else pd.Series(dtype=float)

                    # union of years
                    years = sorted(set(s_target.index.tolist()) | set(s_peer.index.tolist()))
                    if not years:
                        continue

                    comp_df = pd.DataFrame(index=years)
                    comp_df[Campus_Aicode] = s_target.reindex(years).values
                    comp_df[chosen_peer] = s_peer.reindex(years).values

                    comp_df.index.name = "year"
                    st.markdown(f"**{metric}**")
                    st.line_chart(comp_df)
                    st.write("---")
        else:
            st.info("No similar peers returned to compare.")

st.markdown("---")

# ==========================
# ONE-SHOT FORM: WHAT-IF + INTERVENTIONS
# ==========================
st.header("🧪 What-If Scenario Builder (One Step)")

with st.form("scenario_form"):
    campuses = campus_vals
    try:
        default_campus_index = campuses.index(selected_overview_campus)
    except Exception:
        default_campus_index = 0

    chosen_campus = st.selectbox("Select a campus", campuses, index=default_campus_index, key="form_campus")

    campus_subset = df_master[df_master[campus_col] == chosen_campus]
    years = sorted(campus_subset[year_col].dropna().astype(int).unique().tolist())
    years_options = ["latest"] + years
    try:
        default_year_index = years_options.index("latest")
    except Exception:
        default_year_index = 0
    chosen_year = st.selectbox("Select year", years_options, index=default_year_index, key="form_year")
    chosen_year_true = None if chosen_year == "latest" else int(chosen_year)

    # --- FIX: subgroup options are filtered to the selected campus ONLY ---
    campus_groups = []
    if group_col in campus_subset.columns:
        campus_groups = sorted(campus_subset[group_col].dropna().unique().tolist())
    groups_options = ["All"] + campus_groups
    chosen_group = st.selectbox("Subgroup (optional) — only groups for selected campus", groups_options, index=0, key="form_group")
    chosen_group_val = None if chosen_group == "All" else chosen_group

    st.markdown("### Enter Targets or Deltas (leave blank to skip)")
    metrics = [
        "English_ACT", "Math_ACT", "Reading_ACT", "Science_ACT", "Compos_ACT", "Part_Rate_ACT", "Above_Crit_Rate_ACT", "Above_TSI_Both_Rate_ACT", "ERW_SAT", "Math_SAT", "Total_SAT", "Part_Rate_SAT", "Above_Crit_Rate_SAT", "Above_TSI_Both_Rate_SAT", "year", "Campus_Type", "Campus_Type_Description", "Campus_Dropout_2223", "Campus_Retention_2223"
    ]

    targets = {}
    deltas = {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Targets (absolute values)**")
        for m in metrics:
            v = st.text_input(f"Target {m}", key=f"t_{m}")
            if v.strip():
                try: targets[m] = float(v)
                except: pass

    with col2:
        st.markdown("**Deltas (add/subtract)**")
        for m in metrics:
            v = st.text_input(f"Delta {m}", key=f"d_{m}")
            if v.strip():
                try: deltas[m] = float(v)
                except: pass

    confirm = st.form_submit_button("⚡ Run Scenario")

# ==========================
# RUN SCENARIO
# ==========================
if confirm:
    try:
        # If user selected a subgroup that is not present for the chosen campus
        if chosen_group_val is not None and chosen_group_val not in campus_groups:
            st.warning(f"The subgroup '{chosen_group_val}' is not present for campus {chosen_campus}. Falling back to campus-level scenario.")
            chosen_group_val = None  # fallback to campus-level

        cfg_level = "campus_group" if chosen_group_val else "campus"
        cfg_id = (chosen_campus, chosen_group_val) if chosen_group_val else chosen_campus

        cfg = ScenarioConfig(level=cfg_level, id_value=cfg_id, year=chosen_year_true)
        change = ScenarioChange(targets=targets, deltas=deltas)

        try:
            # Attach the selected LLM provider to agents for this run (if any)
            if selected_provider and selected_provider in llm_providers:
                sel_client = llm_providers[selected_provider]
                what_if_agent.llm = sel_client
                intervention_agent.llm = sel_client
                #what_if_agent = WhatIfAgent(df=df_master, id_cols=id_cols, llm_client=sel_client)
                #intervention_agent = InterventionAgent(llm_client=sel_client, id_cols=id_cols)
                

            result, narrative = what_if_agent.run_scenario(cfg, change, scenario_description="EIP UI scenario")
        except ValueError as ve:
            # If _locate_row failed for campus_group, attempt campus-only fallback
            msg = str(ve)
            if "No matching rows for ScenarioConfig" in msg:
                st.warning("Could not find subgroup row for that campus+group. Retrying at campus level instead.")
                cfg = ScenarioConfig(level="campus", id_value=chosen_campus, year=chosen_year_true)
                result, narrative = what_if_agent.run_scenario(cfg, change, scenario_description="EIP UI scenario (campus fallback)")
            else:
                raise ve

        baseline = result.baseline_row
        scenario = result.scenario_row

        # ==========================
        # RISK TILE
        # ==========================
        st.subheader("📊 Updated Risk Score")
        if "risk_score" in scenario.index and pd.notna(scenario["risk_score"]):
            rs = float(scenario["risk_score"])
            tile = f"""
            <div style='padding:15px;border-radius:8px;background:{band_color(rs)};color:white;
                        text-align:center;font-size:18px;'>
                <b>{chosen_campus}</b><br>
                Risk Score: {rs:.1f} — {band_label(rs)}
            </div>
            """
            st.markdown(tile, unsafe_allow_html=True)
        else:
            st.info("Risk score not available in scenario output.")

        # ==========================
        # BASELINE → SCENARIO TABLE
        # ==========================
        st.subheader("Baseline vs Scenario (key metrics)")
        show_cols = result.changed_columns + ["risk_score","Total_SAT","Above_TSI_Both_Rate","Part_Rate"]
        show_cols = [c for c in show_cols if c in baseline.index]
        comp_df = pd.DataFrame({
            "metric": show_cols,
            "baseline": [baseline.get(c) for c in show_cols],
            "scenario": [scenario.get(c) for c in show_cols]
        })
        st.dataframe(comp_df)

        # ==========================
        # NARRATIVE
        # ==========================
        st.subheader("📝 Narrative (LLM)")
        if narrative:
            st.info(narrative)
        else:
            st.info("No LLM narrative available (check GEMINI_API_KEY or LLM client).")

        # ==========================
        # INTERVENTIONS (NORMAL TEXT)
        # ==========================
        st.subheader("🩺 Intervention Recommendations")
        rec = intervention_agent.recommend_for_row(scenario, result)

        try:
            parsed = json.loads(rec) if isinstance(rec, str) else rec
            if isinstance(parsed, dict) and "recommendations" in parsed:
                for i, r in enumerate(parsed["recommendations"], start=1):
                    st.markdown(f"### {i}. {r.get('problem')}")
                    st.write(f"**Intervention:** {r.get('intervention')}")
                    st.write(f"**Expected Effect:** {r.get('effect')}")
                    st.markdown("---")
                if parsed.get("summary_for_leaders"):
                    st.success(parsed["summary_for_leaders"])
            else:
                st.write(parsed)
        except Exception:
            st.write(rec)

    except Exception as e:
        st.error(f"Could not run scenario: {e}")
