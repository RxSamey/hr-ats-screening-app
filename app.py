import os
import io
import re
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st


DATA_FILE_PATH = os.getenv("EMPLOYEE_FINANCE_EXCEL_PATH", "employee_finance_data.xlsx")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_prepare(file_or_path) -> pd.DataFrame:
    if isinstance(file_or_path, str):
        df = pd.read_excel(file_or_path)
    else:
        df = pd.read_excel(file_or_path)
    expected_cols = [
        "Month",
        "ServiceLine",
        "Account",
        "EmployeeId",
        "Revenue",
        "DirectCost",
        "BillableHours",
        "NonBillableHours",
        "AvailableHours",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    df["Month"] = pd.to_datetime(df["Month"])
    df["Revenue"] = df["Revenue"].fillna(0.0).astype(float)
    df["DirectCost"] = df["DirectCost"].fillna(0.0).astype(float)
    df["BillableHours"] = df["BillableHours"].fillna(0.0).astype(float)
    df["NonBillableHours"] = df["NonBillableHours"].fillna(0.0).astype(float)
    df["AvailableHours"] = df["AvailableHours"].fillna(0.0).astype(float)
    df["GrossMargin"] = df["Revenue"] - df["DirectCost"]
    df["MarginPct"] = np.where(
        df["Revenue"] != 0, df["GrossMargin"] / df["Revenue"] * 100.0, np.nan
    )
    df["UtilizationPct"] = np.where(
        df["AvailableHours"] != 0,
        df["BillableHours"] / df["AvailableHours"] * 100.0,
        np.nan,
    )
    return df


def aggregate(df: pd.DataFrame, group_dims: List[str]) -> pd.DataFrame:
    g = (
        df.groupby(["Month"] + group_dims)[
            [
                "Revenue",
                "DirectCost",
                "GrossMargin",
                "BillableHours",
                "NonBillableHours",
                "AvailableHours",
            ]
        ]
        .sum()
        .reset_index()
    )
    g["UtilizationPct"] = np.where(
        g["AvailableHours"] != 0,
        g["BillableHours"] / g["AvailableHours"] * 100.0,
        np.nan,
    )
    g["GrossMargin"] = g["Revenue"] - g["DirectCost"]
    g["MarginPct"] = np.where(
        g["Revenue"] != 0, g["GrossMargin"] / g["Revenue"] * 100.0, np.nan
    )
    metrics_for_mom = ["Revenue", "DirectCost", "GrossMargin", "UtilizationPct"]
    g = g.sort_values(["Month"] + group_dims)
    for metric in metrics_for_mom:
        prev_col = metric + "_Prev"
        mom_abs_col = metric + "_MoMAbs"
        mom_pct_col = metric + "_MoMPct"
        g[prev_col] = g.groupby(group_dims)[metric].shift(1)
        g[mom_abs_col] = g[metric] - g[prev_col]
        g[mom_pct_col] = np.where(
            g[prev_col] != 0,
            g[mom_abs_col] / g[prev_col] * 100.0,
            np.nan,
        )
    rev_group = g.groupby(group_dims)["Revenue"]
    mean_rev = rev_group.transform("mean")
    std_rev = rev_group.transform("std").replace(0, np.nan)
    g["RevenueZScore"] = np.where(
        std_rev.notna(), (g["Revenue"] - mean_rev) / std_rev, 0.0
    )
    g["RevenueAnomaly"] = g["RevenueZScore"].abs() > 2.5
    return g


def format_month_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Month" in df.columns:
        df = df.copy()
        df["Month"] = df["Month"].dt.strftime("%Y-%m")
    return df


st.set_page_config(page_title="Revenue & Cost Analyzer", page_icon="📊", layout="wide")

st.title("Revenue, Cost and Utilization Analyzer")

uploaded_file = st.file_uploader("Upload monthly employee-level Excel file", type=["xlsx", "xls"])

data_load_state = st.empty()

df = None

if uploaded_file is not None:
    data_load_state.text("Loading uploaded file...")
    df = load_and_prepare(uploaded_file)
    data_load_state.text("File loaded successfully.")
else:
    if os.path.exists(DATA_FILE_PATH):
        data_load_state.text(f"Loading default file: {DATA_FILE_PATH}")
        df = load_and_prepare(DATA_FILE_PATH)
        data_load_state.text(f"Loaded default file: {DATA_FILE_PATH}")
    else:
        data_load_state.warning("Upload an Excel file or set EMPLOYEE_FINANCE_EXCEL_PATH.")

if df is not None and not df.empty:
    months = sorted(df["Month"].dt.to_period("M").astype(str).unique().tolist())
    service_lines_all = sorted(df["ServiceLine"].dropna().unique().tolist())

    with st.sidebar:
        st.subheader("Filters")
        selected_month = st.selectbox("Month (YYYY-MM)", months, index=len(months) - 1)
        selected_service_line = st.multiselect("Service Line", ["All"] + service_lines_all, default=["All"])
        util_min, util_max = st.slider(
            "Utilization range (%)",
            min_value=0,
            max_value=200,
            value=(0, 120),
            step=5,
        )
        variance_threshold = st.number_input(
            "Absolute Revenue MoM variance threshold",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )

    df["MonthKey"] = df["Month"].dt.to_period("M").astype(str)
    df_filtered = df[df["MonthKey"] == selected_month].copy()

    if "All" not in selected_service_line:
        df_filtered = df_filtered[df_filtered["ServiceLine"].isin(selected_service_line)]

    agg_sl = aggregate(df, ["ServiceLine"])
    agg_sl_acct = aggregate(df, ["ServiceLine", "Account"])

    agg_sl["MonthKey"] = agg_sl["Month"].dt.to_period("M").astype(str)
    agg_sl_acct["MonthKey"] = agg_sl_acct["Month"].dt.to_period("M").astype(str)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Service Line Summary",
            "Account Summary",
            "Variance Explainer",
            "Employee Utilization",
        ]
    )

    with tab1:
        st.subheader("Service Line Summary")
        sub = agg_sl[agg_sl["MonthKey"] == selected_month].copy()
        if "All" not in selected_service_line:
            sub = sub[sub["ServiceLine"].isin(selected_service_line)]
        if sub.empty:
            st.info("No data for selected filters.")
        else:
            total_var = sub["Revenue_MoMAbs"].abs().sum()
            if total_var == 0:
                sub["RevenueVarContributionPct"] = 0.0
            else:
                sub["RevenueVarContributionPct"] = (
                    sub["Revenue_MoMAbs"].abs() / total_var * 100.0
                )
            if variance_threshold > 0:
                sub = sub[sub["Revenue_MoMAbs"].abs() >= variance_threshold]
            sub = format_month_column(sub)
            display_cols = [
                "Month",
                "ServiceLine",
                "Revenue",
                "Revenue_Prev",
                "Revenue_MoMAbs",
                "Revenue_MoMPct",
                "DirectCost",
                "GrossMargin",
                "MarginPct",
                "UtilizationPct",
                "RevenueVarContributionPct",
                "RevenueAnomaly",
            ]
            sub = sub[display_cols]
            st.table(sub)

    with tab2:
        st.subheader("Account Summary")
        sub = agg_sl_acct[agg_sl_acct["MonthKey"] == selected_month].copy()
        if "All" not in selected_service_line:
            sub = sub[sub["ServiceLine"].isin(selected_service_line)]
        if sub.empty:
            st.info("No data for selected filters.")
        else:
            total_var = sub["Revenue_MoMAbs"].abs().sum()
            if total_var == 0:
                sub["RevenueVarContributionPct"] = 0.0
            else:
                sub["RevenueVarContributionPct"] = (
                    sub["Revenue_MoMAbs"].abs() / total_var * 100.0
                )
            if variance_threshold > 0:
                sub = sub[sub["Revenue_MoMAbs"].abs() >= variance_threshold]
            sub = format_month_column(sub)
            display_cols = [
                "Month",
                "ServiceLine",
                "Account",
                "Revenue",
                "Revenue_Prev",
                "Revenue_MoMAbs",
                "Revenue_MoMPct",
                "DirectCost",
                "GrossMargin",
                "MarginPct",
                "UtilizationPct",
                "RevenueVarContributionPct",
                "RevenueAnomaly",
            ]
            sub = sub[display_cols]
            st.table(sub)

    with tab3:
        st.subheader("Variance Explainer")
        sub = agg_sl[agg_sl["MonthKey"] == selected_month].copy()
        if "All" not in selected_service_line:
            sub = sub[sub["ServiceLine"].isin(selected_service_line)]
        if sub.empty:
            st.info("No data for selected filters.")
        else:
            sub["RevenueVarAbs"] = sub["Revenue_MoMAbs"].abs()
            total_var = sub["Revenue_MoMAbs"].sum()
            total_var_abs = sub["RevenueVarAbs"].sum()
            top_n = st.slider("Top N drivers", min_value=3, max_value=20, value=5, step=1)
            top = sub.sort_values("RevenueVarAbs", ascending=False).head(top_n)
            if total_var > 0:
                direction = "increase"
            elif total_var < 0:
                direction = "decrease"
            else:
                direction = "no net change"
            summary_parts = []
            summary_parts.append(
                f"For {selected_month}, total revenue month-over-month variance is {total_var:,.0f}, indicating an overall {direction}."
            )
            if total_var_abs > 0:
                summary_parts.append(
                    f"The top {top_n} service lines explain most of the absolute movement."
                )
            anomalies = sub[sub["RevenueAnomaly"]]
            if not anomalies.empty:
                names = ", ".join(sorted(anomalies["ServiceLine"].unique().tolist()))
                summary_parts.append(
                    f"Statistical anomalies detected for service lines: {names} (z-score > 2.5 vs historical mean)."
                )
            summary_text = " ".join(summary_parts)
            st.write(summary_text)
            top_display = format_month_column(top)[
                [
                    "Month",
                    "ServiceLine",
                    "Revenue",
                    "Revenue_Prev",
                    "Revenue_MoMAbs",
                    "Revenue_MoMPct",
                    "RevenueZScore",
                    "RevenueAnomaly",
                ]
            ]
            st.table(top_display)

    with tab4:
        st.subheader("Employee Utilization")
        df_emp = df[df["MonthKey"] == selected_month].copy()
        if "All" not in selected_service_line:
            df_emp = df_emp[df_emp["ServiceLine"].isin(selected_service_line)]
        if df_emp.empty:
            st.info("No data for selected filters.")
        else:
            df_emp = df_emp[
                (df_emp["UtilizationPct"].fillna(0) >= util_min)
                & (df_emp["UtilizationPct"].fillna(0) <= util_max)
            ]
            df_emp = df_emp.copy()
            df_emp["Month"] = df_emp["Month"].dt.strftime("%Y-%m")
            display_cols = [
                "Month",
                "ServiceLine",
                "Account",
                "EmployeeId",
                "Revenue",
                "DirectCost",
                "GrossMargin",
                "MarginPct",
                "BillableHours",
                "NonBillableHours",
                "AvailableHours",
                "UtilizationPct",
            ]
            st.table(df_emp[display_cols])
else:
    st.stop()
