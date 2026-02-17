import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ─── PAGE CONFIG (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="HR Integrated Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SHARED DATA LOADING ───────────────────────────────────────
DATA_DIR = Path("data")


@st.cache_data
def load_all_data():
    employees = pd.read_csv(DATA_DIR / "employees.csv")
    attrition = pd.read_csv(DATA_DIR / "attrition.csv")
    hires = pd.read_csv(DATA_DIR / "hires.csv")
    transfers = pd.read_csv(DATA_DIR / "transfers.csv")
    quality_log = pd.read_csv(DATA_DIR / "quality_log.csv")

    employees["Hire_Date"] = pd.to_datetime(employees["Hire_Date"])
    attrition["Exit_Date"] = pd.to_datetime(attrition["Exit_Date"], errors="coerce")
    hires["Hire_Date"] = pd.to_datetime(hires["Hire_Date"])
    transfers["Transfer_Date"] = pd.to_datetime(transfers["Transfer_Date"])

    employees_with_dups = employees.copy()
    employees = employees.drop_duplicates(subset=["Employee_ID"])
    dup_count = len(employees_with_dups) - len(employees)

    return employees, attrition, hires, transfers, quality_log, dup_count


employees, attrition, hires, transfers, quality_log, DUP_COUNT = load_all_data()


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — HR ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_hr_dashboard():
    # ─── CSS ────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        .kpi-card {
            background: #ffffff; border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08);
            overflow: hidden; text-align: center; padding: 0;
        }
        .kpi-top { height: 6px; }
        .kpi-body { padding: 16px 14px 18px; }
        .kpi-label { font-size: 0.75rem; color: #777; text-transform: uppercase; letter-spacing: .4px; margin-bottom: 6px; }
        .kpi-value { font-size: 1.8rem; font-weight: 700; margin-bottom: 3px; }
        .kpi-sub   { font-size: 0.72rem; color: #aaa; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
        .badge-good    { background: #d4edda; color: #155724; }
        .badge-monitor { background: #fff3cd; color: #856404; }
        .badge-action  { background: #f8d7da; color: #721c24; }
        .summary-item { font-size: 0.92rem; line-height: 2; color: #444; }
        .summary-item span { font-weight: 600; color: #1f77b4; }
        .section-hdr { font-size: 0.92rem; font-weight: 600; color: #333; border-bottom: 1px solid #eee; padding-bottom: 6px; margin-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

    # ─── Dimensions ─────────────────────────────────────────────
    GRADE_ORDER = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]
    AGENCIES = sorted(employees["Agency"].unique())
    JOB_GRADES = [g for g in GRADE_ORDER if g in employees["Job_Grade"].values]

    all_dates = pd.concat([
        employees["Hire_Date"], attrition["Exit_Date"].dropna(),
        hires["Hire_Date"], transfers["Transfer_Date"],
    ])
    YEARS = sorted(all_dates.dt.year.dropna().unique().astype(int))

    SET2 = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
            "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]
    C = dict(primary="#1f77b4", success="#2ca02c", warning="#ff7f0e",
             danger="#d62728", info="#17a2b8")

    # ─── Header ─────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#1f77b4;font-size:1.7rem;margin-bottom:2px'>"
        "HR Analytics Dashboard</h1>"
        "<p style='text-align:center;color:#777;font-size:0.85rem;margin-bottom:18px'>"
        "Trends &amp; Monitoring – use the sidebar filters to drill down</p>",
        unsafe_allow_html=True,
    )

    # ─── Sidebar Filters ───────────────────────────────────────
    ALL_LABEL = "All"
    with st.sidebar:
        st.markdown("---")
        st.header("Filters")

        agency_options = [ALL_LABEL] + AGENCIES
        sel_agency_raw = st.multiselect(
            "Agency", options=agency_options, default=[ALL_LABEL],
            help="Select 'All' or pick specific agencies", key="hr_agency",
        )
        if ALL_LABEL in sel_agency_raw or not sel_agency_raw:
            sel_agencies = AGENCIES
        else:
            sel_agencies = [a for a in sel_agency_raw if a != ALL_LABEL]

        grade_options = [ALL_LABEL] + JOB_GRADES
        sel_grade_raw = st.multiselect(
            "Job Grade", options=grade_options, default=[ALL_LABEL],
            help="Select 'All' or pick specific grades", key="hr_grade",
        )
        if ALL_LABEL in sel_grade_raw or not sel_grade_raw:
            sel_grades = JOB_GRADES
        else:
            sel_grades = [g for g in sel_grade_raw if g != ALL_LABEL]

        sel_year = st.selectbox("Year", options=YEARS, index=len(YEARS) - 1, key="hr_year")

    # ─── Filter Engine ─────────────────────────────────────────
    def filter_data(agencies, grades, year):
        yr_start = pd.Timestamp(year, 1, 1)
        yr_end = pd.Timestamp(year, 12, 31)
        emp = employees[employees["Agency"].isin(agencies) & employees["Job_Grade"].isin(grades)].copy()
        emp_ids = set(emp["Employee_ID"])
        att = attrition[
            attrition["Agency"].isin(agencies) & attrition["Job_Grade"].isin(grades) &
            attrition["Exit_Date"].notna() &
            (attrition["Exit_Date"] >= yr_start) & (attrition["Exit_Date"] <= yr_end)
        ].copy()
        hir = hires[
            hires["Agency"].isin(agencies) & hires["Job_Grade"].isin(grades) &
            (hires["Hire_Date"] >= yr_start) & (hires["Hire_Date"] <= yr_end)
        ].copy()
        tra = transfers[
            (transfers["From_Agency"].isin(agencies) | transfers["To_Agency"].isin(agencies)) &
            transfers["Job_Grade"].isin(grades) &
            (transfers["Transfer_Date"] >= yr_start) & (transfers["Transfer_Date"] <= yr_end)
        ].copy()
        return emp, att, hir, tra, emp_ids

    fd_emp, fd_att, fd_hir, fd_tra, fd_emp_ids = filter_data(sel_agencies, sel_grades, sel_year)

    # ─── KPI Calculations ──────────────────────────────────────
    def calc_kpis(emp, att, emp_ids, year):
        as_of = pd.Timestamp(year, 12, 31)
        active = emp[emp["Hire_Date"] <= as_of]
        hist_att = attrition[
            attrition["Employee_ID"].isin(emp_ids) & attrition["Exit_Date"].notna() &
            (attrition["Exit_Date"] <= as_of)
        ]
        hc = max(len(active) - len(hist_att), 0)
        div = hc if hc > 0 else 1
        mtd_start = pd.Timestamp(year, 12, 1)
        ytd_start = pd.Timestamp(year, 1, 1)
        roll_start = pd.Timestamp(year - 1, 12, 31)
        roll_att = attrition[
            attrition["Employee_ID"].isin(emp_ids) & attrition["Exit_Date"].notna() &
            (attrition["Exit_Date"] >= roll_start) & (attrition["Exit_Date"] <= as_of)
        ]
        mtd = len(att[att["Exit_Date"] >= mtd_start]) / div * 100
        ytd = len(att[att["Exit_Date"] >= ytd_start]) / div * 100
        roll = len(roll_att) / 12 / div * 100
        return hc, mtd, ytd, roll

    hc, mtd, ytd, roll = calc_kpis(fd_emp, fd_att, fd_emp_ids, sel_year)

    # ─── KPI Cards ─────────────────────────────────────────────
    kpi_cols = st.columns(4)
    kpi_data = [
        ("Current Headcount", f"{hc:,}", "Active employees", "#17a2b8"),
        ("MTD Attrition", f"{mtd:.2f}%", "Month-to-date", "#ff7f0e"),
        ("YTD Attrition", f"{ytd:.2f}%", "Year-to-date", "#ff7f0e"),
        ("12-Mo Rolling Avg", f"{roll:.2f}%", "Rolling average", "#1f77b4"),
    ]
    for col, (label, value, sub, color) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top" style="background:{color}"></div>
            <div class="kpi-body">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ─── Attrition Trend ───────────────────────────────────────
    def calc_monthly(emp, att, emp_ids, year):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        exits_by_month = np.zeros(12)
        for _, r in att.iterrows():
            exits_by_month[int(r["Exit_Date"].month - 1)] += 1

        # Calculate headcount at the start of each month
        # HC = employees hired before month start - exits before month start
        rates = []
        for m in range(1, 13):
            month_start = pd.Timestamp(year, m, 1)
            hired_before = len(emp[emp["Hire_Date"] < month_start])
            exited_before = len(attrition[
                attrition["Employee_ID"].isin(emp_ids) & attrition["Exit_Date"].notna() &
                (attrition["Exit_Date"] < month_start)
            ])
            hc_at_start = max(hired_before - exited_before, 1)
            rates.append(exits_by_month[m - 1] / hc_at_start * 100)

        ma = [np.mean(rates[:i + 1]) for i in range(12)]
        return months, rates, ma

    months, rates, ma = calc_monthly(fd_emp, fd_att, fd_emp_ids, sel_year)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=months, y=rates, name="Monthly Attrition",
        mode="lines+markers", line=dict(color=C["primary"], width=2.5),
        marker=dict(size=7, color=C["primary"]),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
    ))
    fig_trend.add_trace(go.Scatter(
        x=months, y=ma, name="Cumulative Avg",
        mode="lines", line=dict(color=C["danger"], width=2.5, dash="dash"),
    ))
    fig_trend.update_layout(
        title=f"Attrition Trend – {sel_year}",
        xaxis_title="Month", yaxis_title="Attrition Rate (%)",
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=320, margin=dict(l=50, r=20, t=50, b=40), plot_bgcolor="white",
    )
    fig_trend.update_xaxes(showgrid=False)
    fig_trend.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ─── Headcount by Agency & Grade | Exit Reasons ────────────
    def calc_pivot(emp, emp_ids, grades, year):
        as_of = pd.Timestamp(year, 12, 31)
        active = emp[emp["Hire_Date"] <= as_of].copy()
        exited_ids = set(attrition[
            attrition["Employee_ID"].isin(emp_ids) & attrition["Exit_Date"].notna() &
            (attrition["Exit_Date"] <= as_of)
        ]["Employee_ID"])
        curr = active[~active["Employee_ID"].isin(exited_ids)]
        pivot = curr.groupby(["Agency", "Job_Grade"]).size().unstack(fill_value=0)
        for g in grades:
            if g not in pivot.columns:
                pivot[g] = 0
        pivot = pivot[[g for g in grades if g in pivot.columns]]
        return pivot

    pivot = calc_pivot(fd_emp, fd_emp_ids, sel_grades, sel_year)

    def calc_exit_reasons(att):
        if att.empty:
            return pd.DataFrame(columns=["Exit_Reason", "Voluntary", "Involuntary"])
        grouped = att.groupby(["Exit_Reason", "Exit_Type"]).size().unstack(fill_value=0)
        for col in ["Voluntary", "Involuntary"]:
            if col not in grouped.columns:
                grouped[col] = 0
        grouped["Total"] = grouped["Voluntary"] + grouped["Involuntary"]
        grouped = grouped.sort_values("Total", ascending=True).tail(7)
        return grouped

    exit_reasons = calc_exit_reasons(fd_att)

    col_left, col_right = st.columns(2)
    with col_left:
        if not pivot.empty:
            fig_hc = go.Figure()
            for i, grade in enumerate(pivot.columns):
                fig_hc.add_trace(go.Bar(x=pivot.index, y=pivot[grade], name=grade, marker_color=SET2[i % len(SET2)]))
            fig_hc.update_layout(
                title="Headcount by Agency & Grade", barmode="stack",
                xaxis=dict(tickangle=-40), yaxis=dict(title="Headcount", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=380, margin=dict(l=50, r=20, t=50, b=80), plot_bgcolor="white",
            )
            fig_hc.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_hc, use_container_width=True)
        else:
            st.info("No headcount data for the selected filters.")

    with col_right:
        if not exit_reasons.empty:
            fig_exit = go.Figure()
            fig_exit.add_trace(go.Bar(y=exit_reasons.index, x=exit_reasons["Voluntary"], name="Voluntary", orientation="h", marker_color=C["success"]))
            fig_exit.add_trace(go.Bar(y=exit_reasons.index, x=exit_reasons["Involuntary"], name="Involuntary", orientation="h", marker_color=C["danger"]))
            fig_exit.update_layout(
                title=f"Exit Reasons – {sel_year}", barmode="stack",
                xaxis=dict(title="Count", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=380, margin=dict(l=140, r=20, t=50, b=40), plot_bgcolor="white",
            )
            fig_exit.update_xaxes(gridcolor="#eee")
            fig_exit.update_yaxes(showgrid=False)
            st.plotly_chart(fig_exit, use_container_width=True)
        else:
            st.info("No exit data for the selected filters.")

    # ─── Waterfall | Agency Totals ─────────────────────────────
    def calc_waterfall(emp, att, hir, emp_ids, year):
        yr_start = pd.Timestamp(year, 1, 1)
        opening = len(emp[emp["Hire_Date"] < yr_start])
        prev_exits = len(attrition[
            attrition["Employee_ID"].isin(emp_ids) & attrition["Exit_Date"].notna() &
            (attrition["Exit_Date"] < yr_start)
        ])
        opening = max(opening - prev_exits, 0)
        n_hires = len(hir)
        n_exits = len(att)
        closing = max(opening + n_hires - n_exits, 0)
        return opening, n_hires, n_exits, closing

    opening, n_hires, n_exits, closing = calc_waterfall(fd_emp, fd_att, fd_hir, fd_emp_ids, sel_year)

    col_wf, col_ag = st.columns(2)
    with col_wf:
        labels = ["Opening", "Hires", "Exits", "Closing"]
        bases = [0, opening, opening + n_hires - n_exits, 0]
        vals = [opening, n_hires, n_exits, closing]
        colors = [C["primary"], C["success"], C["danger"], C["primary"]]
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(x=labels, y=bases, marker_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip"))
        fig_wf.add_trace(go.Bar(
            x=labels, y=vals, marker_color=colors,
            text=[f"{v:,}" if i != 2 else f"-{v:,}" for i, v in enumerate(vals)],
            textposition="outside", showlegend=False,
        ))
        fig_wf.update_layout(
            title=f"Headcount Movement – {sel_year}", barmode="stack",
            yaxis=dict(title="Headcount", rangemode="tozero"),
            height=380, margin=dict(l=50, r=20, t=50, b=40), plot_bgcolor="white",
        )
        fig_wf.update_yaxes(gridcolor="#eee")
        fig_wf.update_xaxes(showgrid=False)
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_ag:
        if not pivot.empty:
            totals = pivot.sum(axis=1).sort_values(ascending=True)
            fig_ag = go.Figure(go.Bar(
                y=totals.index, x=totals.values, orientation="h", marker_color=C["info"],
                text=[f"{int(v):,}" for v in totals.values], textposition="outside",
            ))
            fig_ag.update_layout(
                title="Headcount by Agency", xaxis=dict(title="Total Headcount", rangemode="tozero"),
                height=380, margin=dict(l=120, r=40, t=50, b=40), plot_bgcolor="white",
            )
            fig_ag.update_xaxes(gridcolor="#eee")
            fig_ag.update_yaxes(showgrid=False)
            st.plotly_chart(fig_ag, use_container_width=True)
        else:
            st.info("No headcount data for the selected filters.")

    # ─── Summary & Data Quality ────────────────────────────────
    col_sum, col_qual = st.columns(2)
    with col_sum:
        st.markdown(f'<div class="section-hdr">Summary – {sel_year}</div>', unsafe_allow_html=True)
        vol_n = len(fd_att[fd_att["Exit_Type"] == "Voluntary"])
        inv_n = len(fd_att[fd_att["Exit_Type"] == "Involuntary"])
        tot_att = len(fd_att) or 1
        items = [
            ("Employees (filtered)", f"{len(fd_emp):,}"),
            ("Exits this year", f"{len(fd_att):,}"),
            ("  – Voluntary", f"{vol_n:,} ({vol_n / tot_att * 100:.0f}%)"),
            ("  – Involuntary", f"{inv_n:,} ({inv_n / tot_att * 100:.0f}%)"),
            ("Hires this year", f"{len(fd_hir):,}"),
            ("Transfers this year", f"{len(fd_tra):,}"),
        ]
        for label, val in items:
            st.markdown(f'<div class="summary-item">{label}: <span>{val}</span></div>', unsafe_allow_html=True)

    with col_qual:
        st.markdown('<div class="section-hdr">Data Quality Scorecard</div>', unsafe_allow_html=True)
        miss_n = attrition["Exit_Date"].isna().sum()
        tot_att_all = len(attrition)
        late_n = int(quality_log["Files_Late"].sum())
        tot_files = int(quality_log["Files_Received"].sum())
        res_n = int(quality_log["Duplicate_IDs_Resolved"].sum())

        def pct(n, d):
            return f"{n / d * 100:.1f}%" if d else "N/A"

        def badge(status):
            cls = {"Good": "badge-good", "Monitor": "badge-monitor", "Action Needed": "badge-action"}[status]
            return f'<span class="badge {cls}">{status}</span>'

        rows = [
            ("Missing Exit Dates", f"{miss_n} ({pct(miss_n, tot_att_all)})",
             "Action Needed" if tot_att_all and miss_n / tot_att_all > 0.05 else "Good"),
            ("Late File Uploads", f"{late_n} ({pct(late_n, tot_files)})",
             "Monitor" if tot_files and late_n / tot_files > 0.10 else "Good"),
            ("Duplicate IDs", str(DUP_COUNT), "Monitor" if DUP_COUNT > 10 else "Good"),
            ("Duplicates Resolved", f"{res_n} ({pct(res_n, max(DUP_COUNT, 1))})",
             "Good" if DUP_COUNT == 0 or res_n / max(DUP_COUNT, 1) > 0.8 else "Action Needed"),
        ]
        table_html = """<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <thead><tr style="background:#2c3e50;color:#fff;">
            <th style="text-align:left;padding:8px 10px;font-weight:600">Metric</th>
            <th style="text-align:left;padding:8px 10px;font-weight:600">Value</th>
            <th style="text-align:left;padding:8px 10px;font-weight:600">Status</th>
        </tr></thead><tbody>"""
        for i, (metric, value, status) in enumerate(rows):
            bg = "#f8f9fa" if i % 2 == 1 else "#fff"
            table_html += (
                f'<tr><td style="padding:7px 10px;border-bottom:1px solid #eee;background:{bg}">{metric}</td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid #eee;background:{bg}">{value}</td>'
                f'<td style="padding:7px 10px;border-bottom:1px solid #eee;background:{bg}">{badge(status)}</td></tr>'
            )
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — WORKFORCE PLANNING SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════
def page_workforce_planning():
    # ─── CSS ────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .stat-label { font-size: 0.7rem; color: #777; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
        .stat-value { font-size: 1.1rem; font-weight: 700; }
        .chart-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
        .chart-card-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; border-bottom: 1px solid #eee; background: #f8fafc; }
        .chart-card-head h3 { font-family: 'Segoe UI', sans-serif; font-weight: 700; font-size: 0.88rem; color: #333; margin: 0; }
        .chart-card-head .sub { font-size: 0.65rem; color: #999; margin-top: 2px; }
        .chart-card-body { padding: 12px 20px 16px; }
        .hm-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; }
        .hm-table th { padding: 7px 10px; text-align: center; background: #2c3e50; color: #fff; font-weight: 600; border: 1px solid #ddd; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .hm-table th.rh { text-align: left; }
        .hm-table td { padding: 6px 10px; text-align: center; border: 1px solid #eee; font-weight: 500; }
        .hm-table td.rl { text-align: left; color: #333; font-weight: 600; background: #f8f9fa; font-size: 0.72rem; }
        .hm-hi  { background: #d1fae5; color: #065f46; }
        .hm-md  { background: #fef3c7; color: #92400e; }
        .hm-lo  { background: #ffe4e6; color: #9f1239; }
        .hm-z   { background: #f9fafb; color: #aaa; }
        .hm-legend { display: flex; gap: 16px; margin-top: 12px; justify-content: flex-end; }
        .hm-legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.65rem; color: #777; }
        .hm-legend-swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

    # ─── Constants & Data ──────────────────────────────────────
    GRADE_ORDER = ["Junior", "Mid", "Lead", "Senior", "Manager", "Director"]
    COL_BASE = "#1f77b4"; COL_SCA = "#2ca02c"; COL_SCB = "#d62728"
    COL_GREEN = "#2ca02c"; COL_ROSE = "#d62728"
    COL_SURFACE = "white"; COL_BORDER = "#eee"; COL_TEXT = "#333"; COL_DIM = "#777"

    # Compute base headcount
    exited_ids = set(attrition[attrition["Exit_Date"].notna()]["Employee_ID"])
    active_emp = employees[~employees["Employee_ID"].isin(exited_ids)]
    BASE_HC = active_emp.groupby(["Agency", "Job_Grade"]).size().unstack(fill_value=0)
    for g in GRADE_ORDER:
        if g not in BASE_HC.columns:
            BASE_HC[g] = 0
    BASE_HC = BASE_HC[GRADE_ORDER]
    AGENCIES = sorted(BASE_HC.index.tolist())
    GRADES = [g for g in GRADE_ORDER if g in BASE_HC.columns]
    TOTAL_HC = int(BASE_HC.values.sum())

    # Monthly rates
    exits_with_dates = attrition[attrition["Exit_Date"].notna()]
    date_range_months = max((exits_with_dates["Exit_Date"].max() - exits_with_dates["Exit_Date"].min()).days / 30.44, 1) if len(exits_with_dates) > 0 else 12
    hire_range_months = max((hires["Hire_Date"].max() - hires["Hire_Date"].min()).days / 30.44, 1) if len(hires) > 0 else 12
    BASE_EXIT = len(exits_with_dates) / date_range_months
    BASE_HIRE = len(hires) / hire_range_months

    total_exits_count = len(exits_with_dates) or 1
    AG_EXIT_SH = (exits_with_dates.groupby("Agency").size() / total_exits_count).to_dict()
    total_hires_count = len(hires) or 1
    AG_HIRE_SH = (hires.groupby("Agency").size() / total_hires_count).to_dict()
    GR_EXIT_SH = (exits_with_dates.groupby("Job_Grade").size() / total_exits_count).to_dict()
    GR_HIRE_SH = (hires.groupby("Job_Grade").size() / total_hires_count).to_dict()

    # ─── Projection Engine ─────────────────────────────────────
    def ag_hc(agencies):
        return int(BASE_HC.loc[BASE_HC.index.isin(agencies)].values.sum())

    def project(agencies, attr_delta, hire_delta, freeze_on, freeze_months, horizon):
        series = []
        hc = ag_hc(agencies)
        m_exit = sum(BASE_EXIT * AG_EXIT_SH.get(a, 0.125) for a in agencies) * (1 + attr_delta / 100)
        m_hire = sum(BASE_HIRE * AG_HIRE_SH.get(a, 0.125) for a in agencies) * (1 + hire_delta / 100)
        series.append({"m": 0, "hc": hc, "e": 0, "h": 0})
        for i in range(1, horizon + 1):
            frozen = freeze_on and i <= freeze_months
            e = round(m_exit)
            h = 0 if frozen else round(m_hire)
            hc = max(0, hc - e + h)
            series.append({"m": i, "hc": hc, "e": e, "h": h})
        return series

    def project_heatmap(attr_delta, hire_delta, freeze_on, freeze_months, horizon):
        result = {}
        for ag in AGENCIES:
            result[ag] = {}
            for gr in GRADES:
                v = int(BASE_HC.loc[ag, gr]) if ag in BASE_HC.index else 0
                for i in range(1, horizon + 1):
                    frozen = freeze_on and i <= freeze_months
                    e_m = BASE_EXIT * (1 + attr_delta / 100) * AG_EXIT_SH.get(ag, 0.125) * GR_EXIT_SH.get(gr, 0.16) * 6
                    h_m = 0 if frozen else BASE_HIRE * (1 + hire_delta / 100) * AG_HIRE_SH.get(ag, 0.125) * GR_HIRE_SH.get(gr, 0.16) * 6
                    v = max(0, round(v - e_m + h_m))
                result[ag][gr] = v
        return result

    # ─── Topbar ────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#1f77b4;font-size:1.7rem;margin-bottom:2px'>"
        "Workforce Planning</h1>"
        "<p style='text-align:center;color:#777;font-size:0.85rem;margin-bottom:18px'>"
        "Trends &amp; Monitoring – use the sidebar filters to drill down</p>",
        unsafe_allow_html=True,
    )
    #st.markdown(
    #    f'<h1 style="text-align:center;font-family:Segoe UI,sans-serif;font-weight:800;font-size:1.5rem;'
    #    f'letter-spacing:-0.02em;color:#1e293b;margin-bottom:2px">'
    #    f'Workforce <span style="color:#38bdf8">Planning</span> Engine</h1>'
    #    f'<p style="text-align:center;font-size:0.82rem;color:#64748b;margin-bottom:12px">'
    #    f'Scenario Projection &amp; Headcount Analysis · Base: {TOTAL_HC:,} employees</p>',
    #    unsafe_allow_html=True,
    #)
    st.markdown("---")

    # ─── Persistent scenario state ─────────────────────────────
    if "wp_sc_A" not in st.session_state:
        st.session_state.wp_sc_A = {"attrD": 0, "hireD": 0, "freezeOn": False, "freezeMo": 3}
    if "wp_sc_B" not in st.session_state:
        st.session_state.wp_sc_B = {"attrD": 40, "hireD": 0, "freezeOn": True, "freezeMo": 6}

    # ─── Sidebar Controls ──────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚡ Controls")

        st.markdown("**Scenario**")
        scenario = st.selectbox(
            "Scenario", options=["Base", "Scenario A", "Scenario B"],
            index=0, label_visibility="collapsed", key="wp_sel_scenario",
        )
        _sc_colors = {"Base": COL_BASE, "Scenario A": COL_SCA, "Scenario B": COL_SCB}
        _sc_col = _sc_colors[scenario]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:-8px;margin-bottom:4px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{_sc_col}"></div>'
            f'<span style="font-size:0.78rem;font-weight:600;color:{_sc_col}">{scenario} active</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Agency**")
        ALL_LABEL = "All"
        agency_options = [ALL_LABEL] + AGENCIES
        sel_agency_raw = st.multiselect("Agency filter", options=agency_options, default=[ALL_LABEL], label_visibility="collapsed", key="wp_agency")
        if ALL_LABEL in sel_agency_raw or not sel_agency_raw:
            sel_agencies = AGENCIES
        else:
            sel_agencies = [a for a in sel_agency_raw if a != ALL_LABEL]

        st.markdown("---")
        is_base = scenario == "Base"
        sc_key = "wp_sc_A" if scenario == "Scenario A" else "wp_sc_B"

        st.markdown("**What-If Controls**")
        if is_base:
            st.caption("ℹ️ Switch to Scenario A or B to adjust sliders.")
        else:
            stored = st.session_state[sc_key]
            new_attr = st.slider("Δ Attrition %", min_value=0, max_value=200, value=stored["attrD"], key=f"wp_sl_attr_{sc_key}")
            new_hire = st.slider("Δ Hiring %", min_value=0, max_value=200, value=stored["hireD"], key=f"wp_sl_hire_{sc_key}")
            st.session_state[sc_key]["attrD"] = new_attr
            st.session_state[sc_key]["hireD"] = new_hire

        st.markdown("---")
        st.markdown("**Hiring Freeze**")
        if is_base:
            st.caption("ℹ️ Not available in Base scenario.")
        else:
            stored = st.session_state[sc_key]
            new_freeze = st.toggle("Enable hiring freeze", value=stored["freezeOn"], key=f"wp_freeze_toggle_{sc_key}")
            new_freeze_mo = st.number_input("Freeze duration (months)", min_value=1, max_value=24, value=stored["freezeMo"], disabled=not new_freeze, key=f"wp_freeze_mo_{sc_key}")
            st.session_state[sc_key]["freezeOn"] = new_freeze
            st.session_state[sc_key]["freezeMo"] = new_freeze_mo

        st.markdown("---")
        st.markdown("**Projection Horizon**")
        horizon = st.selectbox("Months", options=[12, 18, 24], index=1, label_visibility="collapsed", key="wp_horizon")

        st.markdown("---")
        st.markdown("**Series Legend**")
        st.markdown(
            f'<div style="display:flex;gap:14px;flex-wrap:wrap;">'
            f'<span style="font-size:0.75rem;color:#777">● <span style="color:#1f77b4">Base</span></span>'
            f'<span style="font-size:0.75rem;color:#777">● <span style="color:#2ca02c">Scenario A</span></span>'
            f'<span style="font-size:0.75rem;color:#777">● <span style="color:#d62728">Scenario B</span></span></div>',
            unsafe_allow_html=True,
        )

    # ─── Run Projections ───────────────────────────────────────
    sA = st.session_state.wp_sc_A
    sB = st.session_state.wp_sc_B

    base_series = project(sel_agencies, 0, 0, False, 0, horizon)
    sc_a_series = project(sel_agencies, sA["attrD"], sA["hireD"], sA["freezeOn"], sA["freezeMo"], horizon)
    sc_b_series = project(sel_agencies, sB["attrD"], sB["hireD"], sB["freezeOn"], sB["freezeMo"], horizon)

    if scenario == "Base":
        active_series = base_series
        active_params = {"attrD": 0, "hireD": 0, "freezeOn": False, "freezeMo": 0}
    elif scenario == "Scenario A":
        active_series = sc_a_series
        active_params = dict(sA)
    else:
        active_series = sc_b_series
        active_params = dict(sB)

    # ─── Live Stats ────────────────────────────────────────────
    current_hc = ag_hc(sel_agencies)
    final_base = base_series[-1]["hc"]
    final_active = active_series[-1]["hc"]
    total_exits_stat = sum(d["e"] for d in active_series)
    total_hires_stat = sum(d["h"] for d in active_series)
    net_change = final_active - final_base
    agency_label = "All Agencies" if len(sel_agencies) == len(AGENCIES) else ", ".join(sel_agencies)

    stat_cols = st.columns(6)
    stat_data = [
        ("Agency", agency_label, COL_BASE),
        ("Current HC", f"{current_hc:,}", COL_TEXT),
        (f"Projected ({horizon}mo)", f"{final_active:,}", COL_TEXT),
        ("Total Hires", f"{total_hires_stat:,}", COL_GREEN),
        ("Total Exits", f"{total_exits_stat:,}", COL_ROSE),
        ("Net Change", f"{'+' if net_change >= 0 else ''}{net_change:,}", COL_GREEN if net_change >= 0 else COL_ROSE),
    ]
    for col, (label, value, color) in zip(stat_cols, stat_data):
        col.markdown(f'<div class="stat-label">{label}</div><div class="stat-value" style="color:{color}">{value}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ─── Chart 1: Headcount Projection ─────────────────────────
    st.markdown(
        '<div class="chart-card"><div class="chart-card-head"><div>'
        '<h3>Headcount Projection</h3><div class="sub">Multi-series comparison over the selected horizon</div>'
        '</div></div></div>', unsafe_allow_html=True,
    )

    base_months = [d["m"] for d in base_series]
    base_hcs = [d["hc"] for d in base_series]
    sca_hcs = [d["hc"] for d in sc_a_series]
    scb_hcs = [d["hc"] for d in sc_b_series]

    all_hc_vals = base_hcs + sca_hcs + scb_hcs
    y_min = min(all_hc_vals); y_max = max(all_hc_vals)
    y_pad = max((y_max - y_min) * 0.08, 10)

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(x=base_months, y=base_hcs, name=f"Base: {base_series[-1]['hc']:,}", mode="lines+markers", line=dict(color=COL_BASE, width=2.5), marker=dict(size=4, color=COL_BASE)))
    fig_proj.add_trace(go.Scatter(x=base_months, y=sca_hcs, name=f"Sc.A: {sc_a_series[-1]['hc']:,}", mode="lines", line=dict(color=COL_SCA, width=2.5, dash="dash")))
    fig_proj.add_trace(go.Scatter(x=base_months, y=scb_hcs, name=f"Sc.B: {sc_b_series[-1]['hc']:,}", mode="lines", line=dict(color=COL_SCB, width=2.5, dash="dot")))

    fig_proj.update_layout(
        xaxis=dict(title="Month", tickmode="array",
                   tickvals=list(range(0, horizon + 1, 3 if horizon > 12 else 1)),
                   ticktext=[f"M{m}" for m in range(0, horizon + 1, 3 if horizon > 12 else 1)],
                   gridcolor=COL_BORDER, zerolinecolor=COL_BORDER, color=COL_DIM),
        yaxis=dict(title="Headcount", gridcolor=COL_BORDER, zerolinecolor=COL_BORDER, color=COL_DIM, range=[max(0, y_min - y_pad), y_max + y_pad]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color=COL_DIM)),
        height=320, margin=dict(l=60, r=30, t=40, b=50),
        plot_bgcolor=COL_SURFACE, paper_bgcolor=COL_SURFACE, font=dict(color=COL_DIM), hovermode="x unified",
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ─── Chart 2: Waterfall ────────────────────────────────────
    st.markdown(
        '<div class="chart-card"><div class="chart-card-head"><div>'
        '<h3>Driver Bridge — Scenario vs Base</h3><div class="sub">Contribution of Hires & Exits to active scenario delta</div>'
        '</div></div></div>', unsafe_allow_html=True,
    )

    base_end = base_series[-1]["hc"]; act_end = active_series[-1]["hc"]
    delta_hires = sum(d["h"] for d in active_series) - sum(d["h"] for d in base_series)
    delta_exits = -(sum(d["e"] for d in active_series) - sum(d["e"] for d in base_series))

    wf_labels = ["Base", "Δ Hires", "Δ Exits", "Scenario"]
    wf_bases = [0, base_end, base_end + delta_hires, 0]
    wf_values = [base_end, delta_hires, delta_exits, act_end]
    wf_colors = [COL_BASE, COL_GREEN if delta_hires >= 0 else COL_ROSE, COL_GREEN if delta_exits >= 0 else COL_ROSE, COL_GREEN if act_end >= base_end else COL_ROSE]

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(x=wf_labels, y=wf_bases, marker_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip"))
    fig_wf.add_trace(go.Bar(
        x=wf_labels, y=[abs(v) for v in wf_values], marker_color=wf_colors, showlegend=False,
        text=[f"{base_end:,}", f"{'+' if delta_hires >= 0 else ''}{delta_hires:,}", f"{'+' if delta_exits >= 0 else ''}{delta_exits:,}", f"{act_end:,}"],
        textposition="outside", textfont=dict(color="#333", size=11),
    ))
    fig_wf.update_layout(
        barmode="stack", xaxis=dict(color=COL_DIM, gridcolor=COL_BORDER),
        yaxis=dict(title="Headcount", color=COL_DIM, gridcolor=COL_BORDER, rangemode="tozero"),
        height=280, margin=dict(l=60, r=30, t=30, b=40),
        plot_bgcolor=COL_SURFACE, paper_bgcolor=COL_SURFACE, font=dict(color=COL_DIM),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ─── Chart 3: Heatmap ─────────────────────────────────────
    st.markdown(
        '<div class="chart-card"><div class="chart-card-head"><div>'
        '<h3>Agency × Grade — Projected Headcount</h3><div class="sub">End-of-horizon snapshot with conditional formatting</div>'
        '</div></div></div>', unsafe_allow_html=True,
    )

    hm_data = project_heatmap(active_params["attrD"], active_params["hireD"], active_params["freezeOn"], active_params["freezeMo"], horizon)

    html = '<table class="hm-table"><thead><tr><th class="rh">Agency \\ Grade</th>'
    for g in GRADES:
        html += f"<th>{g}</th>"
    html += '<th style="border-left:2px solid #ddd">Total</th></tr></thead><tbody>'
    for ag in sel_agencies:
        html += f'<tr><td class="rl">{ag}</td>'
        row_total = 0
        for gr in GRADES:
            v = hm_data.get(ag, {}).get(gr, 0); row_total += v
            cls = "hm-hi" if v >= 40 else "hm-md" if v >= 15 else "hm-lo" if v >= 1 else "hm-z"
            html += f'<td class="{cls}">{v}</td>'
        html += f'<td style="border-left:2px solid #ddd;font-weight:700;color:#333">{row_total}</td></tr>'

    html += '<tr style="border-top:2px solid #ddd"><td class="rl" style="color:#1f77b4;font-weight:700">Total</td>'
    grand = 0
    for gr in GRADES:
        col_sum = sum(hm_data.get(ag, {}).get(gr, 0) for ag in sel_agencies); grand += col_sum
        html += f'<td style="font-weight:700;color:#333">{col_sum}</td>'
    html += f'<td style="border-left:2px solid #ddd;font-weight:700;color:#1f77b4">{grand}</td></tr></tbody></table>'
    html += """<div class="hm-legend">
        <div class="hm-legend-item"><div class="hm-legend-swatch" style="background:#d1fae5"></div>High (≥40)</div>
        <div class="hm-legend-item"><div class="hm-legend-swatch" style="background:#fef3c7"></div>Medium (15–39)</div>
        <div class="hm-legend-item"><div class="hm-legend-swatch" style="background:#ffe4e6"></div>Low (1–14)</div>
        <div class="hm-legend-item"><div class="hm-legend-swatch" style="background:#f9fafb"></div>Zero</div></div>"""
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  NAVIGATION
# ═══════════════════════════════════════════════════════════════
PAGES = {
    "📊 HR Analytics Dashboard": page_hr_dashboard,
    "⚡ Workforce Planning": page_workforce_planning,
}

with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:4px'>🏢 HR Suite</h2>"
        "<p style='font-size:0.78rem;color:#888;margin-bottom:12px'>Select a module below</p>",
        unsafe_allow_html=True,
    )
    selected_page = st.radio(
        "Navigation", options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

# Run selected page
PAGES[selected_page]()
