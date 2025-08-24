
# WorkOrderReport4.0 — Single-file Streamlit app
# Updates requested by @jr3140:
# - Add "Cost Center" (from column N / index 13) between Type and Description
# - Abbreviate header to "CC" in UI tables
# - Remove CSV export entirely
# - Keep PDF at 6 columns (exclude Cost Center from PDF)

import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Visualization
import altair as alt

# PDF (ReportLab + Matplotlib image embed)
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image as RLImage
from reportlab.lib.utils import ImageReader

# ----------------- Static mappings -----------------
TYPE_MAP = {
    '0': 'Break In', '1': 'Maintenance Order', '2': 'Material Repair TMJ Order',
    '3': 'Capital Project', '4': 'Urgent Corrective', '5': 'Emergency Order',
    '6': 'PM Restore/Replace', '7': 'PM Inspection', '8': 'Follow Up Maintenance Order',
    '9': 'Standing W.O. - Do not Delete', 'B': 'Marketing', 'C': 'Cost Improvement',
    'D': 'Design Work - ETO', 'E': 'Plant Work - ETO', 'G': 'Governmental/Regulatory',
    'M': 'Model W.O. - Eq Mgmt', 'N': 'Template W.O. - CBM Alerts', 'P': 'Project',
    'R': 'Rework Order', 'S': 'Shop Order', 'T': 'Tool Order', 'W': 'Case',
    'X': 'General Work Request', 'Y': 'Follow Up Work Request', 'Z': 'System Work Request'
}

_TYPE_COLORS = {
    "Break In": "#d62728",
    "Maintenance Order": "#1f77b4",
    "Urgent Corrective": "#ff7f0e",
    "Emergency Order": "#d62728",
    "PM Restore/Replace": "#2ca02c",
    "PM Inspection": "#2ca02c",
    "Follow Up Maintenance Order": "#d4c720",
    "Project": "#9467bd"
}

# Columns to show on dashboard tables (NOTE: includes Cost Center between Type and Description)
DISPLAY_COLUMNS: List[str] = [
    "Name", "Work Order #", "Sum of Hours", "Type",
    "Cost Center",   # inserted here
    "Description", "Problem"
]


# =========================
# Data loading & prep
# =========================
RENAME_MAP = {
    "WO Number": "Work Order #",
    "Work Order Number": "Work Order #",
    "OrderNumber": "Work Order #",
    "Sum Hours": "Sum of Hours",
    "Hours": "Sum of Hours",
    "Prod Date": "Production Date",
    "Prod_Date": "Production Date",
}

def _try_rename(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: RENAME_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    return df

def load_timeworkbook(xlsx_file) -> pd.DataFrame:
    """Load the uploaded 'Time on Work Order' workbook and normalize columns."""
    df = pd.read_excel(xlsx_file, engine="openpyxl")
    df = _try_rename(df)

    # Standardize key columns if present
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).map(lambda x: TYPE_MAP.get(x, x)).fillna("")
    if "Sum of Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)

    # Derive Production Date if possible
    if "Production Date" in df.columns:
        df["Production Date"] = pd.to_datetime(df["Production Date"], errors="coerce").dt.date
    elif "Date" in df.columns:
        df["Production Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    else:
        # allow app to run without date — single bucket
        df["Production Date"] = pd.NaT

    # --- Cost Center from column "N" (index 13) or existing header ---
    if "Cost Center" in df.columns:
        df["Cost Center"] = df["Cost Center"].astype(str).str.strip()
    else:
        try:
            col_n = df.columns[13]  # 0-based index; Excel column N
            df["Cost Center"] = df[col_n].astype(str).str.strip()
        except Exception:
            df["Cost Center"] = pd.NA

    # Fallbacks for essential text columns
    for col in ["Name", "Work Order #", "Description", "Problem"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    return df


def prepare_report_data(time_df: pd.DataFrame, selected_date) -> Dict[str, Any]:
    """Filter by date (if available), build per-craft group payloads and a full detail table."""
    df = time_df.copy()

    # Filter by selected date if column has real dates
    if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df["Production Date"], errors="coerce")):
        if pd.notna(selected_date):
            df = df[pd.to_datetime(df["Production Date"], errors="coerce").dt.date == selected_date]

    # Establish a craft grouping column if present
    craft_col = None
    for cand in ["Craft", "Craft Description", "CraftDescription"]:
        if cand in df.columns:
            craft_col = cand
            break
    if craft_col is None:
        craft_col = "_Craft"
        df[craft_col] = "All Crafts"

    # Ensure numeric hours for aggregation
    df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)

    groups: List[Tuple[str, Dict[str, Any]]] = []
    for craft_name, g in df.groupby(craft_col, dropna=False):
        g = g.copy()
        # Slice columns for dashboard display — keep order
        present_cols = [c for c in DISPLAY_COLUMNS if c in g.columns]
        detail = g[present_cols].copy()
        groups.append((str(craft_name), {"detail": detail}))

    full_detail = pd.concat([p["detail"] for _, p in groups], axis=0) if groups else df[DISPLAY_COLUMNS].copy()

    return {"groups": groups, "full_detail": full_detail}


# =========================
# Dashboard helpers
# =========================
def _craft_dashboard_block(df_detail: pd.DataFrame):
    if df_detail is None or df_detail.empty:
        return
    df = df_detail.copy()
    if "Sum of Hours" not in df.columns:
        return
    df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
    agg = (df.groupby("Type", dropna=False)["Sum of Hours"]
             .sum()
             .reset_index()
             .rename(columns={"Sum of Hours": "hours"})
             .sort_values("hours", ascending=False))
    total = float(agg["hours"].sum()) if not agg.empty else 0.0
    agg["percent"] = np.where(total > 0, (agg["hours"]/total)*100.0, 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Hours", f"{total:,.2f}")
    top_type = "-" if agg.empty else str(agg.iloc[0]["Type"])
    top_pct = 0.0 if agg.empty else float(agg.iloc[0]["percent"])
    c2.metric("Top Type", top_type)
    c3.metric("Top Type %", f"{top_pct:.1f}%")

    # Charts
    color_scale = alt.Scale(domain=list(_TYPE_COLORS.keys()), range=list(_TYPE_COLORS.values()))
    base = alt.Chart(agg).mark_bar().encode(
        x=alt.X("Type:N", sort="-y"),
        tooltip=[alt.Tooltip("Type:N"), alt.Tooltip("hours:Q", format=",.2f")]
    )

    st.caption("Hours by Work Order Type")
    st.altair_chart(
        base.encode(y=alt.Y("hours:Q", title="Hours"),
                    color=alt.Color("Type:N", scale=color_scale)),
        use_container_width=True
    )

    st.caption("% of Craft Hours by Type")
    st.altair_chart(
        base.encode(y=alt.Y("percent:Q", title="% of Craft", axis=alt.Axis(format="~s")),
                    color=alt.Color("Type:N", scale=color_scale)),
        use_container_width=True
    )


def _auto_height(df: pd.DataFrame, row_height: int = 28, min_h: int = 220, max_h: int = 680) -> int:
    rows = max(1, len(df))
    h = int(rows * row_height) + 60
    return max(min_h, min(max_h, h))


# =========================
# PDF helpers
# =========================
def _df_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """Return a PDF-safe, six-column view (exclude Cost Center)."""
    out = df.copy()
    pdf_cols = ["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"]
    pdf_cols = [c for c in pdf_cols if c in out.columns]
    out = out[pdf_cols]
    if "Sum of Hours" in out.columns:
        out["Sum of Hours"] = (
            pd.to_numeric(out["Sum of Hours"], errors="coerce")
              .fillna(0.0)
              .map(lambda x: f"{x:.2f}")
        )
    return out


def build_pdf(report: Dict[str, Any], date_label: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                            leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=24)
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontName="Helvetica",
                                fontSize=8, leading=10, wordWrap="CJK", spaceBefore=0, spaceAfter=0)
    header_style = ParagraphStyle("header", parent=styles["Normal"], fontName="Helvetica-Bold",
                                  fontSize=9, leading=11, spaceBefore=0, spaceAfter=0)

    story = []
    title = Paragraph(f"<b>Work Order Reporting</b> — Report for {date_label}", styles["Title"])
    story.append(title); story.append(Spacer(1, 6))

    # Table column widths (6 columns; Description-sized as 300)
    px_widths = [200, 90, 90, 200, 300, 420]
    page_width, _ = landscape(letter)
    usable = page_width - doc.leftMargin - doc.rightMargin
    scale = usable / sum(px_widths)
    col_widths = [w * scale for w in px_widths]

    header_bg = colors.HexColor("#f0f0f0"); grid_color = colors.HexColor("#d0d0d0")

    # Per-craft sections with a small bar chart + breakdown
    for craft_name, payload in report["groups"]:
        story.append(Paragraph(f"<b>{craft_name}</b>", styles["Heading2"]))

        raw = payload["detail"].copy()
        try:
            tmp = raw.copy()
            tmp["Sum of Hours"] = pd.to_numeric(tmp["Sum of Hours"], errors="coerce").fillna(0.0)
            agg_pdf = (tmp.groupby("Type", dropna=False)["Sum of Hours"].sum()
                         .reset_index()
                         .rename(columns={"Sum of Hours":"hours"})
                         .sort_values("hours", ascending=False))

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.8, 2.6), dpi=150)
            bar_colors = [_TYPE_COLORS.get(str(t), "#1f77b4") for t in agg_pdf["Type"]]
            ax.bar(agg_pdf["Type"], agg_pdf["hours"], color=bar_colors)
            ax.set_title("Hours by Work Order Type")
            ax.set_ylabel("Hours")
            ax.grid(axis="y", linestyle=":", linewidth=0.5)
            for tick in ax.get_xticklabels():
                tick.set_rotation(25)
                tick.set_ha("right")
            fig.tight_layout()

            import io as _io
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_reader = ImageReader(buf)
            chart_width = usable * 0.72
            chart_height = chart_width * 0.36
            story.append(RLImage(img_reader, width=chart_width, height=chart_height))
            story.append(Spacer(1, 6))

            # Breakdown table
            bd_headers = [Paragraph("Type", header_style), Paragraph("Hours", header_style)]
            bd_data = [bd_headers]
            for _, r in agg_pdf.iterrows():
                bd_data.append([Paragraph(str(r["Type"]), cell_style),
                                Paragraph(f"{float(r['hours']):.2f}", cell_style)])
            bd_tbl = Table(bd_data, colWidths=[usable*0.45, usable*0.15], repeatRows=1)
            bd_tbl.setStyle(TableStyle([
                ("FONT", (0,0), (-1,-1), "Helvetica"),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("BACKGROUND", (0,0), (-1,0), header_bg),
                ("GRID", (0,0), (-1,-1), 0.25, grid_color),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
                ("ALIGN", (0,1), (0,-1), "RIGHT"),
                ("ALIGN", (1,1), (1,-1), "LEFT"),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
                ("RIGHTPADDING", (0,0), (-1,-1), 4),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ]))
            story.append(bd_tbl)
            story.append(Spacer(1, 8))
        except Exception as _e:
            logging.exception("PDF chart embed failed")

        df = _df_for_pdf(raw)
        data: List[List[Any]] = []
        headers = [Paragraph(h, header_style) for h in df.columns]
        data.append(headers)
        for _, row in df.iterrows():
            row_vals = []
            for col in df.columns:
                val = "" if pd.isna(row[col]) else str(row[col])
                if col in ("Work Order #", "Sum of Hours"):
                    row_vals.append(val)
                else:
                    row_vals.append(Paragraph(val, cell_style))
            data.append(row_vals)
        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0), header_bg),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("GRID", (0,0), (-1,-1), 0.25, grid_color),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ("ALIGN", (1,1), (1,-1), "RIGHT"),
            ("ALIGN", (2,1), (2,-1), "RIGHT"),
        ]))
        story.append(tbl); story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Work Order Reporting App", layout="wide")
st.title("Work Order Reporting App")

with st.sidebar:
    st.header("Upload file")
    time_file = st.file_uploader("Time on Work Order (.xlsx) – REQUIRED", type=["xlsx"], key="time")

if not time_file:
    st.sidebar.info("⬆️ Upload the **Time on Work Order** export to proceed.")
    st.stop()

try:
    time_df = load_timeworkbook(time_file)
except Exception as e:
    st.sidebar.error(f"File load error: {e}")
    st.stop()

# Date selection (if available), else single "All Data"
if "Production Date" in time_df.columns and not time_df["Production Date"].dropna().empty:
    dates = sorted(pd.to_datetime(time_df["Production Date"]).dt.date.unique())
    date_labels = [datetime.strftime(pd.to_datetime(d), "%m/%d/%Y") for d in dates]
    label_to_date = dict(zip(date_labels, dates))
    selected_label = st.selectbox("Select Production Date", options=date_labels, index=len(date_labels)-1)
    selected_date = label_to_date[selected_label]
else:
    selected_label = "All Data"
    selected_date = pd.NaT
    st.info("No valid 'Production Date' found — showing all data.")

report = prepare_report_data(time_df, selected_date)

# Export PDF
with st.sidebar:
    st.subheader("Export")
    st.download_button(
        "Generate & Download PDF (Landscape)",
        data=build_pdf(report, selected_label),
        file_name=f"workorder_report_{selected_label.replace('/', '-')}.pdf",
        mime="application/pdf",
    )

st.markdown(f"### Report for {selected_label}")

# Column config — note 'Cost Center' label shown as 'CC'
col_cfg = {
    "Name": st.column_config.TextColumn("Name", width=200),
    "Work Order #": st.column_config.TextColumn("Work Order #", width=90),
    "Sum of Hours": st.column_config.NumberColumn("Sum of Hours", format="%.2f", width=90),
    "Type": st.column_config.TextColumn("Type", width=200),

    "Cost Center": st.column_config.TextColumn("CC", width=120),  # abbreviated header

    "Description": st.column_config.TextColumn("Description", width=300),
    "Problem": st.column_config.TextColumn("Problem", width=420),
}

# Render per-craft sections
for craft_name, payload in report["groups"]:
    st.markdown(f"#### {craft_name}")
    df_detail = payload["detail"]

    # Mini dashboard
    _craft_dashboard_block(df_detail)

    # Table
    st.dataframe(
        df_detail,
        use_container_width=True,
        hide_index=True,
        height=_auto_height(df_detail),
        column_config=col_cfg,
    )
    st.markdown("---")
