import io
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st

# PDF
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Charts
import altair as alt

# ----------------- Static mappings (adapt to your env if needed) -----------------
CRAFT_ORDER = [
    'Turns', 'EAF Mech Days', 'EAF Elec Days', 'AOD Mech Days', 'AOD Elec Days',
    'Alloy Mech Days', 'Caster Mech Days', 'Caster Elec Days', 'WTP Mech Days',
    'Baghouse Mech Days', 'Preheater Elec Days', 'Segment Shop',
    'Utilities Mech Days', 'HVAC Elec Days'
]

# Minimal address book sample; replace with your actual source or keep as-is if you upload Time report with names.
ADDRESS_BOOK = [
    {'AddressBookNumber': '1103079', 'Name': 'CONKEL, JOHNATHON J', 'Craft Description': 'Alloy Mech Days'},
    {'AddressBookNumber': '817150',  'Name': 'PETERS, JESSE DANIEL',   'Craft Description': 'AOD Elec Days'},
]

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

DISPLAY_COLUMNS = ["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"]

REQUIRED_TIME_COLUMNS = [
    "AddressBookNumber", "Name", "Production Date", "OrderNumber", "Sum of Hours.",
    "Hours Estimated", "Status", "Type", "PMFrequency", "Description", "Problem",
    "Department", "Location", "Equipment", "PM Number", "PM"
]

# ----------------- Load & normalize data -----------------
def _find_header_row(df_raw: pd.DataFrame) -> int:
    first_col = df_raw.columns[0]
    mask = df_raw[first_col].astype(str).str.strip() == "AddressBookNumber"
    idx = df_raw.index[mask].tolist()
    if idx:
        return idx[0]
    for i in range(min(10, len(df_raw))):
        row_vals = df_raw.iloc[i].astype(str).str.strip().tolist()
        if "AddressBookNumber" in row_vals and "Production Date" in row_vals:
            return i
    raise ValueError("Could not locate header row containing 'AddressBookNumber'.")

def _read_excel_twice(file) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = file.read()
    df_raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    df_hdr = pd.read_excel(io.BytesIO(data))
    return df_raw, df_hdr

def load_timeworkbook(file_like) -> pd.DataFrame:
    df_raw, _ = _read_excel_twice(file_like)
    header_row = _find_header_row(df_raw)
    file_like.seek(0)
    df = pd.read_excel(file_like, header=header_row)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    for c in REQUIRED_TIME_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["AddressBookNumber"] = df["AddressBookNumber"].astype(str).str.strip()
    if "Production Date" in df.columns:
        df["Production Date"] = pd.to_datetime(df["Production Date"], errors="coerce").dt.date

    # Normalize hours
    if "Sum of Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce")
    elif "Sum of Hours." in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours."], errors="coerce")
    elif "Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    else:
        df["Sum of Hours"] = pd.NA

    # Normalize work order #
    if "Work Order Number" in df.columns:
        base_wo = df["Work Order Number"]
    elif "OrderNumber" in df.columns:
        base_wo = df["OrderNumber"]
    elif "WO Number" in df.columns:
        base_wo = df["WO Number"]
    elif "WorkOrderNumber" in df.columns:
        base_wo = df["WorkOrderNumber"]
    else:
        base_wo = pd.Series([pd.NA] * len(df))
    df["Work Order #"] = base_wo.astype(str).str.replace(r"\.0$", "", regex=True)

    if "Problem" not in df.columns:
        df["Problem"] = pd.NA

    return df

def get_craft_order_df() -> pd.DataFrame:
    return pd.DataFrame({"Craft Description": CRAFT_ORDER})

def get_address_book_df() -> pd.DataFrame:
    df = pd.DataFrame(ADDRESS_BOOK)[["AddressBookNumber", "Name", "Craft Description"]]
    df["AddressBookNumber"] = df["AddressBookNumber"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Craft Description"] = df["Craft Description"].astype(str).str.strip()
    return df

def _apply_craft_category(df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    order = order_df["Craft Description"].tolist()
    seen, ordered = set(), []
    for c in order:
        if c not in seen:
            ordered.append(c); seen.add(c)
    categories = ordered + ["Unassigned"]
    df["Craft Description"] = df["Craft Description"].fillna("Unassigned")
    df["Craft Description"] = pd.Categorical(df["Craft Description"], categories=categories, ordered=True)
    return df

def _clean_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _map_type(value):
    key = _clean_code(value)
    if key is None:
        return pd.NA
    out = TYPE_MAP.get(key, key)
    if isinstance(out, str) and out.strip().lower() == "inspection maintenance order":
        return "PM Inspection"
    return out

def prepare_report_data(time_df: pd.DataFrame,
                        addr_df: pd.DataFrame,
                        craft_order_df: pd.DataFrame,
                        selected_date) -> Dict[str, Any]:
    f = time_df[time_df["Production Date"] == selected_date].copy()
    f["AddressBookNumber"] = f["AddressBookNumber"].astype(str).str.strip()
    addr_df["AddressBookNumber"] = addr_df["AddressBookNumber"].astype(str).str.strip()
    merged = f.merge(
        addr_df[["AddressBookNumber", "Craft Description", "Name"]].rename(columns={"Name": "AB_Name"}),
        on="AddressBookNumber",
        how="left"
    )
    merged["Name"] = merged["Name"].fillna(merged["AB_Name"])
    merged = merged.drop(columns=["AB_Name"])

    unmapped = []
    mask_unmapped = merged["Craft Description"].isna() | (merged["Craft Description"].astype(str).str.len() == 0)
    if mask_unmapped.any():
        unmapped = merged.loc[mask_unmapped, ["AddressBookNumber", "Name"]].drop_duplicates().to_dict("records")
    merged.loc[mask_unmapped, "Craft Description"] = "Unassigned"

    merged = _apply_craft_category(merged, craft_order_df)

    for col in DISPLAY_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged["Type"] = merged["Type"].apply(_map_type)

    merged = merged.sort_values(["Craft Description", "Name", "Work Order #"])
    merged["Sum of Hours"] = pd.to_numeric(merged["Sum of Hours"], errors="coerce").round(2)

    groups_payload: List = []
    for craft in list(merged["Craft Description"].cat.categories):
        g_detail = merged[merged["Craft Description"] == craft][DISPLAY_COLUMNS].copy()
        if g_detail.empty:
            continue
        groups_payload.append((str(craft), {"detail": g_detail}))

    full_detail = merged[DISPLAY_COLUMNS].copy()
    return {"groups": groups_payload, "full_detail": full_detail, "unmapped_people": unmapped}

def _auto_height(df: pd.DataFrame) -> int:
    rows = len(df) + 1
    row_px = 35
    header_px = 40
    return min(header_px + rows * row_px, 20000)

# ----------------- PDF helpers & chart export -----------------
def _df_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Sum of Hours" in out.columns:
        out["Sum of Hours"] = pd.to_numeric(out["Sum of Hours"], errors="coerce").fillna(0).map(lambda x: f"{x:.2f}")
    return out

# Shared type colors for dashboard and PDF charts
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

def make_hours_by_type_chart(df_detail: pd.DataFrame):
    """Build a bar chart of total hours by Type with consistent colors."""
    if df_detail is None or df_detail.empty:
        return None
    df = df_detail.copy()
    df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
    agg = (df.groupby("Type", dropna=False)["Sum of Hours"]
             .sum()
             .reset_index()
             .rename(columns={"Sum of Hours": "hours"})
             .sort_values("hours", ascending=False))
    if agg.empty:
        return None
    base = alt.Chart(agg).encode(
        x=alt.X("Type:N", sort="-y", title="Work Order Type"),
        y=alt.Y("hours:Q", title="Hours"),
        tooltip=[alt.Tooltip("Type:N"), alt.Tooltip("hours:Q", format=".2f")]
    )
    color_scale = alt.Scale(domain=list(_TYPE_COLORS.keys()), range=list(_TYPE_COLORS.values()))
    return base.mark_bar().encode(color=alt.Color("Type:N", scale=color_scale))

def altair_to_png_bytes(chart, scale: float = 2.0) -> bytes | None:
    """
    Convert an Altair chart to PNG bytes using altair-saver + vl-convert-python.
    Returns None if conversion fails (fail-soft).
    """
    if chart is None:
        return None
    try:
        from altair_saver import save
        buf = io.BytesIO()
        save(chart, fp=buf, fmt="png", scale=scale)  # uses vl-convert backend if installed
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def build_pdf(report: Dict[str, Any], date_label: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(letter),
        leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=24
    )
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontName="Helvetica",
                                fontSize=8, leading=10, wordWrap="CJK", spaceBefore=0, spaceAfter=0)
    header_style = ParagraphStyle("header", parent=styles["Normal"], fontName="Helvetica-Bold",
                                  fontSize=9, leading=11, spaceBefore=0, spaceAfter=0)

    story = []
    title = Paragraph(f"<b>Work Order Reporting App</b> — Report for {date_label}", styles["Title"])
    story.append(title); story.append(Spacer(1, 6))

    # --------- All-crafts dashboard image on page 1 ---------
    full_df = report.get("full_detail")
    if full_df is not None and not getattr(full_df, "empty", True):
        try:
            chart = make_hours_by_type_chart(full_df)
            png = altair_to_png_bytes(chart, scale=2.0)
        except Exception:
            png = None
        if png:
            page_width, _ = landscape(letter)
            usable = page_width - doc.leftMargin - doc.rightMargin
            img = RLImage(io.BytesIO(png))
            img._restrictSize(usable, 260)  # keep aspect ratio, cap height
            story.append(Paragraph("<b>Hours by Work Order Type — All Crafts</b>", styles["Heading2"]))
            story.append(img); story.append(Spacer(1, 10))

    # --------- Table layout ---------
    px_widths = [200, 90, 90, 200, 300, 420]  # Description reduced to 300 px
    page_width, _ = landscape(letter)
    usable = page_width - doc.leftMargin - doc.rightMargin
    scale = usable / sum(px_widths)
    col_widths = [w * scale for w in px_widths]
    header_bg = colors.HexColor("#f0f0f0"); grid_color = colors.HexColor("#d0d0d0")

    # --------- Per-craft sections ---------
    for craft_name, payload in report["groups"]:
        story.append(Paragraph(f"<b>{craft_name}</b>", styles["Heading2"]))
        df = _df_for_pdf(payload["detail"])

        # Per-craft chart (fail-soft)
        try:
            c_png = altair_to_png_bytes(make_hours_by_type_chart(payload["detail"]), scale=2.0)
        except Exception:
            c_png = None
        if c_png:
            c_img = RLImage(io.BytesIO(c_png))
            c_img._restrictSize(usable, 220)
            story.append(c_img); story.append(Spacer(1, 6))

        data = []
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

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Work Order Reporting App", layout="wide")
st.title("Work Order Reporting App")

with st.sidebar:
    st.header("Upload file")
    time_file = st.file_uploader("Time on Work Order (.xlsx) – REQUIRED", type=["xlsx"], key="time")

if not time_file:
    st.sidebar.info("⬆️ Upload the **Time on Work Order** export to proceed.")
    st.stop()

# Load inputs
try:
    time_df = load_timeworkbook(time_file)
    craft_df = get_craft_order_df()
    addr_df = get_address_book_df()
except Exception as e:
    st.sidebar.error(f"File load error: {e}")
    st.stop()

if "Production Date" not in time_df.columns or time_df["Production Date"].dropna().empty:
    st.sidebar.error("No valid 'Production Date' values found in the Time on Work Order file.")
    st.stop()

dates = sorted(pd.to_datetime(time_df["Production Date"]).dt.date.unique())
date_labels = [datetime.strftime(pd.to_datetime(d), "%m/%d/%Y") for d in dates]
label_to_date = dict(zip(date_labels, dates))
selected_label = st.selectbox("Select Production Date", options=date_labels, index=len(date_labels)-1)
selected_date = label_to_date[selected_label]

# Build current report for on-screen view (not for PDF generation timing)
report_preview = prepare_report_data(time_df, addr_df, craft_df, selected_date)

# ----------------- NEW: Two-step PDF flow to avoid stale bytes -----------------
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_label" not in st.session_state:
    st.session_state.pdf_label = None

def build_pdf_now():
    # Recompute JUST-IN-TIME to ensure charts & data match current selection
    fresh_report = prepare_report_data(time_df, addr_df, craft_df, selected_date)
    st.session_state.pdf_label = selected_label
    st.session_state.pdf_bytes = build_pdf(fresh_report, selected_label)

with st.sidebar:
    st.subheader("Export")
    st.button("Build PDF with charts", on_click=build_pdf_now)

    # Optional: quick self-test for headless chart export (visible in the sidebar)
    with st.expander("Chart export self-test"):
        try:
            from altair_saver import save
            _df = pd.DataFrame({"Type": ["A", "B"], "hours": [1, 2]})
            _chart = alt.Chart(_df).mark_bar().encode(x="Type:N", y="hours:Q")
            _buf = io.BytesIO()
            save(_chart, fp=_buf, fmt="png", scale=1.5)
            st.caption("Altair PNG export: ✅ OK")
        except Exception as e:
            st.error(f"Altair PNG export failed: {e}")
            st.info("Ensure requirements include: altair-saver, vl-convert-python")

    if st.session_state.pdf_bytes is not None:
        st.success(f"PDF ready for {st.session_state.pdf_label}")
        st.download_button(
            "Download PDF (Landscape)",
            data=st.session_state.pdf_bytes,
            file_name=f"workorder_report_{st.session_state.pdf_label.replace('/', '-')}.pdf",
            mime="application/pdf",
            key="download_pdf_ready",
        )

# ----------------- On-screen dashboard & tables -----------------
st.markdown(f"### Report for {selected_label}")

# Per-craft on-screen mini dashboard and table
col_cfg = {
    "Name": st.column_config.TextColumn("Name", width=200),
    "Work Order #": st.column_config.TextColumn("Work Order #", width=90),
    "Sum of Hours": st.column_config.NumberColumn("Sum of Hours", format="%.2f", width=90),
    "Type": st.column_config.TextColumn("Type", width=200),
    "Description": st.column_config.TextColumn("Description", width=300),
    "Problem": st.column_config.TextColumn("Problem", width=420),
}

def _craft_dashboard_block(df_detail: pd.DataFrame):
    if df_detail is None or df_detail.empty:
        return
    df = df_detail.copy()
    df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
    agg = (df.groupby("Type", dropna=False)["Sum of Hours"]
             .sum()
             .reset_index()
             .rename(columns={"Sum of Hours": "hours"})
             .sort_values("hours", ascending=False))
    total = float(agg["hours"].sum())
    agg["percent"] = np.where(total > 0, (agg["hours"]/max(total, 1e-9))*100.0, 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Hours", f"{total:,.2f}")
    top_type = agg.iloc[0]["Type"] if not agg.empty else "-"
    top_pct  = agg.iloc[0]["percent"] if not agg.empty else 0.0
    c2.metric("Top Type", f"{top_type}")
    c3.metric("% in Top Type", f"{top_pct:.1f}%")

    base = alt.Chart(agg).encode(
        x=alt.X("Type:N", sort="-y", title="Work Order Type"),
        y=alt.Y("hours:Q", title="Hours"),
        tooltip=[alt.Tooltip("Type:N"), alt.Tooltip("hours:Q", format=".2f")]
    )
    color_scale = alt.Scale(domain=list(_TYPE_COLORS.keys()), range=list(_TYPE_COLORS.values()))
    st.altair_chart(base.mark_bar().encode(color=alt.Color("Type:N", scale=color_scale)), use_container_width=True)

for craft_name, payload in report_preview["groups"]:
    st.markdown(f"#### {craft_name}")
    df_detail = payload["detail"]
    _craft_dashboard_block(df_detail)
    st.dataframe(
        df_detail,
        use_container_width=True,
        hide_index=True,
        height=_auto_height(df_detail),
        column_config=col_cfg
    )
    st.markdown("---")

# CSV export of filtered detail
if not report_preview["full_detail"].empty:
    csv = report_preview["full_detail"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered detail (CSV)",
        data=csv,
        file_name=f"workorder_detail_{selected_label.replace('/', '-')}.csv",
        mime="text/csv",
    )
