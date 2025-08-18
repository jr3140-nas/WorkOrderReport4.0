import io
from datetime import datetime
from typing import Dict, Any
import logging, List

import pandas as pd
import numpy as np
import streamlit as st

# PDF
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---- Hard-coded mappings (from your uploaded files at build time) ----
CRAFT_ORDER = ['Turns', 'EAF Mech Days', 'EAF Elec Days', 'AOD Mech Days', 'AOD Elec Days', 'Alloy Mech Days', 'Caster Mech Days', 'Caster Elec Days', 'WTP Mech Days', 'Baghouse Mech Days', 'Preheater Elec Days', 'Segment Shop', 'Utilities Mech Days', 'HVAC Elec Days']
ADDRESS_BOOK = [{'AddressBookNumber': '1103079', 'Name': 'CONKEL, JOHNATHON J', 'Craft Description': 'Alloy Mech Days'}, {'AddressBookNumber': '817150', 'Name': 'PETERS, JESSE DANIEL', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '648991', 'Name': 'JONES, TERRELL D.', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '136792', 'Name': 'MCKINNEY, CHRIS ALVIE', 'Craft Description': 'AOD Mech Days'}, {'AddressBookNumber': '1142730', 'Name': 'CHRISTERSON, NATHANIEL BENJAMEN', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1150094', 'Name': 'WRIGHT, KEVIN BRADLEY', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1064305', 'Name': 'DALTON II, JEFFERY WAYNE', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1115109', 'Name': 'GANDER, ANTHONY T', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1055943', 'Name': 'HEFFELMIRE, RONALD SCOTT', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '1112813', 'Name': 'KOONS, ANDREW LEWIS ALAN', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '95636', 'Name': 'MORRISON, GEORGE D.', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '586013', 'Name': 'DENNIS, SHAWN MICHAEL', 'Craft Description': 'EAF Elec Days'}, {'AddressBookNumber': '1137121', 'Name': 'STEWART, THOMAS JASON', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '1106595', 'Name': 'WASH, MICHAEL DAVID', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '178909', 'Name': 'LEMASTER, DANIEL M.', 'Craft Description': 'HVAC Elec Days'}, {'AddressBookNumber': '1115133', 'Name': 'BROCK, TREVOR COLE', 'Craft Description': 'Preheater Elec Days'}, {'AddressBookNumber': '133760', 'Name': 'BRIGHTWELL, JEFFERY W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '173665', 'Name': 'CRAIG, JAMES D.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '336719', 'Name': 'DEEN, ALAN J.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1151409', 'Name': 'DEMAREE, MATTHEW CHRISTOPHER', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '848802', 'Name': 'KLOSS, CHARLES W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '95644', 'Name': 'SMITH, JAMES M.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1104469', 'Name': 'WATSON, JACOB LEYTON', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1103976', 'Name': 'BAUGHMAN, THOMAS BRUCE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1066095', 'Name': 'HELTON, MICHAEL AJ', 'Craft Description': 'Turns'}, {'AddressBookNumber': '44231', 'Name': 'REED, BRIAN L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1167030', 'Name': 'STROUD, MATTHEW T.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '164380', 'Name': 'WARREN, MARK L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '185050', 'Name': 'WHOBREY, BRADLEY G.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1103132', 'Name': 'WILLIAMS II, STEVEN FOSTER', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1106747', 'Name': 'BANTA, BENJAMIN GAYLE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1165384', 'Name': 'BOHART, WILLIAM M.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1144250', 'Name': 'CAREY, JOSEPH MICHAEL', 'Craft Description': 'Turns'}, {'AddressBookNumber': '770363', 'Name': 'CORMAN, DAVID H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1149608', 'Name': 'DIEDERICH, JOSEPH W', 'Craft Description': 'Turns'}, {'AddressBookNumber': '193471', 'Name': 'GRAY, DENNIS C.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '109866', 'Name': 'HOWARD, LARRY D.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1141761', 'Name': 'PHILLIPS, TIMOTHY CRAIG RYAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '272006', 'Name': 'SEE, JOHN JOURDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '106260', 'Name': 'SPILLMAN, WILLIAM H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1131299', 'Name': 'STEWART, BRADFORD LEE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1109876', 'Name': 'STOKES, MATHEW DAVID', 'Craft Description': 'Turns'}, {'AddressBookNumber': '234448', 'Name': 'THOMAS, CODY JORDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1107352', 'Name': 'WATKINS, KENNETH EDWARD', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1096665', 'Name': 'ATWELL, TALON BRADLEY', 'Craft Description': 'Turns'}, {'AddressBookNumber': '108986', 'Name': 'ROGERS, CHARLES D.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '206092', 'Name': 'TURNER, SHANE M.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '1089377', 'Name': 'ROSE, CAMERON CHASE', 'Craft Description': 'WTP Mech Days'}]
TYPE_MAP = {'0': 'Break In', '1': 'Maintenance Order', '2': 'Material Repair TMJ Order', '3': 'Capital Project', '4': 'Urgent Corrective', '5': 'Emergency Order', '6': 'PM Restore/Replace', '7': 'PM Inspection', '8': 'Follow Up Maintenance Order', '9': 'Standing W.O. - Do not Delete', 'B': 'Marketing', 'C': 'Cost Improvement', 'D': 'Design Work - ETO', 'E': 'Plant Work - ETO', 'G': 'Governmental/Regulatory', 'M': 'Model W.O. - Eq Mgmt', 'N': 'Template W.O. - CBM Alerts', 'P': 'Project', 'R': 'Rework Order', 'S': 'Shop Order', 'T': 'Tool Order', 'W': 'Case', 'X': 'General Work Request', 'Y': 'Follow Up Work Request', 'Z': 'System Work Request'}

DISPLAY_COLUMNS = ["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"]

REQUIRED_TIME_COLUMNS = [
    "AddressBookNumber", "Name", "Production Date", "OrderNumber", "Sum of Hours.",
    "Hours Estimated", "Status", "Type", "PMFrequency", "Description", "Problem",
    "Department", "Location", "Equipment", "PM Number", "PM"
]

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

    # Normalize 'Sum of Hours'
    if "Sum of Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce")
    elif "Sum of Hours." in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours."], errors="coerce")
    elif "Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    else:
        df["Sum of Hours"] = pd.NA

    # Normalize work order into 'Work Order #'
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

# ---------- PDF builder with wrapping & fit-to-width ----------
def _df_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Sum of Hours" in out.columns:
        out["Sum of Hours"] = pd.to_numeric(out["Sum of Hours"], errors="coerce").fillna(0).map(lambda x: f"{x:.2f}")
    return out

def build_pdf(report: Dict[str, Any], date_label: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=24)
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, wordWrap="CJK", spaceBefore=0, spaceAfter=0)
    header_style = ParagraphStyle("header", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11, spaceBefore=0, spaceAfter=0)

    story = []
    title = Paragraph(f"<b>Work Order Reporting App</b> — Report for {date_label}", styles["Title"])
    story.append(title); story.append(Spacer(1, 6))

    # Column widths (Description reduced to 300)
    px_widths = [200, 90, 90, 200, 300, 420]
    page_width, _ = landscape(letter)
    usable = page_width - doc.leftMargin - doc.rightMargin
    scale = usable / sum(px_widths)
    col_widths = [w * scale for w in px_widths]

    header_bg = colors.HexColor("#f0f0f0"); grid_color = colors.HexColor("#d0d0d0")

    for craft_name, payload in report["groups"]:
    story.append(Paragraph(f"<b>{craft_name}</b>", styles["Heading2"]))
    # --- PDF: Hours bar chart + breakdown (hardened) ---
    raw = payload["detail"]
    try:
        tmp = raw.copy()
        tmp["Sum of Hours"] = pd.to_numeric(tmp["Sum of Hours"], errors="coerce").fillna(0.0)
        agg_pdf = (tmp.groupby("Type", dropna=False)["Sum of Hours"].sum().reset_index().rename(columns={"Sum of Hours":"hours"}).sort_values("hours", ascending=False))
        # Matplotlib headless backend
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.8, 2.6), dpi=150)
        colors = [_TYPE_COLORS.get(str(t), "#1f77b4") for t in agg_pdf["Type"]]
        ax.bar(agg_pdf["Type"], agg_pdf["hours"], color=colors)
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
        # Scale chart relative to page width
        chart_width = usable * 0.72
        chart_height = chart_width * 0.36  # keep aspect
        story.append(RLImage(img_reader, width=chart_width, height=chart_height))
        story.append(Spacer(1, 6))
        # Breakdown table: Type right, Hours left
        bd_headers = [Paragraph("Type", header_style), Paragraph("Hours", header_style)]
        bd_data = [bd_headers]
        for _, r in agg_pdf.iterrows():
            bd_data.append([Paragraph(str(r["Type"]), cell_style), Paragraph(f"{float(r["hours"]):.2f}", cell_style)])
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
    df = _df_for_pdf(payload["detail"])
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


# === Mini-dashboard helpers (hours-only) ===
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

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
    if total <= 0:
        total = 0.0
    agg["percent"] = np.where(total > 0, (agg["hours"]/total)*100.0, 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Hours", f"{total:,.2f}")
    top_type = agg.iloc[0]["Type"] if not agg.empty else "-"
    top_pct  = agg.iloc[0]["percent"] if not agg.empty else 0.0
    c2.metric("Top Type", f"{top_type}")
    c3.metric("% in Top Type", f"{top_pct:.1f}%")

    base = alt.Chart(agg).encode(
        x=alt.X("Type:N", sort="-y", title="Work Order Type"),
        tooltip=[alt.Tooltip("Type:N"),
                 alt.Tooltip("hours:Q", format=".2f")]
    )
    color_scale = alt.Scale(domain=list(_TYPE_COLORS.keys()),
                            range=list(_TYPE_COLORS.values()))

    st.caption("Hours by Work Order Type")
    st.altair_chart(
        base.mark_bar().encode(y=alt.Y("hours:Q", title="Hours"),
                               color=alt.Color("Type:N", scale=color_scale)),
        use_container_width=True
    )

    st.caption("Breakdown (Hours & % of Craft Total)")
    st.dataframe(agg[["Type", "hours"]], use_container_width=True, hide_index=True)

def _style_breakdown(df):
    try:
        return (df.style
                  .set_properties(subset=["Type"], **{"text-align":"right"})
                  .set_properties(subset=["hours"], **{"text-align":"left"}))
    except Exception:
        return df

# === End helpers ===

# === Table styling for Type colors ===
_TYPE_COLORS = {
    "Break In": "#d62728",
    "Maintenance Order": "#1f77b4",
    "Urgent Corrective": "#ff7f0e",
    "Emergency Order": "#d62728",
    "PM Restore/Replace": "#2ca02c",
    "PM Inspection": "#2ca02c",
    "Follow Up Maintenance Order": "#d4c720",
    "Project": "#9467bd",
}

def _style_types(df):
    if df is None or df.empty or "Type" not in df.columns:
        return df
    def _style_cell(v):
        color = _TYPE_COLORS.get(str(v), None)
        return f"background-color: {color}; color: white; font-weight: 600;" if color else ""
    try:
        return df.style.applymap(_style_cell, subset=["Type"])
    except Exception:
        # Fallback: return unstyled if Styler isn't supported in the running Streamlit version
        return df
# === End table styling ===



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

report = prepare_report_data(time_df, addr_df, craft_df, selected_date)

st.sidebar.subheader("Export")
st.sidebar.download_button(
    "Generate & Download PDF (Landscape)",
    data=build_pdf(report, selected_label),
    file_name=f"workorder_report_{selected_label.replace('/', '-')}.pdf",
    mime="application/pdf",
)

st.markdown(f"### Report for {selected_label}")

col_cfg = {
    "Name": st.column_config.TextColumn("Name", width=200),
    "Work Order #": st.column_config.TextColumn("Work Order #", width=90),
    "Sum of Hours": st.column_config.NumberColumn("Sum of Hours", format="%.2f", width=90),
    "Type": st.column_config.TextColumn("Type", width=200),
    "Description": st.column_config.TextColumn("Description", width=300),  # reduced to 300 px
    "Problem": st.column_config.TextColumn("Problem", width=420),
}

for craft_name, payload in report["groups"]:
    st.markdown(f"#### {craft_name}")
    df_detail = payload["detail"]
    _craft_dashboard_block(df_detail)
    st.dataframe(
        _style_types(df_detail),
        use_container_width=True,
        hide_index=True,
        height=_auto_height(df_detail),
        column_config=col_cfg
    )
    st.markdown("---")

if not report["full_detail"].empty:
    csv = report["full_detail"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered detail (CSV)",
        data=csv,
        file_name=f"workorder_detail_{selected_label.replace('/', '-')}.csv",
        mime="text/csv",
    )