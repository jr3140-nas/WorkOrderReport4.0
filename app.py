import io
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# PDF
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---- Excel loader ----
EXPECTED_HEADERS = [
    "AddressBookNumber","Name","Production Date","OrderNumber","Work Order #","Sum of Hours","Sum of Hours.",
    "Hours Estimated","Status","Type","PMFrequency","PM Frequency","Description","Problem",
    "Department","Location","Equipment","PM Number","PM"
]

RENAME_MAP = {
    "OrderNumber": "Work Order #",
    "Order Number": "Work Order #",
    "Sum of Hours.": "Sum of Hours",
    "PM Frequency": "PMFrequency"
}

def _find_header_row(df: pd.DataFrame) -> int | None:
    for i in range(min(30, len(df))):
        row_vals = df.iloc[i].astype(str).str.strip()
        matches = sum(1 for v in row_vals if v in EXPECTED_HEADERS)
        if matches >= 3:
            return i
    return None

def _normalize_columns(cols) -> list[str]:
    normalized = []
    for c in cols:
        c2 = str(c).strip()
        c2 = RENAME_MAP.get(c2, c2)
        normalized.append(c2)
    return normalized

def load_timeworkbook(uploaded_file) -> pd.DataFrame:
    # Read all sheets and return the first one that contains the expected headers
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object, engine="openpyxl")
        hdr = _find_header_row(raw)
        if hdr is None:
            continue
        df = raw.iloc[hdr+1:].copy()
        df.columns = _normalize_columns(raw.iloc[hdr].tolist())
        # Keep only expected columns if present
        # Ensure required fields exist
        for need in ["Name","Work Order #","Sum of Hours","Type","Description","Problem"]:
            if need not in df.columns:
                df[need] = pd.NA
        # Coerce types
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
        # Clean work order display
        if df["Work Order #"].dtype != object:
            df["Work Order #"] = df["Work Order #"].astype(str)
        df["Work Order #"] = df["Work Order #"].astype(str).str.replace(r"\.0$", "", regex=True)
        # Map numeric/letter types through TYPE_MAP if present
        try:
            # TYPE_MAP may be defined elsewhere in file; fall back if not
            tm = TYPE_MAP
            df["Type"] = df["Type"].map(lambda x: tm.get(str(x), str(x)) if pd.notna(x) else x)
        except Exception:
            pass
        return df.reset_index(drop=True)
    raise ValueError("Could not locate header row in any sheet. Please upload the standard 'Time on Work Order' export.")


# ---- Hard-coded mappings ----
CRAFT_ORDER = ['Turns', 'EAF Mech Days', 'EAF Elec Days', 'AOD Mech Days', 'AOD Elec Days', 'Alloy Mech Days', 'Caster Mech Days', 'Caster Elec Days', 'WTP Mech Days', 'Baghouse Mech Days', 'Preheater Elec Days', 'Segment Shop', 'Utilities Mech Days', 'HVAC Elec Days']
ADDRESS_BOOK = [{'AddressBookNumber': '1103079', 'Name': 'CONKEL, JOHNATHON J', 'Craft Description': 'Alloy Mech Days'}, {'AddressBookNumber': '817150', 'Name': 'PETERS, JESSE DANIEL', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '648991', 'Name': 'JONES, TERRELL D.', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '136792', 'Name': 'MCKINNEY, CHRIS ALVIE', 'Craft Description': 'AOD Mech Days'}, {'AddressBookNumber': '1142730', 'Name': 'CHRISTERSON, NATHANIEL BENJAMEN', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1150094', 'Name': 'WRIGHT, KEVIN BRADLEY', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1064305', 'Name': 'DALTON II, JEFFERY WAYNE', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1115109', 'Name': 'GANDER, ANTHONY T', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1055943', 'Name': 'HEFFELMIRE, RONALD SCOTT', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '1112813', 'Name': 'KOONS, ANDREW LEWIS ALAN', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '95636', 'Name': 'MORRISON, GEORGE D.', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '586013', 'Name': 'DENNIS, SHAWN MICHAEL', 'Craft Description': 'EAF Elec Days'}, {'AddressBookNumber': '1137121', 'Name': 'STEWART, THOMAS JASON', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '1106595', 'Name': 'WASH, MICHAEL DAVID', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '178909', 'Name': 'LEMASTER, DANIEL M.', 'Craft Description': 'HVAC Elec Days'}, {'AddressBookNumber': '1115133', 'Name': 'BROCK, TREVOR COLE', 'Craft Description': 'Preheater Elec Days'}, {'AddressBookNumber': '133760', 'Name': 'BRIGHTWELL, JEFFERY W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '173665', 'Name': 'CRAIG, JAMES D.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '336719', 'Name': 'DEEN, ALAN J.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1151409', 'Name': 'DEMAREE, MATTHEW CHRISTOPHER', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '848802', 'Name': 'KLOSS, CHARLES W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '95644', 'Name': 'SMITH, JAMES M.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1104469', 'Name': 'WATSON, JACOB LEYTON', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1103976', 'Name': 'BAUGHMAN, THOMAS BRUCE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1066095', 'Name': 'HELTON, MICHAEL AJ', 'Craft Description': 'Turns'}, {'AddressBookNumber': '44231', 'Name': 'REED, BRIAN L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1167030', 'Name': 'STROUD, MATTHEW T.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '164380', 'Name': 'WARREN, MARK L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '185050', 'Name': 'WHOBREY, BRADLEY G.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1103132', 'Name': 'WILLIAMS II, STEVEN FOSTER', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1106747', 'Name': 'BANTA, BENJAMIN GAYLE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1165384', 'Name': 'BOHART, WILLIAM M.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1144250', 'Name': 'CAREY, JOSEPH MICHAEL', 'Craft Description': 'Turns'}, {'AddressBookNumber': '770363', 'Name': 'CORMAN, DAVID H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1149608', 'Name': 'DIEDERICH, JOSEPH W', 'Craft Description': 'Turns'}, {'AddressBookNumber': '193471', 'Name': 'GRAY, DENNIS C.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '109866', 'Name': 'HOWARD, LARRY D.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1141761', 'Name': 'PHILLIPS, TIMOTHY CRAIG RYAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '272006', 'Name': 'SEE, JOHN JOURDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '106260', 'Name': 'SPILLMAN, WILLIAM H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1131299', 'Name': 'STEWART, BRADFORD LEE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1109876', 'Name': 'STOKES, MATHEW DAVID', 'Craft Description': 'Turns'}, {'AddressBookNumber': '234448', 'Name': 'THOMAS, CODY JORDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1107352', 'Name': 'WATKINS, KENNETH EDWARD', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1096665', 'Name': 'ATWELL, TALON BRADLEY', 'Craft Description': 'Turns'}, {'AddressBookNumber': '108986', 'Name': 'ROGERS, CHARLES D.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '206092', 'Name': 'TURNER, SHANE M.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '1089377', 'Name': 'ROSE, CAMERON CHASE', 'Craft Description': 'WTP Mech Days'}]
TYPE_MAP = {'0': 'Break In', '1': 'Maintenance Order', '2': 'Material Repair TMJ Order', '3': 'Capital Project', '4': 'Urgent Corrective', '5': 'Emergency Order', '6': 'PM Restore/Replace', '7': 'PM Inspection', '8': 'Follow Up Maintenance Order', '9': 'Standing W.O. - Do not Delete', 'B': 'Marketing', 'C': 'Cost Improvement', 'D': 'Design Work - ETO', 'E': 'Plant Work - ETO', 'G': 'Governmental/Regulatory', 'M': 'Model W.O. - Eq Mgmt', 'N': 'Template W.O. - CBM Alerts', 'P': 'Project', 'R': 'Rework Order', 'S': 'Shop Order', 'T': 'Tool Order', 'W': 'Case', 'X': 'General Work Request', 'Y': 'Follow Up Work Request', 'Z': 'System Work Request'}
TYPE_COLOR_HEX = {}
DEFAULT_SLICE = "#d9d9d9"


# ---- Helpers to expose hard-coded craft + address book as DataFrames ----
def get_craft_order_df() -> pd.DataFrame:
    try:
        return pd.DataFrame({"Craft Description": CRAFT_ORDER})
    except Exception:
        return pd.DataFrame({"Craft Description": []})

def get_address_book_df() -> pd.DataFrame:
    try:
        df = pd.DataFrame(ADDRESS_BOOK)
        cols = [c for c in ["AddressBookNumber", "Name", "Craft Description"] if c in df.columns]
        return df[cols].copy() if cols else df.copy()
    except Exception:
        return pd.DataFrame(columns=["AddressBookNumber", "Name", "Craft Description"])

# ==== Color configuration for pie charts ====
# Colorblind-safe palette (Okabe–Ito with extras)
_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00",
    "#CC79A7", "#F0E442", "#000000", "#8DD3C7", "#80B1D3",
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#A6CEE3", "#FF7F00",
    "#33A02C", "#E31A1C", "#B15928", "#6A3D9A"
]

def _populate_type_colors():
    """Populate TYPE_COLOR_HEX from TYPE_MAP when empty, ensuring stable colors."""
    global TYPE_COLOR_HEX
    if not isinstance(TYPE_COLOR_HEX, dict):
        TYPE_COLOR_HEX = {}
    if len(TYPE_COLOR_HEX) == 0:
        try:
            type_names = list(dict.fromkeys(list(TYPE_MAP.values())))  # preserve insertion order
        except Exception:
            type_names = []
        for i, name in enumerate(type_names):
            if name and name not in TYPE_COLOR_HEX:
                TYPE_COLOR_HEX[name] = _PALETTE[i % len(_PALETTE)]
        TYPE_COLOR_HEX.setdefault("Unspecified", DEFAULT_SLICE)

# Populate on import
_populate_type_colors()

# Explicit overrides (plant standard)
TYPE_COLOR_HEX.update({
    "PM Inspection": "#009E73",   # green
    "Emergency Order": "#E31A1C", # red
})
# ============================================

# Populate on import
_populate_type_colors()

# ---- Explicit color overrides for specific Types ----
TYPE_COLOR_HEX.update({
    "PM Inspection": "#009E73",   # green
    "Emergency Order": "#E31A1C", # red
})


def _auto_height(df: pd.DataFrame) -> int:
    rows = len(df) + 1
    row_px = 35
    header_px = 40
    return min(header_px + rows * row_px, 20000)

def _type_color(name: str) -> str:
    # normalize for lookup
    if name is None:
        return DEFAULT_SLICE
    key = str(name).strip()
    return TYPE_COLOR_HEX.get(key, DEFAULT_SLICE)



def _pie_for_group(df_detail: pd.DataFrame, title: str):
    s = df_detail.groupby("Type", dropna=False)["Sum of Hours"].sum().sort_values(ascending=False)
    if s.empty or s.fillna(0).sum() <= 0:
        return None

    labels = [("Unspecified" if (pd.isna(k) or str(k).strip()=="") else str(k).strip()) for k in s.index.tolist()]
    values = s.values.tolist()
    colors = [_type_color(lbl) for lbl in labels]

    total = float(sum(values))
    if total <= 0:
        return None

    # Larger canvas for less crowding
    fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=120)
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=1.0,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"}
    )

    import math
    label_fs = 8
    for w, lbl, val in zip(wedges, labels, values):
        if val <= 0:
            continue
        ang = (w.theta2 + w.theta1) / 2.0
        ang_rad = math.radians(ang)
        x = math.cos(ang_rad)
        y = math.sin(ang_rad)
        r_text = 1.25
        tx = r_text * x
        ty = r_text * y
        ha = "left" if tx >= 0 else "right"
        text = f"{lbl} — {val:.2f}h"
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(tx, ty),
            ha=ha, va="center",
            fontsize=label_fs,
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", linewidth=0.6)
        )

    ax.axis("equal")
    ax.set_title(title, fontsize=12, pad=6)
    plt.tight_layout()
    return fig



# ---------- PDF builder (unchanged pie-in-PDF omitted for now) ----------
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

    px_widths = [200, 90, 90, 200, 300, 420]
    page_width, _ = landscape(letter)
    usable = page_width - doc.leftMargin - doc.rightMargin
    scale = usable / sum(px_widths)
    col_widths = [w * scale for w in px_widths]

    header_bg = colors.HexColor("#f0f0f0"); grid_color = colors.HexColor("#d0d0d0")

    for craft_name, payload in report["groups"]:
        story.append(Paragraph(f"<b>{craft_name}</b>", styles["Heading2"]))
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

# Build report
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
    "Description": st.column_config.TextColumn("Description", width=300),
    "Problem": st.column_config.TextColumn("Problem", width=420),
}

for craft_name, payload in report["groups"]:
    st.markdown(f"#### {craft_name}")
    df_detail = payload["detail"]
    # Pie chart of hours by Type for this craft (smaller and colored)
    fig = _pie_for_group(df_detail, "Hours by Type")
    if fig is not None:
        st.pyplot(fig, use_container_width=False)
    # Table
    st.dataframe(
        df_detail,
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

# ---- Helpers to expose hard-coded craft + address book as DataFrames ----
def get_craft_order_df() -> pd.DataFrame:
    try:
        return pd.DataFrame({"Craft Description": CRAFT_ORDER})
    except Exception:
        # Fallback: empty DF with expected column
        return pd.DataFrame({"Craft Description": []})

def get_address_book_df() -> pd.DataFrame:
    try:
        df = pd.DataFrame(ADDRESS_BOOK)
        # Keep only the columns used downstream
        cols = [c for c in ["AddressBookNumber", "Name", "Craft Description"] if c in df.columns]
        if cols:
            return df[cols].copy()
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["AddressBookNumber", "Name", "Craft Description"])
