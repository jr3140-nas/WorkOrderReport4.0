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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- PDF builder with wrapping & fit-to-width ----------
def _df_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Sum of Hours" in out.columns:
        out["Sum of Hours"] = pd.to_numeric(out["Sum of Hours"], errors="coerce").fillna(0).map(lambda x: f"{x:.2f}")
    return out


def build_pdf(report: Dict[str, Any], date_label: str) -> bytes:
    """
    Generate a landscape PDF that includes:
      - Section per craft with the detail table (DISPLAY_COLUMNS)
      - A bar chart of "Hours by Type" for that craft
      - A small breakdown table: Type | Hours | % of Craft
    """
    buffer = io.BytesIO()
    # Create document (A4 letter landscape)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=24
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        spaceBefore=6,
        spaceAfter=8
    )
    header_style = ParagraphStyle(
        "header",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        spaceBefore=0,
        spaceAfter=0
    )
    cell_style = ParagraphStyle(
        "cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        wordWrap="CJK",
        spaceBefore=0,
        spaceAfter=0
    )

    grid_color = colors.Color(0.75, 0.75, 0.75)
    header_bg = colors.HexColor("#e9eef3")

    story = []
    # Document header
    story.append(Paragraph(f"Work Order Report — {date_label}", title_style))
    story.append(Spacer(1, 8))

    # Iterate crafts
    for craft_name, payload in report.get("groups", []):
        df_detail = payload.get("detail")
        if df_detail is None or df_detail.empty:
            continue

        # Section heading
        story.append(Paragraph(f"{craft_name}", styles["Heading3"]))
        story.append(Spacer(1, 4))

        # Build the main table for this craft (DISPLAY_COLUMNS expected in df)
        df = _df_for_pdf(df_detail[DISPLAY_COLUMNS].copy() if set(DISPLAY_COLUMNS).issubset(df_detail.columns) else df_detail.copy())
        data = [ [Paragraph(col, header_style) for col in df.columns] ]
        for _, row in df.iterrows():
            data.append([Paragraph(str(row[c]) if pd.notna(row[c]) else "", cell_style) for c in df.columns])

        tbl = Table(data, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), header_bg),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("GRID", (0,0), (-1,-1), 0.25, grid_color),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ("ALIGN", (2,1), (2,-1), "RIGHT"),  # Sum of Hours typically in col 2 after Name, Work Order #
        ]))
        story.append(tbl)

        # Add chart and breakdown for this craft
        try:
            df_num = df_detail.copy()
            # Ensure numeric
            if "Sum of Hours" in df_num.columns:
                df_num["Sum of Hours"] = pd.to_numeric(df_num["Sum of Hours"], errors="coerce").fillna(0.0)
            else:
                # If column is "Sum of Hours." or similar, try to normalize
                if "Sum of Hours." in df_num.columns:
                    df_num["Sum of Hours"] = pd.to_numeric(df_num["Sum of Hours."], errors="coerce").fillna(0.0)
                else:
                    # No hours column—skip chart cleanly
                    raise ValueError("Missing 'Sum of Hours' column")

            # Group by Type
            agg = (df_num.groupby("Type", dropna=False)["Sum of Hours"]
                        .sum()
                        .reset_index()
                        .rename(columns={"Sum of Hours": "hours"})
                        .sort_values("hours", ascending=False))
            total = float(agg["hours"].sum()) if not agg.empty else 0.0
            agg["percent"] = (agg["hours"]/total*100.0).where(total > 0, 0.0)

            # Render bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=150)
            ax.bar(agg["Type"].astype(str), agg["hours"])
            ax.set_xlabel("Work Order Type")
            ax.set_ylabel("Hours")
            ax.set_title(f"Hours by Type — {craft_name}")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            fig.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            plt.close(fig)
            img_buf.seek(0)

            # Insert chart image
            story.append(Spacer(1, 6))
            story.append(RLImage(img_buf, width=520, height=195))
            story.append(Spacer(1, 6))

            # Breakdown table (Type, Hours, %)
            bd_data = [
                [Paragraph("<b>Type</b>", header_style),
                 Paragraph("<b>Hours</b>", header_style),
                 Paragraph("<b>% of Craft</b>", header_style)]
            ]
            for _, r in agg.iterrows():
                bd_data.append([
                    Paragraph(str(r["Type"]), cell_style),
                    Paragraph(f"{float(r['hours']):.2f}", cell_style),
                    Paragraph(f"{float(r['percent']):.1f}%", cell_style),
                ])
            bd_tbl = Table(bd_data, repeatRows=1, hAlign="LEFT")
            bd_tbl.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f3f6")),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
                ("RIGHTPADDING", (0,0), (-1,-1), 4),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ]))
            story.append(bd_tbl)
            story.append(Spacer(1, 10))

        except Exception as e:
            story.append(Paragraph(f"[Chart generation skipped: {e}]", styles["Normal"]))
            story.append(Spacer(1, 8))

    # Finalize document
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


if ("report" in locals()) and isinstance(report, dict) \
   and ("full_detail" in report) and hasattr(report["full_detail"], "empty") \
   and (not report["full_detail"].empty) and ("selected_label" in locals()):
    csv = report["full_detail"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered detail (CSV)",
        data=csv,
        file_name=f"workorder_detail_{selected_label.replace('/', '-')}.csv",
        mime="text/csv",
    )