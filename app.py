
import io, html
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

CRAFT_ORDER = ['Turns', 'EAF Mech Days', 'EAF Elec Days', 'AOD Mech Days', 'AOD Elec Days', 'Alloy Mech Days', 'Caster Mech Days', 'Caster Elec Days', 'WTP Mech Days', 'Baghouse Mech Days', 'Preheater Elec Days', 'Segment Shop', 'Utilities Mech Days', 'HVAC Elec Days']
ADDRESS_BOOK = [{'AddressBookNumber': '1103079', 'Name': 'CONKEL, JOHNATHON J', 'Craft Description': 'Alloy Mech Days'}, {'AddressBookNumber': '817150', 'Name': 'PETERS, JESSE DANIEL', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '648991', 'Name': 'JONES, TERRELL D.', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '136792', 'Name': 'MCKINNEY, CHRIS ALVIE', 'Craft Description': 'AOD Mech Days'}, {'AddressBookNumber': '1142730', 'Name': 'CHRISTERSON, NATHANIEL BENJAMEN', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1150094', 'Name': 'WRIGHT, KEVIN BRADLEY', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1064305', 'Name': 'DALTON II, JEFFERY WAYNE', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1115109', 'Name': 'GANDER, ANTHONY T', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1055943', 'Name': 'HEFFELMIRE, RONALD SCOTT', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '1112813', 'Name': 'KOONS, ANDREW LEWIS ALAN', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '95636', 'Name': 'MORRISON, GEORGE D.', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '586013', 'Name': 'DENNIS, SHAWN MICHAEL', 'Craft Description': 'EAF Elec Days'}, {'AddressBookNumber': '1137121', 'Name': 'STEWART, THOMAS JASON', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '1106595', 'Name': 'WASH, MICHAEL DAVID', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '178909', 'Name': 'LEMASTER, DANIEL M.', 'Craft Description': 'HVAC Elec Days'}, {'AddressBookNumber': '1115133', 'Name': 'BROCK, TREVOR COLE', 'Craft Description': 'Preheater Elec Days'}, {'AddressBookNumber': '133760', 'Name': 'BRIGHTWELL, JEFFERY W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '173665', 'Name': 'CRAIG, JAMES D.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '336719', 'Name': 'DEEN, ALAN J.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1151409', 'Name': 'DEMAREE, MATTHEW CHRISTOPHER', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '848802', 'Name': 'KLOSS, CHARLES W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '95644', 'Name': 'SMITH, JAMES M.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1104469', 'Name': 'WATSON, JACOB LEYTON', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1103976', 'Name': 'BAUGHMAN, THOMAS BRUCE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1066095', 'Name': 'HELTON, MICHAEL AJ', 'Craft Description': 'Turns'}, {'AddressBookNumber': '44231', 'Name': 'REED, BRIAN L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1167030', 'Name': 'STROUD, MATTHEW T.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '164380', 'Name': 'WARREN, MARK L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '185050', 'Name': 'WHOBREY, BRADLEY G.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1103132', 'Name': 'WILLIAMS II, STEVEN FOSTER', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1106747', 'Name': 'BANTA, BENJAMIN GAYLE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1165384', 'Name': 'BOHART, WILLIAM M.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1144250', 'Name': 'CAREY, JOSEPH MICHAEL', 'Craft Description': 'Turns'}, {'AddressBookNumber': '770363', 'Name': 'CORMAN, DAVID H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1149608', 'Name': 'DIEDERICH, JOSEPH W', 'Craft Description': 'Turns'}, {'AddressBookNumber': '193471', 'Name': 'GRAY, DENNIS C.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '109866', 'Name': 'HOWARD, LARRY D.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1141761', 'Name': 'PHILLIPS, TIMOTHY CRAIG RYAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '272006', 'Name': 'SEE, JOHN JOURDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '106260', 'Name': 'SPILLMAN, WILLIAM H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1131299', 'Name': 'STEWART, BRADFORD LEE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1109876', 'Name': 'STOKES, MATHEW DAVID', 'Craft Description': 'Turns'}, {'AddressBookNumber': '234448', 'Name': 'THOMAS, CODY JORDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1107352', 'Name': 'WATKINS, KENNETH EDWARD', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1096665', 'Name': 'ATWELL, TALON BRADLEY', 'Craft Description': 'Turns'}, {'AddressBookNumber': '108986', 'Name': 'ROGERS, CHARLES D.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '206092', 'Name': 'TURNER, SHANE M.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '1089377', 'Name': 'ROSE, CAMERON CHASE', 'Craft Description': 'WTP Mech Days'}]
TYPE_MAP = {'0': 'Break In', '1': 'Maintenance Order', '2': 'Material Repair TMJ Order', '3': 'Capital Project', '4': 'Urgent Corrective', '5': 'Emergency Order', '6': 'PM Restore/Replace', '7': 'PM Inspection', '8': 'Follow Up Maintenance Order', '9': 'Standing W.O. - Do not Delete', 'B': 'Marketing', 'C': 'Cost Improvement', 'D': 'Design Work - ETO', 'E': 'Plant Work - ETO', 'G': 'Governmental/Regulatory', 'M': 'Model W.O. - Eq Mgmt', 'N': 'Template W.O. - CBM Alerts', 'P': 'Project', 'R': 'Rework Order', 'S': 'Shop Order', 'T': 'Tool Order', 'W': 'Case', 'X': 'General Work Request', 'Y': 'Follow Up Work Request', 'Z': 'System Work Request'}
TYPE_COLOR_MAP = {}

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

    if "Sum of Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce")
    elif "Sum of Hours." in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours."], errors="coerce")
    elif "Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    else:
        df["Sum of Hours"] = pd.NA

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
        result = pd.NA
    else:
        result = TYPE_MAP.get(key, key)
    if isinstance(result, str) and result.strip().lower() == "inspection maintenance order":
        return "PM Inspection"
    return result

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

COL_WIDTHS = {"Name":200, "Work Order #":90, "Sum of Hours":90, "Type":200, "Description":420, "Problem":420}

def _cell_html(col, val):
    text = "" if pd.isna(val) else str(val)
    style = ""
    if col == "Type" and text in TYPE_COLOR_MAP:
        style = f"background-color: {TYPE_COLOR_MAP[text]}; color: #1f1f1f;"
    if col in ["Description", "Problem"]:
        wrap_style = "white-space: normal;"
    else:
        wrap_style = "white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
    return f"<td style='padding:6px 10px;border-bottom:1px solid #eee;{wrap_style}{style}'>{html.escape(text)}</td>"

def dataframe_to_html_table(df: pd.DataFrame) -> str:
    ths = "".join([f"<th style='text-align:left;padding:8px 10px;border-bottom:1px solid #ddd;width:{COL_WIDTHS.get(c,160)}px;'>{html.escape(c)}</th>" for c in df.columns])
    rows = []
    for _, r in df.iterrows():
        tds = "".join([_cell_html(c, r[c]) for c in df.columns])
        rows.append(f"<tr>{tds}</tr>")
    tbody = "".join(rows)
    style = (
        "<style>"
        ".wo-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 0.95rem; }"
        ".wo-table tr:nth-child(even) td { background-color: #fafafa; }"
        "</style>"
    )
    return style + f"<table class='wo-table'><thead><tr>{ths}</tr></thead><tbody>{tbody}</tbody></table>"

st.set_page_config(page_title="Work Order Reporting App", layout="wide")
st.title("Work Order Reporting App")

with st.sidebar:
    st.header("Upload file")
    time_file = st.file_uploader("Time on Work Order (.xlsx) – REQUIRED", type=["xlsx"], key="time")

if not time_file:
    st.info("⬆️ Upload the **Time on Work Order** export to proceed.")
    st.stop()

try:
    time_df = load_timeworkbook(time_file)
    craft_df = get_craft_order_df()
    addr_df = get_address_book_df()
except Exception as e:
    st.error(f"File load error: {e}")
    st.stop()

if "Production Date" not in time_df.columns or time_df["Production Date"].dropna().empty:
    st.error("No valid 'Production Date' values found in the Time on Work Order file.")
    st.stop()

dates = sorted(pd.to_datetime(time_df["Production Date"]).dt.date.unique())
date_labels = [datetime.strftime(pd.to_datetime(d), "%m/%d/%Y") for d in dates]
label_to_date = dict(zip(date_labels, dates))
selected_label = st.selectbox("Select Production Date", options=date_labels, index=len(date_labels)-1)
selected_date = label_to_date[selected_label]

report = prepare_report_data(time_df, addr_df, craft_df, selected_date)
st.markdown(f"### Report for {selected_label}")

for craft_name, payload in report["groups"]:
    st.markdown(f"#### {craft_name}")
    df_detail = payload["detail"]
    html_table = dataframe_to_html_table(df_detail)
    components.html(html_table, height=_auto_height(df_detail) + 60, scrolling=False)
    st.markdown("---")

if not report["full_detail"].empty:
    csv = report["full_detail"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered detail (CSV)",
        data=csv,
        file_name=f"workorder_detail_{selected_label.replace('/', '-')}.csv",
        mime="text/csv",
    )
