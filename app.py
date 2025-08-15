import io
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st

# -------------------------------
# HARD-CODED REFERENCE DATA
# -------------------------------
CRAFT_ORDER = ['Turns', 'EAF Mech Days', 'EAF Elec Days', 'AOD Mech Days', 'AOD Elec Days', 'Alloy Mech Days', 'Caster Mech Days', 'Caster Elec Days', 'WTP Mech Days', 'Baghouse Mech Days', 'Preheater Elec Days', 'Segment Shop', 'Utilities Mech Days', 'HVAC Elec Days']
ADDRESS_BOOK = [{'AddressBookNumber': '1103079', 'Name': 'CONKEL, JOHNATHON J', 'Craft Description': 'Alloy Mech Days'}, {'AddressBookNumber': '817150', 'Name': 'PETERS, JESSE DANIEL', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '648991', 'Name': 'JONES, TERRELL D.', 'Craft Description': 'AOD Elec Days'}, {'AddressBookNumber': '136792', 'Name': 'MCKINNEY, CHRIS ALVIE', 'Craft Description': 'AOD Mech Days'}, {'AddressBookNumber': '1142730', 'Name': 'CHRISTERSON, NATHANIEL BENJAMEN', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1150094', 'Name': 'WRIGHT, KEVIN BRADLEY', 'Craft Description': 'Baghouse Mech Days'}, {'AddressBookNumber': '1064305', 'Name': 'DALTON II, JEFFERY WAYNE', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1115109', 'Name': 'GANDER, ANTHONY T', 'Craft Description': 'Caster Elec Days'}, {'AddressBookNumber': '1055943', 'Name': 'HEFFELMIRE, RONALD SCOTT', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '1112813', 'Name': 'KOONS, ANDREW LEWIS ALAN', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '95636', 'Name': 'MORRISON, GEORGE D.', 'Craft Description': 'Caster Mech Days'}, {'AddressBookNumber': '586013', 'Name': 'DENNIS, SHAWN MICHAEL', 'Craft Description': 'EAF Elec Days'}, {'AddressBookNumber': '1137121', 'Name': 'STEWART, THOMAS JASON', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '1106595', 'Name': 'WASH, MICHAEL DAVID', 'Craft Description': 'EAF Mech Days'}, {'AddressBookNumber': '178909', 'Name': 'LEMASTER, DANIEL M.', 'Craft Description': 'HVAC Elec Days'}, {'AddressBookNumber': '1115133', 'Name': 'BROCK, TREVOR COLE', 'Craft Description': 'Preheater Elec Days'}, {'AddressBookNumber': '133760', 'Name': 'BRIGHTWELL, JEFFERY W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '173665', 'Name': 'CRAIG, JAMES D.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '336719', 'Name': 'DEEN, ALAN J.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1151409', 'Name': 'DEMAREE, MATTHEW CHRISTOPHER', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '848802', 'Name': 'KLOSS, CHARLES W.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '95644', 'Name': 'SMITH, JAMES M.', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1104469', 'Name': 'WATSON, JACOB LEYTON', 'Craft Description': 'Segment Shop'}, {'AddressBookNumber': '1103976', 'Name': 'BAUGHMAN, THOMAS BRUCE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1066095', 'Name': 'HELTON, MICHAEL AJ', 'Craft Description': 'Turns'}, {'AddressBookNumber': '44231', 'Name': 'REED, BRIAN L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1167030', 'Name': 'STROUD, MATTHEW T.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '164380', 'Name': 'WARREN, MARK L.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '185050', 'Name': 'WHOBREY, BRADLEY G.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1103132', 'Name': 'WILLIAMS II, STEVEN FOSTER', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1106747', 'Name': 'BANTA, BENJAMIN GAYLE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1165384', 'Name': 'BOHART, WILLIAM M.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1144250', 'Name': 'CAREY, JOSEPH MICHAEL', 'Craft Description': 'Turns'}, {'AddressBookNumber': '770363', 'Name': 'CORMAN, DAVID H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1149608', 'Name': 'DIEDERICH, JOSEPH W', 'Craft Description': 'Turns'}, {'AddressBookNumber': '193471', 'Name': 'GRAY, DENNIS C.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '109866', 'Name': 'HOWARD, LARRY D.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1141761', 'Name': 'PHILLIPS, TIMOTHY CRAIG RYAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '272006', 'Name': 'SEE, JOHN JOURDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '106260', 'Name': 'SPILLMAN, WILLIAM H.', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1131299', 'Name': 'STEWART, BRADFORD LEE', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1109876', 'Name': 'STOKES, MATHEW DAVID', 'Craft Description': 'Turns'}, {'AddressBookNumber': '234448', 'Name': 'THOMAS, CODY JORDAN', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1107352', 'Name': 'WATKINS, KENNETH EDWARD', 'Craft Description': 'Turns'}, {'AddressBookNumber': '1096665', 'Name': 'ATWELL, TALON BRADLEY', 'Craft Description': 'Turns'}, {'AddressBookNumber': '108986', 'Name': 'ROGERS, CHARLES D.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '206092', 'Name': 'TURNER, SHANE M.', 'Craft Description': 'Utilities Mech Days'}, {'AddressBookNumber': '1089377', 'Name': 'ROSE, CAMERON CHASE', 'Craft Description': 'WTP Mech Days'}]

# Columns to display (and their order)
DISPLAY_COLUMNS = ["Name", "Work Order Number", "Sum of Hours", "Type", "Description", "Problem"]

# -------------------------------
# Helpers
# -------------------------------
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
    # Ensure required columns exist
    for c in REQUIRED_TIME_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # Normalize IDs & dates
    df["AddressBookNumber"] = df["AddressBookNumber"].astype(str).str.strip()
    if "Production Date" in df.columns:
        df["Production Date"] = pd.to_datetime(df["Production Date"], errors="coerce").dt.date
    # Normalize hours & work order number labels for downstream
    # Hours: accept 'Sum of Hours.', 'Sum of Hours', or 'Hours' and create a unified 'Sum of Hours'
    if "Sum of Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce")
    elif "Sum of Hours." in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours."], errors="coerce")
    elif "Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    else:
        df["Sum of Hours"] = pd.NA

    # Work order number: derive unified 'Work Order Number'
    if "Work Order Number" in df.columns:
        df["Work Order Number"] = df["Work Order Number"]
    elif "OrderNumber" in df.columns:
        df["Work Order Number"] = df["OrderNumber"]
    elif "WO Number" in df.columns:
        df["Work Order Number"] = df["WO Number"]
    elif "WorkOrderNumber" in df.columns:
        df["Work Order Number"] = df["WorkOrderNumber"]
    else:
        df["Work Order Number"] = pd.NA

    # Ensure Problem column exists
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

def prepare_report_data(time_df: pd.DataFrame,
                        addr_df: pd.DataFrame,
                        craft_order_df: pd.DataFrame,
                        selected_date) -> Dict[str, Any]:
    f = time_df[time_df["Production Date"] == selected_date].copy()

    # Merge craft from address book by AddressBookNumber
    f["AddressBookNumber"] = f["AddressBookNumber"].astype(str).str.strip()
    addr_df["AddressBookNumber"] = addr_df["AddressBookNumber"].astype(str).str.strip()
    merged = f.merge(
        addr_df[["AddressBookNumber", "Craft Description", "Name"]].rename(columns={"Name": "AB_Name"}),
        on="AddressBookNumber",
        how="left"
    )
    merged["Name"] = merged["Name"].fillna(merged["AB_Name"])
    merged = merged.drop(columns=["AB_Name"])

    # Handle unmapped
    unmapped = []
    mask_unmapped = merged["Craft Description"].isna() | (merged["Craft Description"].astype(str).str.len() == 0)
    if mask_unmapped.any():
        unmapped = merged.loc[mask_unmapped, ["AddressBookNumber", "Name"]].drop_duplicates().to_dict("records")
        merged.loc[mask_unmapped, "Craft Description"] = "Unassigned"

    merged = _apply_craft_category(merged, craft_order_df)

    # Build display detail with only requested columns
    # Make sure all display columns exist (create empty if missing)
    for col in DISPLAY_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    # Sort detail by Name then Work Order Number
    merged = merged.sort_values(["Craft Description", "Name", "Work Order Number"])

    # Per-person summary (sum of hours & unique work orders)
    person_summary = (
        merged.groupby(["Craft Description", "Name", "AddressBookNumber"], dropna=False)
              .agg(**{"Sum of Hours": ("Sum of Hours", "sum"),
                      "Work Orders": ("Work Order Number", "nunique")})
              .reset_index()
    )
    # Split into per-craft payloads
    groups_payload: List = []
    for craft in list(merged["Craft Description"].cat.categories):
        g_detail = merged[merged["Craft Description"] == craft][DISPLAY_COLUMNS].copy()
        if g_detail.empty:
            continue
        g_summary = person_summary[person_summary["Craft Description"] == craft][["Name", "AddressBookNumber", "Sum of Hours", "Work Orders"]].copy()
        g_summary = g_summary.sort_values(["Sum of Hours", "Name"], ascending=[False, True])
        groups_payload.append((str(craft), {"person_summary": g_summary, "detail": g_detail}))

    full_detail = merged[DISPLAY_COLUMNS].copy()

    return {"groups": groups_payload, "full_detail": full_detail, "unmapped_people": unmapped}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Work Order Daily Report", layout="wide")
st.title("Work Order Reporting App — Flat, Hard-Coded Reference")
st.caption("Upload the Time on Work Order export. Report only shows: Name, Work Order Number, Sum of Hours, Type, Description, Problem.")

with st.sidebar:
    st.header("1) Upload file")
    time_file = st.file_uploader("Time on Work Order (.xlsx) – REQUIRED", type=["xlsx"], key="time")
    st.markdown("---")
    st.header("2) Options")
    show_detail = st.checkbox("Show detailed rows", value=True)
    show_person_summary = st.checkbox("Show per-person summary", value=True)

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

if report["unmapped_people"]:
    with st.expander("⚠️ Unmapped personnel (not found in Address Book)", expanded=False):
        st.dataframe(pd.DataFrame(report["unmapped_people"], columns=["AddressBookNumber", "Name"]).sort_values("Name"))

for craft_name, payload in report["groups"]:
    st.markdown(f"#### {craft_name}")
    if show_person_summary and not payload["person_summary"].empty:
        st.subheader("Per-Person Summary (Sum of Hours & WO Count)")
        st.dataframe(payload["person_summary"], use_container_width=True)
    if show_detail and not payload["detail"].empty:
        st.subheader("Detail (Selected Columns)")
        st.dataframe(payload["detail"], use_container_width=True)
    st.markdown("---")

if not report["full_detail"].empty:
    csv = report["full_detail"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered detail (CSV - selected columns)",
        data=csv,
        file_name=f"workorder_detail_{selected_label.replace('/', '-')}.csv",
        mime="text/csv",
    )