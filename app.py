import io
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st

# Use matplotlib for PDF creation and charts.  Matplotlib is widely available and
# allows us to render charts and multiple pages in a single PDF without relying
# on the reportlab dependency.  We import the PdfPages backend to append each
# figure as a page in the report.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- Hard-coded mappings (from your uploaded files at build time) ----
CRAFT_ORDER = [
    "Turns",
    "EAF Mech Days",
    "EAF Elec Days",
    "AOD Mech Days",
    "AOD Elec Days",
    "Alloy Mech Days",
    "Caster Mech Days",
    "Caster Elec Days",
    "WTP Mech Days",
    "Baghouse Mech Days",
    "Preheater Elec Days",
    "Segment Shop",
    "Utilities Mech Days",
    "HVAC Elec Days",
]

# Address book mapping of employee numbers to name and craft description.
# This data is embedded so the application works without external dependencies.
ADDRESS_BOOK = [
    {"AddressBookNumber": "1103079", "Name": "CONKEL, JOHNATHON J", "Craft Description": "Alloy Mech Days"},
    {"AddressBookNumber": "817150", "Name": "PETERS, JESSE DANIEL", "Craft Description": "AOD Elec Days"},
    {"AddressBookNumber": "648991", "Name": "JONES, TERRELL D.", "Craft Description": "AOD Elec Days"},
    {"AddressBookNumber": "136792", "Name": "MCKINNEY, CHRIS ALVIE", "Craft Description": "AOD Mech Days"},
    {"AddressBookNumber": "1142730", "Name": "CHRISTERSON, NATHANIEL BENJAMEN", "Craft Description": "Baghouse Mech Days"},
    {"AddressBookNumber": "1150094", "Name": "WRIGHT, KEVIN BRADLEY", "Craft Description": "Baghouse Mech Days"},
    {"AddressBookNumber": "1064305", "Name": "DALTON II, JEFFERY WAYNE", "Craft Description": "Caster Elec Days"},
    {"AddressBookNumber": "1115109", "Name": "GANDER, ANTHONY T", "Craft Description": "Caster Elec Days"},
    {"AddressBookNumber": "1055943", "Name": "HEFFELMIRE, RONALD SCOTT", "Craft Description": "Caster Mech Days"},
    {"AddressBookNumber": "1112813", "Name": "KOONS, ANDREW LEWIS ALAN", "Craft Description": "Caster Mech Days"},
    {"AddressBookNumber": "95636", "Name": "MORRISON, GEORGE D.", "Craft Description": "Caster Mech Days"},
    {"AddressBookNumber": "586013", "Name": "DENNIS, SHAWN MICHAEL", "Craft Description": "EAF Elec Days"},
    {"AddressBookNumber": "1137121", "Name": "STEWART, THOMAS JASON", "Craft Description": "EAF Mech Days"},
    {"AddressBookNumber": "1106595", "Name": "WASH, MICHAEL DAVID", "Craft Description": "EAF Mech Days"},
    {"AddressBookNumber": "178909", "Name": "LEMASTER, DANIEL M.", "Craft Description": "HVAC Elec Days"},
    {"AddressBookNumber": "1115133", "Name": "BROCK, TREVOR COLE", "Craft Description": "Preheater Elec Days"},
    {"AddressBookNumber": "133760", "Name": "BRIGHTWELL, JEFFERY W.", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "173665", "Name": "CRAIG, JAMES D.", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "336719", "Name": "DEEN, ALAN J.", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "1151409", "Name": "DEMAREE, MATTHEW CHRISTOPHER", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "848802", "Name": "KLOSS, CHARLES W.", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "95644", "Name": "SMITH, JAMES M.", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "1104469", "Name": "WATSON, JACOB LEYTON", "Craft Description": "Segment Shop"},
    {"AddressBookNumber": "1103976", "Name": "BAUGHMAN, THOMAS BRUCE", "Craft Description": "Turns"},
    {"AddressBookNumber": "1066095", "Name": "HELTON, MICHAEL AJ", "Craft Description": "Turns"},
    {"AddressBookNumber": "44231", "Name": "REED, BRIAN L.", "Craft Description": "Turns"},
    {"AddressBookNumber": "1167030", "Name": "STROUD, MATTHEW T.", "Craft Description": "Turns"},
    {"AddressBookNumber": "164380", "Name": "WARREN, MARK L.", "Craft Description": "Turns"},
    {"AddressBookNumber": "185050", "Name": "WHOBREY, BRADLEY G.", "Craft Description": "Turns"},
    {"AddressBookNumber": "1103132", "Name": "WILLIAMS II, STEVEN FOSTER", "Craft Description": "Turns"},
    {"AddressBookNumber": "1106747", "Name": "BANTA, BENJAMIN GAYLE", "Craft Description": "Turns"},
    {"AddressBookNumber": "1165384", "Name": "BOHART, WILLIAM M.", "Craft Description": "Turns"},
    {"AddressBookNumber": "1144250", "Name": "CAREY, JOSEPH MICHAEL", "Craft Description": "Turns"},
    {"AddressBookNumber": "770363", "Name": "CORMAN, DAVID H.", "Craft Description": "Turns"},
    {"AddressBookNumber": "1149608", "Name": "DIEDERICH, JOSEPH W", "Craft Description": "Turns"},
    {"AddressBookNumber": "193471", "Name": "GRAY, DENNIS C.", "Craft Description": "Turns"},
    {"AddressBookNumber": "109866", "Name": "HOWARD, LARRY D.", "Craft Description": "Turns"},
    {"AddressBookNumber": "1141761", "Name": "PHILLIPS, TIMOTHY CRAIG RYAN", "Craft Description": "Turns"},
    {"AddressBookNumber": "272006", "Name": "SEE, JOHN JOURDAN", "Craft Description": "Turns"},
    {"AddressBookNumber": "106260", "Name": "SPILLMAN, WILLIAM H.", "Craft Description": "Turns"},
    {"AddressBookNumber": "1131299", "Name": "STEWART, BRADFORD LEE", "Craft Description": "Turns"},
    {"AddressBookNumber": "1109876", "Name": "STOKES, MATHEW DAVID", "Craft Description": "Turns"},
    {"AddressBookNumber": "234448", "Name": "THOMAS, CODY JORDAN", "Craft Description": "Turns"},
    {"AddressBookNumber": "1107352", "Name": "WATKINS, KENNETH EDWARD", "Craft Description": "Turns"},
    {"AddressBookNumber": "1096665", "Name": "ATWELL, TALON BRADLEY", "Craft Description": "Turns"},
    {"AddressBookNumber": "108986", "Name": "ROGERS, CHARLES D.", "Craft Description": "Utilities Mech Days"},
    {"AddressBookNumber": "206092", "Name": "TURNER, SHANE M.", "Craft Description": "Utilities Mech Days"},
    {"AddressBookNumber": "1089377", "Name": "ROSE, CAMERON CHASE", "Craft Description": "WTP Mech Days"},
]

# Mapping from work order type codes to descriptive names.
TYPE_MAP: Dict[str, str] = {
    "0": "Break In",
    "1": "Maintenance Order",
    "2": "Material Repair TMJ Order",
    "3": "Capital Project",
    "4": "Urgent Corrective",
    "5": "Emergency Order",
    "6": "PM Restore/Replace",
    "7": "PM Inspection",
    "8": "Follow Up Maintenance Order",
    "9": "Standing W.O. - Do not Delete",
    "B": "Marketing",
    "C": "Cost Improvement",
    "D": "Design Work - ETO",
    "E": "Plant Work - ETO",
    "G": "Governmental/Regulatory",
    "M": "Model W.O. - Eq Mgmt",
    "N": "Template W.O. - CBM Alerts",
    "P": "Project",
    "R": "Rework Order",
    "S": "Shop Order",
    "T": "Tool Order",
    "W": "Case",
    "X": "General Work Request",
    "Y": "Follow Up Work Request",
    "Z": "System Work Request",
}

# Columns used for display and included in the PDF and data downloads.
DISPLAY_COLUMNS: List[str] = ["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"]

# Columns expected in the uploaded Excel export.  If any are missing they will be
# created with missing values.
REQUIRED_TIME_COLUMNS: List[str] = [
    "AddressBookNumber",
    "Name",
    "Production Date",
    "OrderNumber",
    "Sum of Hours.",
    "Hours Estimated",
    "Status",
    "Type",
    "PMFrequency",
    "Description",
    "Problem",
    "Department",
    "Location",
    "Equipment",
    "PM Number",
    "PM",
]

def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Locate the header row in an Excel export by searching for a row that
    contains the expected column names.  Raises if the header cannot be found.
    """
    first_col = df_raw.columns[0]
    mask = df_raw[first_col].astype(str).str.strip() == "AddressBookNumber"
    idx = df_raw.index[mask].tolist()
    if idx:
        return idx[0]
    # Fallback: search the first few rows for known column names.
    for i in range(min(10, len(df_raw))):
        row_vals = df_raw.iloc[i].astype(str).str.strip().tolist()
        if "AddressBookNumber" in row_vals and "Production Date" in row_vals:
            return i
    raise ValueError("Could not locate header row containing 'AddressBookNumber'.")

def _read_excel_twice(file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the uploaded Excel file twice: once without headers and once with
    pandas' default header detection.  This allows us to locate the header row and
    preserve dtypes."""
    data = file.read()
    df_raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    df_hdr = pd.read_excel(io.BytesIO(data))
    return df_raw, df_hdr

def load_timeworkbook(file_like) -> pd.DataFrame:
    """Load and clean the Time on Work Order export."""
    df_raw, _ = _read_excel_twice(file_like)
    header_row = _find_header_row(df_raw)
    file_like.seek(0)
    df = pd.read_excel(file_like, header=header_row)
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    # Ensure all required columns exist
    for c in REQUIRED_TIME_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["AddressBookNumber"] = df["AddressBookNumber"].astype(str).str.strip()
    if "Production Date" in df.columns:
        df["Production Date"] = pd.to_datetime(df["Production Date"], errors="coerce").dt.date
    # Normalize 'Sum of Hours' column names and values
    if "Sum of Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce")
    elif "Sum of Hours." in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours."], errors="coerce")
    elif "Hours" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    else:
        df["Sum of Hours"] = pd.NA
    # Normalize work order number into 'Work Order #'
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
    """Return a DataFrame with the craft order."""
    return pd.DataFrame({"Craft Description": CRAFT_ORDER})

def get_address_book_df() -> pd.DataFrame:
    """Return a cleaned DataFrame of the address book."""
    df = pd.DataFrame(ADDRESS_BOOK)[["AddressBookNumber", "Name", "Craft Description"]]
    df["AddressBookNumber"] = df["AddressBookNumber"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Craft Description"] = df["Craft Description"].astype(str).str.strip()
    return df

def _apply_craft_category(df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    """Assign categories to the craft description column so crafts are ordered consistently."""
    order = order_df["Craft Description"].tolist()
    seen: set[str] = set()
    ordered: List[str] = []
    for c in order:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    categories = ordered + ["Unassigned"]
    df["Craft Description"] = df["Craft Description"].fillna("Unassigned")
    df["Craft Description"] = pd.Categorical(df["Craft Description"], categories=categories, ordered=True)
    return df

def _clean_code(value: object) -> str | None:
    """Normalize numeric work order codes to strings and return None for missing values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _map_type(value):
    """Map a work order type code to its descriptive name using TYPE_MAP."""
    key = _clean_code(value)
    if key is None:
        return pd.NA
    out = TYPE_MAP.get(key, key)
    # Clean up the inspection maintenance wording
    if isinstance(out, str) and out.strip().lower() == "inspection maintenance order":
        return "PM Inspection"
    return out

def prepare_report_data(
    time_df: pd.DataFrame,
    addr_df: pd.DataFrame,
    craft_order_df: pd.DataFrame,
    selected_date,
) -> Dict[str, Any]:
    """Prepare the report data for the selected production date.

    Returns a dictionary with keys:
      - ``groups``: a list of tuples (craft_name, payload) where payload contains a
        'detail' DataFrame with the selected columns.
      - ``full_detail``: the complete filtered DataFrame of selected columns.
      - ``unmapped_people``: a list of address book entries that could not be mapped.
    """
    f = time_df[time_df["Production Date"] == selected_date].copy()
    f["AddressBookNumber"] = f["AddressBookNumber"].astype(str).str.strip()
    addr_df["AddressBookNumber"] = addr_df["AddressBookNumber"].astype(str).str.strip()
    merged = f.merge(
        addr_df[["AddressBookNumber", "Craft Description", "Name"]].rename(columns={"Name": "AB_Name"}),
        on="AddressBookNumber",
        how="left",
    )
    merged["Name"] = merged["Name"].fillna(merged["AB_Name"])
    merged = merged.drop(columns=["AB_Name"])
    # Identify unmapped people
    unmapped: List[Dict[str, Any]] = []
    mask_unmapped = merged["Craft Description"].isna() | (merged["Craft Description"].astype(str).str.len() == 0)
    if mask_unmapped.any():
        unmapped = (
            merged.loc[mask_unmapped, ["AddressBookNumber", "Name"]]
            .drop_duplicates()
            .to_dict("records")
        )
    merged.loc[mask_unmapped, "Craft Description"] = "Unassigned"
    merged = _apply_craft_category(merged, craft_order_df)
    # Ensure display columns exist
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
    """Calculate a table height for the Streamlit dataframe component."""
    rows = len(df) + 1
    row_px = 35
    header_px = 40
    return min(header_px + rows * row_px, 20000)

def _df_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for inclusion in the PDF."""
    out = df.copy()
    if "Sum of Hours" in out.columns:
        out["Sum of Hours"] = (
            pd.to_numeric(out["Sum of Hours"], errors="coerce").fillna(0).map(lambda x: f"{x:.2f}")
        )
    return out

# -----------------------------------------------------------------------------
# PDF helper functions
#
# The following helpers render figures for the PDF export.  To more closely
# mirror the Streamlit dashboard layout, each craft will have a summary page
# containing the bar chart of hours by work order type, some key metrics,
# and a breakdown table.  Additional pages are then added for the detailed
# work order table, splitting the rows across multiple pages as needed.  These
# helpers return matplotlib Figure objects which the main ``build_pdf``
# function writes to the PdfPages instance.

def _create_summary_figure(df_detail: pd.DataFrame, craft_name: str) -> plt.Figure:
    """
    Create a summary figure for a single craft.  The layout consists of a bar
    chart showing hours by work order type, a metrics text block, and a
    breakdown table of hours and percentages by type.  This attempts to
    approximate the Streamlit dashboard view.

    :param df_detail: DataFrame with columns including 'Type' and 'Sum of Hours'.
    :param craft_name: Name of the craft group.
    :return: A matplotlib Figure ready for saving to the PDF.
    """
    # Prepare aggregated data
    df = df_detail.copy()
    df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
    agg = (
        df.groupby("Type", dropna=False)["Sum of Hours"]
        .sum()
        .reset_index()
        .rename(columns={"Sum of Hours": "hours"})
        .sort_values("hours", ascending=False)
    )
    total = float(agg["hours"].sum())
    agg["percent"] = np.where(total > 0, (agg["hours"] / total) * 100.0, 0.0)
    top_type = agg.iloc[0]["Type"] if not agg.empty else "-"
    top_pct = agg.iloc[0]["percent"] if not agg.empty else 0.0

    # Create figure and axes using a grid layout
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{craft_name} — Summary", fontsize=16, y=0.98)
    # Bar chart axis occupies the left portion of the page
    ax_bar = fig.add_axes([0.05, 0.35, 0.55, 0.5])
    default_color = "#333333"
    colors_list = []
    for t in agg["Type"]:
        if isinstance(t, str):
            colors_list.append(_TYPE_COLORS.get(str(t), default_color))
        else:
            colors_list.append(default_color)
    ax_bar.bar(agg["Type"].astype(str), agg["hours"], color=colors_list)
    ax_bar.set_title("Hours by Work Order Type", fontsize=12)
    ax_bar.set_xlabel("Type")
    ax_bar.set_ylabel("Hours")
    ax_bar.tick_params(axis="x", rotation=45, labelsize=8)
    # Reposition metrics to the right side above the breakdown table to avoid overlap
    fig.text(0.65, 0.82, f"Total Hours: {total:,.2f}", fontsize=10, va="top")
    fig.text(0.65, 0.79, f"Top Type: {top_type}", fontsize=10, va="top")
    fig.text(0.65, 0.76, f"% in Top Type: {top_pct:.1f}%", fontsize=10, va="top")
    # Breakdown table axis occupies the right portion below the metrics
    # Allocate the breakdown table to the lower half of the right side.  Using a height
    # of 0.4 (from 0.35 to 0.75) leaves room above for the metrics.
    ax_tbl = fig.add_axes([0.65, 0.35, 0.3, 0.4])
    ax_tbl.axis("off")
    tbl_data = [
        [str(row["Type"]), f"{row['hours']:.2f}", f"{row['percent']:.1f}%"]
        for _, row in agg.iterrows()
    ]
    col_labels = ["Type", "Hours", "%"]
    table = ax_tbl.table(
        cellText=tbl_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    return fig


def _create_detail_table_figures(
    df_detail: pd.DataFrame, craft_name: str, rows_per_page: int = 20
) -> List[plt.Figure]:
    """
    Split a detailed DataFrame into multiple figures for inclusion in the PDF.

    The tables generated by this function adhere to a fixed column width
    specification to ensure that each column fits neatly within the page
    margins regardless of the content.  Each table page includes a header row
    followed by up to ``rows_per_page`` data rows.  Every row is rendered
    at a consistent height; however, if wrapping is required in either the
    ``Description`` or ``Problem`` columns then that particular row's height
    is doubled to accommodate the additional lines of text.

    :param df_detail: DataFrame with the detail rows to display.
    :param craft_name: Name of the craft group.
    :param rows_per_page: Maximum number of rows per page.
    :return: List of matplotlib Figures.
    """
    import textwrap

    # Fixed relative widths for each of the six columns.  These sum to 1.0
    # and should not be changed without adjusting all other values.  The
    # ordering corresponds to the column order produced by _df_for_pdf:
    # Name, Work Order #, Sum of Hours, Type, Description, Problem.
    # Adjusted relative widths for the six columns.  The user requested that
    # the "Work Order #" and "Sum of Hours" columns be reduced by 50% and
    # "Type" be reduced by 30%.  The freed space has been redistributed
    # primarily to the description and problem columns.  These fractions
    # collectively sum to 1.0:
    #   Name          : 0.23
    #   Work Order #  : 0.06  (50% reduction from 0.12)
    #   Sum of Hours  : 0.05  (50% reduction from 0.10)
    #   Type          : 0.105 (30% reduction from 0.15)
    #   Description   : 0.26
    #   Problem       : 0.295
    col_widths = [0.23, 0.06, 0.05, 0.105, 0.26, 0.295]

    figures: List[plt.Figure] = []
    # Convert to a simple DataFrame for PDF output
    df_pdf = _df_for_pdf(df_detail)
    col_labels = df_pdf.columns.tolist()
    total_rows = len(df_pdf)
    # For each page, slice the DataFrame into chunks of at most rows_per_page rows
    for page_num, start in enumerate(range(0, total_rows, rows_per_page), start=1):
        chunk = df_pdf.iloc[start : start + rows_per_page].copy()
        # Wrap long text in Description and Problem columns.  Only wrap when
        # the content length exceeds roughly 60 characters.  Long words are
        # broken to avoid expanding the column width.  The wrap width is
        # chosen empirically to produce approximately two lines of text for
        # very long descriptions.
        for col in ["Description", "Problem"]:
            if col in chunk.columns:
                chunk[col] = (
                    chunk[col]
                    .astype(str)
                    .apply(lambda x: textwrap.fill(x, width=60, break_long_words=True, break_on_hyphens=True))
                )
        # Create the figure and axis for this page.  Use a consistent
        # landscape orientation matching the rest of the report (11" x 8.5").
        fig = plt.figure(figsize=(11, 8.5))
        title = f"{craft_name} — Detail (Page {page_num}/{(total_rows + rows_per_page - 1) // rows_per_page})"
        fig.suptitle(title, fontsize=14, y=0.95)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.85])
        ax.axis("off")
        # Build the table using the wrapped data.  All cells are left‑aligned and
        # fixed column widths are supplied via ``colWidths``.  These widths are
        # relative proportions and will remain consistent across pages.
        table = ax.table(
            cellText=chunk.values,
            colLabels=col_labels,
            cellLoc="left",
            loc="upper left",
            colWidths=col_widths,
        )
        # Disable automatic font scaling and set a uniform font size for
        # consistency across pages.  A small font is necessary to fit
        # multiple rows on a landscape page.
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        # Determine a base height for the rows.  We reserve 0.85 of the axis
        # height for the table body (the remaining space is for title and
        # margins).  Each data row is allotted the same base height and the
        # header row uses the same base height.  When text wraps within a
        # row, we double the height of that specific row.
        base_height = 0.85 / (rows_per_page + 1)
        # Set header row height
        for col_idx in range(len(col_labels)):
            table[(0, col_idx)].set_height(base_height)
        # Set heights for each data row
        for i in range(len(chunk)):
            row_height = base_height
            # Inspect the Description and Problem cells for wrapping
            for col in ["Description", "Problem"]:
                try:
                    cell_text = str(chunk.iloc[i][col])
                except Exception:
                    cell_text = ""
                if "\n" in cell_text:
                    row_height = base_height * 2
                    break
            for col_idx in range(len(col_labels)):
                table[(i + 1, col_idx)].set_height(row_height)
        figures.append(fig)
    return figures

def build_pdf(report: Dict[str, Any], date_label: str) -> bytes:
    """
    Construct a PDF report with summary charts for each craft group along with
    overall statistics.  This implementation uses matplotlib to render bar charts
    and writes each figure to a multipage PDF via PdfPages.  A title page and an
    optional overall summary page are included before the per‑craft pages.

    :param report: The dictionary produced by ``prepare_report_data`` with keys
        ``groups`` (list of (craft_name, payload) tuples) and
        ``full_detail`` (DataFrame containing all records).
    :param date_label: The selected date string used in the report title and
        output file name.
    :return: Bytes representing the PDF document.
    """
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(11, 8.5))
        fig_title.text(0.5, 0.6, "Work Order Reporting App", fontsize=24, ha="center")
        fig_title.text(0.5, 0.5, f"Report for {date_label}", fontsize=16, ha="center")
        plt.axis("off")
        pdf.savefig(fig_title)
        plt.close(fig_title)

        # Overall summary page
        full_detail = report.get("full_detail")
        if isinstance(full_detail, pd.DataFrame) and not full_detail.empty:
            summary_fig = _create_summary_figure(full_detail, "Overall Summary")
            pdf.savefig(summary_fig)
            plt.close(summary_fig)

            # Do not include full-detail pages for the overall summary.  The
            # detail tables will be presented under each craft section.

        # Per-craft pages
        for craft_name, payload in report.get("groups", []):
            df_detail = payload.get("detail")
            if df_detail is None or df_detail.empty:
                continue
            # Summary page for this craft
            craft_summary_fig = _create_summary_figure(df_detail, craft_name)
            pdf.savefig(craft_summary_fig)
            plt.close(craft_summary_fig)
            # Detail table pages for this craft
            craft_detail_figs = _create_detail_table_figures(df_detail, craft_name)
            for fig in craft_detail_figs:
                pdf.savefig(fig)
                plt.close(fig)
    buffer.seek(0)
    return buffer.read()

# === Mini-dashboard helpers for Streamlit ===
import altair as alt

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

def _craft_dashboard_block(df_detail: pd.DataFrame) -> None:
    """Render dashboard metrics and chart for a single craft in the Streamlit app."""
    if df_detail is None or df_detail.empty:
        return
    df = df_detail.copy()
    df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
    agg = (
        df.groupby("Type", dropna=False)["Sum of Hours"]
        .sum()
        .reset_index()
        .rename(columns={"Sum of Hours": "hours"})
        .sort_values("hours", ascending=False)
    )
    total = float(agg["hours"].sum())
    if total <= 0:
        total = 0.0
    agg["percent"] = np.where(total > 0, (agg["hours"] / total) * 100.0, 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Hours", f"{total:,.2f}")
    top_type = agg.iloc[0]["Type"] if not agg.empty else "-"
    top_pct = agg.iloc[0]["percent"] if not agg.empty else 0.0
    c2.metric("Top Type", f"{top_type}")
    c3.metric("% in Top Type", f"{top_pct:.1f}%")
    base = alt.Chart(agg).encode(
        x=alt.X("Type:N", sort="-y", title="Work Order Type"),
        tooltip=[alt.Tooltip("Type:N"), alt.Tooltip("hours:Q", format=".2f")],
    )
    color_scale = alt.Scale(domain=list(_TYPE_COLORS.keys()), range=list(_TYPE_COLORS.values()))
    st.caption("Hours by Work Order Type")
    st.altair_chart(
        base.mark_bar().encode(
            y=alt.Y("hours:Q", title="Hours"),
            color=alt.Color("Type:N", scale=color_scale),
        ),
        use_container_width=True,
    )
    st.caption("Breakdown (Hours & % of Craft Total)")
    st.dataframe(_style_breakdown(agg[["Type", "hours"]]), use_container_width=True, hide_index=True)

def _style_breakdown(df: pd.DataFrame):
    """Return a pandas Styler so 'Type' is right‑aligned and 'hours' is left‑aligned."""
    if df is None or df.empty:
        return df
    try:
        styler = df.style
        styler = styler.set_properties(subset=["Type"], **{"text-align": "right"})
        styler = styler.set_properties(subset=["hours"], **{"text-align": "left"})
        styler = styler.set_table_styles(
            [
                {"selector": "th.col_heading.level0.col0", "props": [("text-align", "right")]},
                {"selector": "th.col_heading.level0.col1", "props": [("text-align", "left")]},
            ],
            overwrite=False,
        )
        return styler
    except Exception:
        # Fallback: return the raw DataFrame if something goes wrong with styling
        return df

# === Table styling for Type colors ===
def _style_types(df: pd.DataFrame):
    """Apply background colors based on the Type to the DataFrame in Streamlit."""
    if df is None or df.empty or "Type" not in df.columns:
        return df
    def _style_cell(v):
        color = _TYPE_COLORS.get(str(v), None)
        return f"background-color: {color}; color: white; font-weight: 600;" if color else ""
    try:
        return df.style.applymap(_style_cell, subset=["Type"])
    except Exception:
        # Fallback: return unstyled if Styler isn't supported
        return df

# === Streamlit page configuration and UI ===
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
selected_label = st.selectbox("Select Production Date", options=date_labels, index=len(date_labels) - 1)
selected_date = label_to_date[selected_label]

report = prepare_report_data(time_df, addr_df, craft_df, selected_date)

st.sidebar.subheader("Export")
# To generate an exact snapshot of the current dashboard as a PDF, we embed
# a client‑side script that leverages html2canvas and jsPDF.  When the
# button is clicked, the sidebar is temporarily hidden, the entire page is
# captured as a high‑resolution canvas, and a landscape PDF is generated
# and automatically downloaded.  After generation, the sidebar is restored.
import streamlit.components.v1 as components  # type: ignore
snapshot_button_html = """
<button id=\"snapshotBtn\" style=\"margin-top:10px;padding:6px 12px;font-size:14px;\">Download Dashboard PDF</button>
<script src=\"https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js\"></script>
<script src=\"https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js\"></script>
<script>
  document.getElementById('snapshotBtn').addEventListener('click', function() {
    const { jsPDF } = window.jspdf;
    const sidebar = document.querySelector('section[data-testid="stSidebar"]');
    const originalDisplay = sidebar ? sidebar.style.display : '';
    if (sidebar) {
      sidebar.style.display = 'none';
    }
    setTimeout(function() {
      html2canvas(document.body, { scale: 2 }).then(canvas => {
        const imgData = canvas.toDataURL('image/png');
        // Create a landscape PDF with dimensions based on the captured canvas
        const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [canvas.width, canvas.height] });
        pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
        pdf.save('dashboard.pdf');
        if (sidebar) {
          sidebar.style.display = originalDisplay;
        }
      });
    }, 300);
  });
</script>
"""
components.html(snapshot_button_html, height=120)

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
        column_config=col_cfg,
    )
    st.markdown("---")

if not report["full_detail"].empty:
    csv_bytes = report["full_detail"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered detail (CSV)",
        data=csv_bytes,
        file_name=f"workorder_detail_{selected_label.replace('/', '-')}.csv",
        mime="text/csv",
    )