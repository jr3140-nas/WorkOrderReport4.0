
import io
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# ReportLab
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# Altair chart export (optional)
import altair as alt
try:
    # Prefer vl-convert-python which doesn't require node
    from altair_saver import save as alt_save
    HAVE_ALTAIR_SAVER = True
except Exception:
    HAVE_ALTAIR_SAVER = False

def _hours_by_type_chart(df_detail: pd.DataFrame) -> Optional[alt.Chart]:
    if df_detail is None or df_detail.empty:
        return None
    df = df_detail.copy()
    if "Sum of Hours" not in df.columns or "Type" not in df.columns:
        return None
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
        tooltip=[alt.Tooltip("Type:N"), alt.Tooltip("hours:Q", format=".2f")]
    )
    chart = base.mark_bar().encode(y=alt.Y("hours:Q", title="Hours"))
    chart = chart.properties(width=900, height=360)
    return chart

def _chart_to_png_bytes(chart: "alt.Chart") -> Optional[bytes]:
    if chart is None:
        return None
    if not HAVE_ALTAIR_SAVER:
        return None
    try:
        buf = io.BytesIO()
        # altair_saver infers format from filename; use BytesIO via temp PNG file-like name
        # Work around by saving to bytes via alt_save with fmt='png' returning bytes (supported)
        png_bytes = alt_save(chart, fmt="png")
        return png_bytes
    except Exception:
        return None

def build_dashboard_pdf(df_detail: pd.DataFrame, title_suffix: str = "") -> bytes:
    """Build a PDF including key visuals and summary tables from the dashboard."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                            rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    story = []
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    subtitle_style = styles["Heading3"]
    normal = styles["BodyText"]

    # Title
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title_text = f"Work Order Dashboard {('— ' + title_suffix) if title_suffix else ''}"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph(f"Generated: {now}", subtitle_style))
    story.append(Spacer(1, 12))

    # Visual: Hours by Type
    chart = _hours_by_type_chart(df_detail)
    chart_png = _chart_to_png_bytes(chart) if chart is not None else None
    if chart_png:
        img_buf = io.BytesIO(chart_png)
        # Scale image to page width
        img = Image(img_buf, width=760, height=304)  # fits landscape letter margins nicely
        story.append(Paragraph("Hours by Work Order Type", subtitle_style))
        story.append(img)
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("Hours by Work Order Type (chart unavailable in this environment)", subtitle_style))
        story.append(Spacer(1, 6))

    # Summary table: hours and percent by Type
    df = df_detail.copy()
    if "Sum of Hours" in df.columns and "Type" in df.columns:
        df["Sum of Hours"] = pd.to_numeric(df["Sum of Hours"], errors="coerce").fillna(0.0)
        agg = (df.groupby("Type", dropna=False)["Sum of Hours"].sum().reset_index())
        total = float(agg["Sum of Hours"].sum()) or 0.0
        agg["% of Total"] = np.where(total > 0, (agg["Sum of Hours"]/total)*100.0, 0.0)
        agg = agg.sort_values("Sum of Hours", ascending=False)
        data = [["Type", "Hours", "% of Total"]] + [[str(t), f"{h:.2f}", f"{p:.1f}%"] for t,h,p in agg.values]
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.black),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (1,1), (-1,-1), "RIGHT"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # Optional: include first N rows of detail
    if not df_detail.empty:
        cols = [c for c in ["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"] if c in df_detail.columns]
        df_small = df_detail[cols].head(25)
        data = [cols] + [list(map(lambda x: str(x) if x is not None else "", row)) for row in df_small.itertuples(index=False)]
        tbl = Table(data, repeatRows=1, colWidths=[180, 100, 80, 160, 260, 260][:len(cols)])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (2,1), (2,-1), "RIGHT"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        story.append(Paragraph("Sample of filtered detail (first 25 rows)", subtitle_style))
        story.append(tbl)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
