
import re
from pathlib import Path

APP = Path("app.py")
if not APP.exists():
    print("ERROR: app.py not found in the current directory. Run this script from your repo root.")
    raise SystemExit(1)

src = APP.read_text(encoding="utf-8")

original_src = src
backup = Path("app.py.bak")
backup.write_text(src, encoding="utf-8")

changed = False
report = []

# 1) Ensure Cost Center extraction (from header or column N index 13).
# Insert after a likely normalization area; otherwise add a safe guard near top-level load function.
# We try to locate a function that loads Excel, commonly named load_timeworkbook or similar.

def inject_cost_center_extraction(s: str) -> str:
    pattern_funcs = [
        r"def\s+load_timeworkbook\s*\([^\)]*\)\s*:\s*(?:\n[ \t].*)+",
        r"def\s+load_.*work.*\([^\)]*\)\s*:\s*(?:\n[ \t].*)+",
    ]
    insertion_code = (
        "\n"
        "    # --- Cost Center from column 'N' (0-based index 13) ---\n"
        "    if 'Cost Center' in df.columns:\n"
        "        df['Cost Center'] = df['Cost Center'].astype(str).str.strip()\n"
        "    else:\n"
        "        try:\n"
        "            col_n = df.columns[13]\n"
        "            df['Cost Center'] = df[col_n].astype(str).str.strip()\n"
        "        except Exception:\n"
        "            df['Cost Center'] = pd.NA\n"
    )
    # Insert after the first block that sets up df (after 'df =' line) inside the load function
    func_match = re.search(r"(def\s+load_timeworkbook\s*\([^\)]*\)\s*:\s*\n)((?:[ \t].*\n)+)", s)
    if func_match:
        body = func_match.group(2)
        # find a safe anchor line where df likely exists
        anchor = re.search(r"^[ \t]*df\s*=\s*.*\n", body, flags=re.MULTILINE)
        if anchor and "Cost Center" not in body:
            insert_pos = func_match.start(2) + anchor.end()
            return s[:insert_pos] + insertion_code + s[insert_pos:]
    # Fallback: if we can't find the exact anchor but no existing Cost Center logic, append at end of function
    if "Cost Center" not in s:
        s = re.sub(r"(def\s+load_timeworkbook\s*\([^\)]*\)\s*:\s*\n)((?:[ \t].*\n)+)",
                   lambda m: m.group(1) + m.group(2) + insertion_code, s, count=1)
    return s

before = src
src = inject_cost_center_extraction(src)
if src != before:
    changed = True
    report.append("Injected Cost Center extraction (from header or column N).")
else:
    # If already present, note it
    if "Cost Center" in src:
        report.append("Cost Center extraction appears to already exist; skipped.")
    else:
        report.append("WARNING: Could not automatically insert Cost Center extraction (no load function found).")

# 2) Ensure DISPLAY_COLUMNS (or equivalent) includes Cost Center between Type and Description.
def ensure_display_columns(s: str) -> str:
    # Try to find a list named DISPLAY_COLUMNS
    m = re.search(r"(DISPLAY_COLUMNS\s*=\s*\[\s*)([^\]]+)(\])", s, flags=re.DOTALL)
    if not m:
        return s
    items = m.group(2)
    # Normalize spacing
    cols = [c.strip().strip("',\"") for c in re.findall(r"['\"][^'\"]+['\"]", items)]
    # Insert Cost Center after Type and before Description
    def insert_cc(seq):
        if "Cost Center" in seq:
            return seq
        try:
            i_type = seq.index("Type")
        except ValueError:
            return seq
        # Place before Description if it exists right after Type
        insert_at = i_type + 1
        seq = seq[:insert_at] + ["Cost Center"] + seq[insert_at:]
        return seq
    new_cols = insert_cc(cols[:])
    if new_cols != cols:
        new_items = ", ".join([f'"{c}"' for c in new_cols])
        return s[:m.start(2)] + new_items + s[m.end(2):]
    return s

before = src
src = ensure_display_columns(src)
if src != before:
    changed = True
    report.append('Inserted "Cost Center" into DISPLAY_COLUMNS between Type and Description.')
else:
    report.append('DISPLAY_COLUMNS unchanged (already includes "Cost Center" or not found).')

# 3) Column config: set "Cost Center": TextColumn("CC", ...)
def set_colcfg_cc_label(s: str) -> str:
    # Look for a dict assigned to col_cfg = { ... }
    m = re.search(r"(col_cfg\s*=\s*\{\s*)((?:.|\n)*?)(\}\s*)", s)
    if not m:
        return s
    body = m.group(2)
    # If a Cost Center entry exists, replace its label with "CC"; else add it after Type
    # Replace existing label
    new_body = re.sub(
        r'"Cost Center"\s*:\s*st\.column_config\.TextColumn\(\s*["\'][^"\']*["\']\s*,',
        r'"Cost Center": st.column_config.TextColumn("CC",',
        body
    )
    if new_body == body:
        # No existing entry — try to insert after the "Type" entry
        new_body = re.sub(
            r'("Type"\s*:\s*st\.column_config\.TextColumn\([^\)]*\)\s*,\s*)',
            r'\1"Cost Center": st.column_config.TextColumn("CC", width=120),\n        ',
            body
        )
    if new_body != body:
        return s[:m.start(2)] + new_body + s[m.end(2):]
    return s

before = src
src = set_colcfg_cc_label(src)
if src != before:
    changed = True
    report.append('Updated column_config: "Cost Center" header label set to "CC" (added if missing).')
else:
    report.append('column_config unchanged (entry possibly already present).')

# 4) Remove CSV export button/feature
def remove_csv_block(s: str) -> str:
    # Remove common patterns for CSV download button blocks
    patterns = [
        # Pattern where csv or csv_bytes is created then st.download_button called
        r"\n[ \t]*if\s+not\s+report\[[\"']full_detail[\"']\]\.empty:\s*\n"
        r"(?:[ \t]*[A-Za-z_][A-Za-z0-9_]*\s*=\s*report\[[\"']full_detail[\"']\]\.to_csv\(index=False\).*?\n)"
        r"(?:[ \t]*st\.download_button\([\s\S]*?\)\s*\n)",
        # More general: any st.download_button whose label mentions CSV
        r"\n[ \t]*st\.download_button\([\s\S]*?(CSV|csv)[\s\S]*?\)\s*\n",
    ]
    for pat in patterns:
        s_new = re.sub(pat, "\n", s, flags=re.MULTILINE)
        if s_new != s:
            s = s_new
    return s

before = src
src = remove_csv_block(src)
if src != before:
    changed = True
    report.append("Removed CSV export button/feature blocks.")
else:
    report.append("No CSV download blocks found or already removed.")

# 5) Keep PDF at 6 columns (drop Cost Center inside _df_for_pdf if needed).
def enforce_pdf_six_columns(s: str) -> str:
    # Try to identify _df_for_pdf and make sure it selects only the 6 columns
    func = re.search(r"(def\s+_df_for_pdf\s*\([^\)]*\)\s*:\s*\n)((?:[ \t].*\n)+)", s)
    if not func:
        return s
    body = func.group(2)
    if "pdf_cols" in body and "Cost Center" in body:
        body = re.sub(r"pdf_cols\s*=\s*\[([^\]]+)\]",
                      'pdf_cols = ["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"]',
                      body)
        return s[:func.start(2)] + body + s[func.end(2):]
    # If pdf_cols not present, we try to enforce selection near return
    body2 = re.sub(
        r"out\s*=\s*out\[[^\]]+\]",
        'out = out[["Name", "Work Order #", "Sum of Hours", "Type", "Description", "Problem"]]',
        body
    )
    return s[:func.start(2)] + body2 + s[func.end(2):]

before = src
src = enforce_pdf_six_columns(src)
if src != before:
    changed = True
    report.append("Ensured PDF export keeps 6 columns (no Cost Center in PDF).")
else:
    report.append("PDF column selection unchanged (already 6 columns or function not found).")

# Write back
if changed:
    APP.write_text(src, encoding="utf-8")
    print("✅ Patch applied successfully.")
else:
    print("ℹ️ No changes were necessary (app already matched desired state).")

print("\n--- Summary ---")
for line in report:
    print("-", line)
print("----------------")
