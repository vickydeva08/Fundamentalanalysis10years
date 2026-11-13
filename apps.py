import io
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="10-Year Fundamental Analysis & DCF Model",
    page_icon="📊",
    layout="wide",
)

st.title("📊 10-Year Fundamental Analysis & DCF Valuation Model")
st.markdown(
    "Enter any **Screener.in** company URL (e.g., TCS, ITC, Reliance) to automatically scrape financial statements, run forensic checks, execute a 10-year DCF model, and export a fully-formatted 5-tab Excel workbook."
)

# Input form & interactive sliders
url = st.text_input(
    "Screener.in Company URL:",
    value="https://www.screener.in/company/TCS/consolidated/",
    help="Paste the consolidated or standalone Screener URL.",
)

col1, col2, col3, col4 = st.columns(4)
with col1:
  wacc = st.slider("WACC / Discount Rate", 0.08, 0.18, 0.125, 0.005)
with col2:
  stage1_g = st.slider("Stage 1 Growth (Y1-Y5)", 0.04, 0.20, 0.08, 0.005)
with col3:
  stage2_g = st.slider("Stage 2 Growth (Y6-Y10)", 0.03, 0.15, 0.06, 0.005)
with col4:
  term_g = st.slider("Terminal Growth Rate (g)", 0.02, 0.06, 0.045, 0.005)


@st.cache_data(show_spinner=True)
def fetch_and_process_screener(screener_url, wacc_val, s1_val, s2_val, gt_val):
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/126.0.0.0 Safari/537.36"
      )
  }
  resp = requests.get(screener_url, headers=headers, timeout=20)
  resp.raise_for_status()
  soup = BeautifulSoup(resp.text, "html.parser")

  # Extract metadata (CMP, Market Cap)
  meta = {}
  for li in soup.select("#top-ratios li"):
    name_el = li.select_one(".name")
    val_el = li.select_one(".number")
    if name_el and val_el:
      key = name_el.get_text(strip=True)
      raw_val = val_el.get_text(strip=True).replace(",", "")
      try:
        meta[key] = float(raw_val)
      except ValueError:
        meta[key] = raw_val

  # Extract financial tables
  tables = {}
  for sec_id in ["profit-loss", "balance-sheet", "cash-flow"]:
    sec = soup.find("section", id=sec_id)
    if not sec:
      continue
    table = sec.find("table")
    if not table:
      continue
    cols = [th.get_text(strip=True) for th in table.select("thead th")]
    cols[0] = "Metric"
    rows = []
    for tr in table.select("tbody tr"):
      cells = [
          td.get_text(strip=True).replace("+", "").strip()
          for td in tr.find_all("td")
      ]
      if not cells or len(cells) != len(cols):
        continue
      parsed = [cells[0]]
      for val in cells[1:]:
        clean = val.replace(",", "").replace("%", "")
        try:
          num = float(clean)
          if "%" in val:
            num = num / 100.0
          parsed.append(num)
        except ValueError:
          parsed.append(0.0)
      rows.append(parsed)
    df = pd.DataFrame(rows, columns=cols).set_index("Metric")
    fy_cols = [c for c in df.columns if c.upper() != "TTM"][-10:]
    tables[sec_id] = df[fy_cols]

  return meta, tables


if st.button("🚀 Run Automated Valuation Model", type="primary"):
  try:
    meta, tables = fetch_and_process_screener(url, wacc, stage1_g, stage2_g, term_g)
    pnl = tables["profit-loss"]
    bs = tables["balance-sheet"]
    cf = tables["cash-flow"]
    years = list(pnl.columns)
    n = len(years)

    def get_row(df, candidates, default=0.0):
      for label in candidates:
        for idx in df.index:
          if label.lower() in idx.lower():
            return df.loc[idx].astype(float).values
      return np.full(df.shape[1], default, dtype=float)

    sales = get_row(pnl, ["Sales", "Revenue"])
    op_profit = get_row(pnl, ["Operating Profit", "Financing Profit"])
    depr = get_row(pnl, ["Depreciation"])
    tax_pct = get_row(pnl, ["Tax %"])
    pat = get_row(pnl, ["Net Profit"])
    ebit = op_profit
    eff_tax = np.where(tax_pct > 1, tax_pct / 100.0, tax_pct)
    nopat = ebit * (1.0 - eff_tax)

    total_equity = get_row(bs, ["Equity Capital"]) + get_row(bs, ["Reserves"])
    total_debt = get_row(bs, ["Borrowings"])
    investments = get_row(bs, ["Investments"])
    invested_cap = total_equity + total_debt - investments

    cfo = get_row(cf, ["Cash from Operating Activity"])
    fixed_assets = get_row(bs, ["Fixed Assets"])
    cwip = get_row(bs, ["CWIP"])

    capex = np.zeros(n)
    capex[0] = depr[0]
    for i in range(1, n):
      capex[i] = max(
          (fixed_assets[i] + cwip[i])
          - (fixed_assets[i - 1] + cwip[i - 1])
          + depr[i],
          0.0,
      )
    fcff = cfo - capex

    # DCF Model Projections
    base_fcff = fcff[-1]
    proj_fcff = []
    curr = base_fcff
    for yr in range(1, 11):
      g = stage1_g if yr <= 5 else stage2_g
      curr *= 1.0 + g
      proj_fcff.append(curr)

    pv_fcff = [f / ((1.0 + wacc) ** idx) for idx, f in enumerate(proj_fcff, 1)]
    sum_pv_fcf = sum(pv_fcff)
    term_val = proj_fcff[-1] * (1.0 + term_g) / (wacc - term_g)
    pv_tv = term_val / ((1.0 + wacc) ** 10)
    enterprise_val = sum_pv_fcf + pv_tv
    net_debt = total_debt[-1] - investments[-1]
    equity_val = enterprise_val - net_debt

    cmp = float(meta.get("Current Price", 0.0))
    mcap = float(meta.get("Market Cap", 0.0))
    shares_cr = (
        (mcap / cmp) if (cmp > 0 and mcap > 0) else (total_equity[-1] / 10.0)
    )
    fair_value = equity_val / shares_cr if shares_cr > 0 else 0.0
    mos = ((fair_value - cmp) / fair_value) if fair_value > 0 else 0.0

    st.success("✅ Valuation and Financial Model Generated Successfully!")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Current Market Price (CMP)", f"₹{cmp:,.2f}")
    col_b.metric(
        "DCF Intrinsic Fair Value", f"₹{fair_value:,.2f}", delta=f"{mos:+.1%} MoS"
    )
    col_c.metric("10-Yr Cumulative FCFF", f"₹{np.sum(fcff):,.0f} Cr")

    # Build Excel Workbook in memory
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    navy = "1B365D"
    font_hdr = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_hdr = PatternFill(start_color=navy, end_color=navy, fill_type="solid")

    ws1 = wb.create_sheet("Executive Summary")
    ws1["A1"] = "AUTOMATED FUNDAMENTAL & DCF SUMMARY"
    ws1.append([])
    ws1.append(["Metric", "Value"])
    for c in range(1, 3):
      ws1.cell(3, c).font = font_hdr
      ws1.cell(3, c).fill = fill_hdr

    summary_rows = [
        ["Current Market Price", cmp],
        ["Fair Value per Share", fair_value],
        ["Margin of Safety (%)", mos],
        ["Enterprise Value (₹ Cr)", enterprise_val],
        ["Equity Value (₹ Cr)", equity_val],
    ]
    for r_i, r_val in enumerate(summary_rows, start=4):
      ws1.append(r_val)

    excel_buffer = io.BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Complete 5-Tab Excel Model",
        data=excel_buffer,
        file_name="Fundamental_Analysis_DCF_Model.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )

  except Exception as e:
    st.error(f"Error processing URL: {e}")