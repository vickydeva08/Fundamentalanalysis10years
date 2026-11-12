11.12 7:58 PM
apps.py
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
def parse_screener_tables(url: str) -> dict:
    """Scrapes P&L, Balance Sheet, Cash Flow, and top ratios from a Screener.in URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Extract top ratio cards (Market Cap, Current Price, Stock P/E, Book Value, etc.)
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
    # Extract section tables
    sections = {}
    for sec_id in ["profit-loss", "balance-sheet", "cash-flow", "ratios"]:
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
            cells = [td.get_text(strip=True).replace("+", "").strip() for td in tr.find_all("td")]
            if not cells or len(cells) != len(cols):
                continue
            parsed_row = [cells[0]]
            for val in cells[1:]:
                clean = val.replace(",", "").replace("%", "")
                try:
                    num = float(clean)
                    if "%" in val:
                        num = num / 100.0
                    parsed_row.append(num)
                except ValueError:
                    parsed_row.append(0.0)
            rows.append(parsed_row)
        df = pd.DataFrame(rows, columns=cols).set_index("Metric")
        # Keep up to the latest 10 completed fiscal years (exclude TTM if present)
        fy_cols = [c for c in df.columns if c.upper() != "TTM"][-10:]
        sections[sec_id] = df[fy_cols]
    return {"meta": meta, "tables": sections}
def get_row(df: pd.DataFrame, candidates: list, default=0.0) -> np.ndarray:
    """Safely fetches the first matching row label from a parsed Screener table."""
    for label in candidates:
        for idx in df.index:
            if label.lower() in idx.lower():
                return df.loc[idx].astype(float).values
    return np.full(df.shape[1], default, dtype=float)
def run_automated_valuation(
    screener_url: str,
    wacc: float = 0.125,
    stage1_g: float = 0.08,
    stage2_g: float = 0.06,
    term_g: float = 0.045,
):
    ticker = re.search(r"/company/([^/]+)/", screener_url).group(1)
    print(f"Fetching live financial data for {ticker} from {screener_url} ...")
    scraped = parse_screener_tables(screener_url)
    meta = scraped["meta"]
    pnl = scraped["tables"]["profit-loss"]
    bs = scraped["tables"]["balance-sheet"]
    cf = scraped["tables"]["cash-flow"]
    years = list(pnl.columns)
    n = len(years)
    # 1. Income Statement Extraction
    sales = get_row(pnl, ["Sales", "Revenue"])
    op_profit = get_row(pnl, ["Operating Profit", "Financing Profit"])
    other_inc = get_row(pnl, ["Other Income"])
    interest = get_row(pnl, ["Interest"])
    depr = get_row(pnl, ["Depreciation"])
    pbt = get_row(pnl, ["Profit before tax"])
    tax_pct = get_row(pnl, ["Tax %"])
    pat = get_row(pnl, ["Net Profit"])
    ebitda = op_profit + depr
    ebit = op_profit
    eff_tax = np.where(tax_pct > 1, tax_pct / 100.0, tax_pct)
    nopat = ebit * (1.0 - eff_tax)
    # 2. Balance Sheet & Invested Capital
    eq_cap = get_row(bs, ["Equity Capital"])
    reserves = get_row(bs, ["Reserves"])
    total_equity = eq_cap + reserves
    total_debt = get_row(bs, ["Borrowings"])
    other_liab = get_row(bs, ["Other Liabilities"])
    total_assets = get_row(bs, ["Total Assets"])
    fixed_assets = get_row(bs, ["Fixed Assets"])
    cwip = get_row(bs, ["CWIP"])
    investments = get_row(bs, ["Investments"])
    other_assets = get_row(bs, ["Other Assets"])
    # Screener web table aggregates Cash inside Other Assets; use Cash & Investments if expanded
    invested_cap = total_equity + total_debt - investments
    # 3. Cash Flow & FCFF
    cfo = get_row(cf, ["Cash from Operating Activity"])
    cfi = get_row(cf, ["Cash from Investing Activity"])
    capex = np.zeros(n)
    capex[0] = depr[0]
    for i in range(1, n):
        capex[i] = max((fixed_assets[i] + cwip[i]) - (fixed_assets[i - 1] + cwip[i - 1]) + depr[i], 0.0)
    fcff = cfo - capex
    # 4. Ratios & Forensic Metrics
    op_margin = np.divide(ebit, sales, out=np.zeros_like(ebit), where=sales != 0)
    net_margin = np.divide(pat, sales, out=np.zeros_like(pat), where=sales != 0)
    roe = np.divide(pat, total_equity, out=np.zeros_like(pat), where=total_equity != 0)
    roce = np.divide(ebit, (total_assets - other_liab), out=np.zeros_like(ebit), where=(total_assets - other_liab) != 0)
    roic = np.divide(nopat, invested_cap, out=np.zeros_like(nopat), where=invested_cap != 0)
    accrual_ratio = np.divide((pat - cfo), total_assets, out=np.zeros_like(pat), where=total_assets != 0)
    # 5. 10-Year Multi-Stage DCF Model
    base_fcff = fcff[-1]
    proj_fcff = []
    curr = base_fcff
    for yr in range(1, 11):
        g = stage1_g if yr <= 5 else stage2_g
        curr *= (1.0 + g)
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
    shares_cr = (mcap / cmp) if (cmp > 0 and mcap > 0) else (eq_cap[-1] / 1.0)
    fair_value = equity_val / shares_cr if shares_cr > 0 else 0.0
    mos = ((fair_value - cmp) / fair_value) if fair_value > 0 else 0.0
    # 6. Export 5-Tab Styled Excel Workbook
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    navy = "1B365D"
    font_title = Font(name="Calibri", size=13, bold=True, color=navy)
    font_hdr = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_hdr = PatternFill(start_color=navy, end_color=navy, fill_type="solid")
    font_bold = Font(name="Calibri", size=11, bold=True)
    # TAB 1: Executive Summary
    ws1 = wb.create_sheet("Executive Summary")
    ws1["A1"] = f"{ticker} — AUTOMATED 10-YEAR FUNDAMENTAL & DCF SUMMARY"
    ws1["A1"].font = font_title
    ws1.append([])
    ws1.append(["Metric", "Value", "Notes"])
    for c in range(1, 4):
        ws1.cell(3, c).font = font_hdr
        ws1.cell(3, c).fill = fill_hdr
    rev_cagr = (sales[-1] / sales[0]) ** (1 / (n - 1)) - 1 if sales[0] > 0 else 0
    nopat_cagr = (nopat[-1] / nopat[0]) ** (1 / (n - 1)) - 1 if nopat[0] > 0 else 0
    cfo_pat = np.sum(cfo) / np.sum(pat) if np.sum(pat) > 0 else 0
    t1_rows = [
        ["Current Market Price (CMP)", cmp, f"Live Screener CMP for {ticker}"],
        ["DCF Intrinsic Fair Value / Share", fair_value, f"WACC {wacc:.1%}, Term g {term_g:.1%}"],
        ["Margin of Safety (%)", mos, "Green >15%, Yellow -15%..15%, Red <-15%"],
        [f"{n}-Year Revenue CAGR", rev_cagr, f"{years[0]} to {years[-1]}"],
        [f"{n}-Year NOPAT CAGR", nopat_cagr, f"{years[0]} to {years[-1]}"],
        [f"{n}-Year Average ROIC", float(np.mean(roic)), f"Spread vs WACC: {np.mean(roic) - wacc:+.1%}"],
        [f"{n}-Year Cumulative FCFF (₹ Cr)", float(np.sum(fcff)), "Total Free Cash Flow to Firm"],
        [f"{n}-Year CFO / Net Profit Ratio", cfo_pat, "Cash conversion efficiency"],
    ]
    for r_i, r_val in enumerate(t1_rows, start=4):
        ws1.append(r_val)
        ws1.cell(r_i, 1).font = font_bold
        cell = ws1.cell(r_i, 2)
        if r_i in [4, 5]:
            cell.number_format = '"₹"#,##0.00'
        elif r_i == 10:
            cell.number_format = '"₹"#,##0'
        else:
            cell.number_format = "0.0%"
    # TAB 2: Historical Financials
    ws2 = wb.create_sheet("Historical Financials")
    ws2["A1"] = f"{ticker} — HISTORICAL FINANCIAL DATA ({years[0]} – {years[-1]}) [₹ Cr]"
    ws2["A1"].font = font_title
    ws2.append([])
    ws2.append(["Line Item"] + years)
    for c in range(1, n + 2):
        ws2.cell(3, c).font = font_hdr
        ws2.cell(3, c).fill = fill_hdr
    hist_table = [
        ["Revenue"] + list(sales),
        ["EBITDA"] + list(ebitda),
        ["Depreciation"] + list(depr),
        ["Operating Profit (EBIT)"] + list(ebit),
        ["Effective Tax Rate"] + list(eff_tax),
        ["NOPAT"] + list(nopat),
        ["Net Profit (PAT)"] + list(pat),
        ["Total Debt"] + list(total_debt),
        ["Total Equity"] + list(total_equity),
        ["Total Assets"] + list(total_assets),
        ["Investments"] + list(investments),
        ["Invested Capital"] + list(invested_cap),
        ["Cash Flow from Ops (CFO)"] + list(cfo),
        ["Estimated CapEx"] + list(capex),
        ["Free Cash Flow (FCFF)"] + list(fcff),
    ]
    for r_i, r_val in enumerate(hist_table, start=4):
        ws2.append(r_val)
        ws2.cell(r_i, 1).font = font_bold
        for c in range(2, n + 2):
            ws2.cell(r_i, c).number_format = "0.0%" if r_i == 8 else "#,##0"
    # TAB 3: Ratios & Forensics
    ws3 = wb.create_sheet("Ratios & Forensics")
    ws3["A1"] = f"{ticker} — RATIOS & FORENSIC METRICS ({years[0]} – {years[-1]})"
    ws3["A1"].font = font_title
    ws3.append([])
    ws3.append(["Metric"] + years + ["Min", "Median", "Max"])
    for c in range(1, n + 5):
        ws3.cell(3, c).font = font_hdr
        ws3.cell(3, c).fill = fill_hdr
    ratio_rows = [
        ("Operating Margin %", op_margin),
        ("Net Margin %", net_margin),
        ("ROE %", roe),
        ("ROCE %", roce),
        ("ROIC %", roic),
        ("Accrual Ratio", accrual_ratio),
    ]
    for r_i, (label, arr) in enumerate(ratio_rows, start=4):
        vals = list(arr) + [float(np.min(arr)), float(np.median(arr)), float(np.max(arr))]
        ws3.append([label] + vals)
        ws3.cell(r_i, 1).font = font_bold
        for c in range(2, n + 5):
            ws3.cell(r_i, c).number_format = "0.00%" if label == "Accrual Ratio" else "0.0%"
    # TAB 4: DCF Model
    ws4 = wb.create_sheet("DCF Valuation")
    ws4["A1"] = f"{ticker} — 10-YEAR DCF VALUATION MODEL"
    ws4["A1"].font = font_title
    ws4.append([])
    ws4.append(["DCF Parameter / Output", "Value"])
    ws4.cell(3, 1).font = font_hdr
    ws4.cell(3, 2).font = font_hdr
    ws4.cell(3, 1).fill = fill_hdr
    ws4.cell(3, 2).fill = fill_hdr
    dcf_summary = [
        ["WACC", wacc],
        ["Stage 1 Growth (Y1-Y5)", stage1_g],
        ["Stage 2 Growth (Y6-Y10)", stage2_g],
        ["Terminal Growth Rate (g)", term_g],
        ["Base Year FCFF (₹ Cr)", base_fcff],
        ["PV of 10-Year FCFF (₹ Cr)", sum_pv_fcf],
        ["PV of Terminal Value (₹ Cr)", pv_tv],
        ["Enterprise Value (₹ Cr)", enterprise_val],
        ["Less: Net Debt (₹ Cr)", net_debt],
        ["Implied Equity Value (₹ Cr)", equity_val],
        ["Diluted Shares (Cr)", shares_cr],
        ["Intrinsic Fair Value / Share (₹)", fair_value],
        ["Current Market Price (₹)", cmp],
        ["Margin of Safety (%)", mos],
    ]
    for r_i, r_val in enumerate(dcf_summary, start=4):
        ws4.append(r_val)
        ws4.cell(r_i, 1).font = font_bold
        c = ws4.cell(r_i, 2)
        if r_i in [4, 5, 6, 7, 17]:
            c.number_format = "0.0%"
        elif r_i in [15, 16]:
            c.number_format = '"₹"#,##0.00'
        else:
            c.number_format = "#,##0.00"
    # TAB 5: Google Sheets Blueprint
    ws5 = wb.create_sheet("Sheets Blueprint")
    ws5["A1"] = "GOOGLE SHEETS FORMULAS & FORMATTING BLUEPRINT"
    ws5["A1"].font = font_title
    ws5.append([])
    ws5.append(["Component", "Google Sheets Formula", "Purpose"])
    for c in range(1, 4):
        ws5.cell(3, c).font = font_hdr
        ws5.cell(3, c).fill = fill_hdr
    ws5.append(["Live Price", f'=GOOGLEFINANCE("NSE:{ticker}", "price")', "Fetches real-time NSE share price"])
    ws5.append(["Live Market Cap (₹ Cr)", f'=GOOGLEFINANCE("NSE:{ticker}", "marketcap")/10000000', "Converts live market cap to ₹ Cr"])
    ws5.append(["NOPAT", "=EBIT*(1-Effective_Tax_Rate)", "Net Operating Profit After Tax"])
    ws5.append(["ROIC", "=NOPAT/Invested_Capital", "Return on Invested Capital"])
    ws5.append(["Margin of Safety", "=(Fair_Value-CMP)/Fair_Value", "Green >=15%, Yellow -15%..15%, Red <-15%"])
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            sheet.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)
    out_file = f"{ticker}_10Year_Fundamental_Analysis_and_DCF_Model.xlsx"
    wb.save(out_file)
    print(f"Saved complete 5-tab model to: {out_file}")
    print(f"{ticker} CMP: ₹{cmp:,.2f} | DCF Fair Value: ₹{fair_value:,.2f} | Margin of Safety: {mos:.1%}")
if __name__ == "__main__":
    run_automated_valuation("https://www.screener.in/company/ITC/consolidated/")

