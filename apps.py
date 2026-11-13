import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io
import openpyxl
from openpyxl.styles import Font, PatternFill

st.set_page_config(page_title="10-Year Fundamental Analysis & DCF Model", layout="wide")

st.title("📊 10-Year Fundamental Analysis & DCF Valuation Model")
st.markdown("Enter Screener.in URL to analyze Income Statement, Balance Sheet, Cash Flow, Ratios, Piotroski F-Score, and 10-Year DCF Projections with YoY Growth.")

url = st.text_input("Screener.in Company URL:", value="https://www.screener.in/company/TCS/consolidated/")

COL1, COL2, COL3, COL4 = st.columns(4)
with COL1:
    wacc = st.slider("WACC / Discount Rate", 0.08, 0.18, 0.125, 0.005)
with COL2:
    stage1_g = st.slider("Stage 1 Growth (Y1-Y5)", 0.04, 0.20, 0.08, 0.005)
with COL3:
    stage2_g = st.slider("Stage 2 Growth (Y6-Y10)", 0.03, 0.15, 0.06, 0.005)
with COL4:
    term_g = st.slider("Terminal Growth Rate (g)", 0.02, 0.06, 0.045, 0.005)

@st.cache_data(show_spinner=True)
def fetch_screener_data(screener_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(screener_url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    meta = {}
    for li in soup.select("#top-ratios li"):
        n_el, v_el = li.select_one(".name"), li.select_one(".number")
        if n_el and v_el:
            try:
                meta[n_el.get_text(strip=True)] = float(v_el.get_text(strip=True).replace(",", ""))
            except ValueError:
                meta[n_el.get_text(strip=True)] = v_el.get_text(strip=True)

    tables = {}
    for sec_id in ["profit-loss", "balance-sheet", "cash-flow"]:
        sec = soup.find("section", id=sec_id)
        if not sec: continue
        table = sec.find("table")
        if not table: continue
        cols = [th.get_text(strip=True) for th in table.select("thead th")]
        cols[0] = "Metric"
        rows = []
        for tr in table.select("tbody tr"):
            cells = [td.get_text(strip=True).replace("+", "").strip() for td in tr.find_all("td")]
            if not cells or len(cells) != len(cols): continue
            parsed = [cells[0]]
            for val in cells[1:]:
                clean = val.replace(",", "").replace("%", "")
                try:
                    num = float(clean)
                    parsed.append(num / 100.0 if "%" in val else num)
                except ValueError:
                    parsed.append(0.0)
            rows.append(parsed)
        df = pd.DataFrame(rows, columns=cols).set_index("Metric")
        fy_cols = [c for c in df.columns if c.upper() != "TTM"][-10:]
        tables[sec_id] = df[fy_cols]

    return meta, tables

def compute_yoy_dataframe(data_arrays, index_labels, years):
    rows = []
    index_with_yoy = []
    for label, arr in zip(index_labels, data_arrays):
        rows.append(arr)
        index_with_yoy.append(label)
        yoy = []
        for i in range(len(arr)):
            if i == 0 or np.isnan(arr[i-1]) or arr[i-1] == 0:
                yoy.append(np.nan)
            else:
                yoy.append((arr[i] - arr[i-1]) / abs(arr[i-1]) * 100.0)
        rows.append(yoy)
        index_with_yoy.append(f"{label} (YoY %)")
    return pd.DataFrame(rows, columns=years, index=index_with_yoy)

if st.button("🚀 Run Comprehensive Valuation Model", type="primary"):
    try:
        meta, tables = fetch_screener_data(url)
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
            return np.full(n, default, dtype=float)

        sales = get_row(pnl, ["Sales", "Revenue"])
        raw_mat = get_row(pnl, ["Raw Material"])
        power = get_row(pnl, ["Power"])
        employee = get_row(pnl, ["Employee"])
        other_mfr = get_row(pnl, ["Other Mfr"])
        cogs_items = raw_mat + power + employee + other_mfr
        
        selling = get_row(pnl, ["Selling"])
        other_exp = get_row(pnl, ["Other Expenses"])
        sga_items = selling + other_exp
        
        op_profit = get_row(pnl, ["Operating Profit", "Financing Profit"])
        other_inc = get_row(pnl, ["Other Income"])
        depr = get_row(pnl, ["Depreciation"])
        interest = get_row(pnl, ["Interest"])
        pbt = get_row(pnl, ["Profit before tax"])
        tax_pct = get_row(pnl, ["Tax %"])
        pat = get_row(pnl, ["Net Profit"])

        eff_tax = np.where(tax_pct > 1, tax_pct / 100.0, tax_pct)
        nopat = op_profit * (1.0 - eff_tax)

        eq_cap = get_row(bs, ["Equity Capital"])
        reserves = get_row(bs, ["Reserves"])
        total_equity = eq_cap + reserves
        total_debt = get_row(bs, ["Borrowings"])
        other_liab = get_row(bs, ["Other Liabilities"])
        total_assets = get_row(bs, ["Total Assets"])
        fix_assets = get_row(bs, ["Fixed Assets", "Net Block"])
        cwip = get_row(bs, ["CWIP"])
        investments = get_row(bs, ["Investments"])
        cash_bank = get_row(bs, ["Cash & Bank"])
        cash_investments = cash_bank + investments
        invested_cap = total_equity + total_debt - cash_investments

        cfo = get_row(cf, ["Cash from Operating Activity"])
        capex = np.zeros(n)
        capex[0] = depr[0]
        for i in range(1, n):
            capex[i] = max((fix_assets[i] + cwip[i]) - (fix_assets[i-1] + cwip[i-1]) + depr[i], 0.0)
        fcff = cfo - capex

        pe_ratio = np.divide(meta.get("Market Cap", 0.0), pat, out=np.zeros(n), where=pat!=0)
        pb_ratio = np.divide(meta.get("Market Cap", 0.0), total_equity, out=np.zeros(n), where=total_equity!=0)
        op_margin = np.divide(op_profit, sales, out=np.zeros(n), where=sales!=0)
        net_margin = np.divide(pat, sales, out=np.zeros(n), where=sales!=0)
        roe = np.divide(pat, total_equity, out=np.zeros(n), where=total_equity!=0)
        roce = np.divide(op_profit, (total_assets - other_liab), out=np.zeros(n), where=(total_assets-other_liab)!=0)
        roic = np.divide(nopat, invested_cap, out=np.zeros(n), where=invested_cap!=0)
        accrual_ratio = np.divide((pat - cfo), total_assets, out=np.zeros(n), where=total_assets!=0)

        f_scores = []
        for i in range(1, n):
            score = (
                int(pat[i] > 0) + int(cfo[i] > 0) +
                int((pat[i]/total_assets[i]) > (pat[i-1]/total_assets[i-1])) +
                int(cfo[i]/total_assets[i] > pat[i]/total_assets[i]) +
                int(total_debt[i]/total_assets[i] < total_debt[i-1]/total_assets[i-1]) +
                int(total_equity[i]/total_assets[i] > total_equity[i-1]/total_assets[i-1]) +
                int(sales[i]/fix_assets[i] > sales[i-1]/fix_assets[i-1]) +
                int(op_profit[i]/sales[i] > op_profit[i-1]/sales[i-1])
            )
            f_scores.append(score)
        f_scores = [f_scores[0]] + f_scores

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
        net_debt = total_debt[-1] - cash_investments[-1]
        equity_val = enterprise_val - net_debt

        cmp = float(meta.get("Current Price", 0.0))
        mcap = float(meta.get("Market Cap", 0.0))
        shares_cr = (mcap / cmp) if (cmp > 0 and mcap > 0) else (total_equity[-1] / 10.0)
        fair_value = equity_val / shares_cr if shares_cr > 0 else 0.0
        mos = ((fair_value - cmp) / fair_value) if fair_value > 0 else 0.0

        st.success("✅ Comprehensive Model Generated Successfully!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price (CMP)", f"₹{cmp:,.2f}")
        c2.metric("DCF Fair Value", f"₹{fair_value:,.2f}")
        c3.metric("Margin of Safety", f"{mos:+.1%}")
        c4.metric("10-Yr Cum FCFF", f"₹{np.sum(fcff):,.0f} Cr")

        pnl_df = compute_yoy_dataframe([sales, cogs_items, sga_items, op_profit, other_inc, depr, interest, pbt, eff_tax*100, nopat, pat], ["Revenue", "COGS", "SG&A", "Operating EBITDA", "Other Income", "Depreciation", "Interest", "PBT", "Tax Rate (%)", "NOPAT", "Net Income"], years)
        bs_df = compute_yoy_dataframe([total_debt, total_equity, total_assets, cash_investments, invested_cap], ["Total Debt", "Total Equity", "Total Assets", "Cash & Investments", "Invested Capital"], years)
        cf_df = compute_yoy_dataframe([cfo, capex, fcff], ["CFO", "CapEx", "Free Cash Flow (FCFF)"], years)

        tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios & Forensic", "10-Yr DCF Projections"])
        with tabs[0]:
            st.dataframe(pnl_df.style.format(lambda v: f"{v:,.2f}" if not np.isnan(v) else "N/A"))
        with tabs[1]:
            st.dataframe(bs_df.style.format(lambda v: f"{v:,.2f}" if not np.isnan(v) else "N/A"))
        with tabs[2]:
            st.dataframe(cf_df.style.format(lambda v: f"{v:,.2f}" if not np.isnan(v) else "N/A"))
        with tabs[3]:
            st.dataframe(pd.DataFrame([pe_ratio, pb_ratio, op_margin*100, roe*100, roce*100, roic*100, accrual_ratio, f_scores], index=["P/E Ratio", "P/B Ratio", "Operating Margin (%)", "ROE (%)", "ROCE (%)", "ROIC (%)", "Accrual Ratio", "Piotroski F-Score"], columns=years).style.format("{:,.2f}"))
        with tabs[4]:
            proj_years = [f"FY{int(years[-1][-2:]) + i}" for i in range(1, 11)]
            st.dataframe(pd.DataFrame([proj_fcff, pv_fcff], index=["Projected FCFF", "Present Value"], columns=proj_years).style.format("{:,.2f}"))

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
            ws1.cell(3, c).font = font_hdr; ws1.cell(3, c).fill = fill_hdr

        summary_rows = [["CMP", cmp], ["Fair Value", fair_value], ["Margin of Safety (%)", mos], ["Enterprise Value (₹ Cr)", enterprise_val], ["Equity Value (₹ Cr)", equity_val]]
        for r in summary_rows: ws1.append(r)

        excel_buffer = io.BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)
        st.download_button("📥 Download Excel Model", data=excel_buffer, file_name="Fundamental_Analysis_DCF_Model.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error processing URL: {e}")