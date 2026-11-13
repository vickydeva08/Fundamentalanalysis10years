import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

st.set_page_config(page_title="10-Year Fundamental & Promoter Analysis Model", layout="wide")

st.title("📊 Fundamental Analysis, DCF & Promoter Holding Model")
st.markdown("Enter Screener.in URL to analyze Annual Statements, Quarters, Balance Sheet, Cash Flow, Ratios, Promoter Holding Pattern (%), and 10-Year DCF Projections.")

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

    q_sec = soup.find("section", id="quarters")
    if q_sec:
        q_table = q_sec.find("table")
        if q_table:
            q_cols = [th.get_text(strip=True) for th in q_table.select("thead th")]
            q_cols[0] = "Metric"
            q_rows = []
            for tr in q_table.select("tbody tr"):
                cells = [td.get_text(strip=True).replace("+", "").strip() for td in tr.find_all("td")]
                if not cells or len(cells) != len(q_cols): continue
                parsed = [cells[0]]
                for val in cells[1:]:
                    clean = val.replace(",", "").replace("%", "")
                    try:
                        num = float(clean)
                        parsed.append(num / 100.0 if "%" in val else num)
                    except ValueError:
                        parsed.append(0.0)
                q_rows.append(parsed)
            q_df = pd.DataFrame(q_rows, columns=q_cols).set_index("Metric")
            q_cols_list = [c for c in q_df.columns if c.upper() != "TTM"][-10:]
            tables["quarters"] = q_df[q_cols_list]

    sh_sec = soup.find("section", id="shareholding")
    if sh_sec:
        sh_table = sh_sec.find("table")
        if sh_table:
            sh_cols = [th.get_text(strip=True) for th in sh_table.select("thead th")]
            sh_cols[0] = "Metric"
            sh_rows = []
            for tr in sh_table.select("tbody tr"):
                cells = [td.get_text(strip=True).replace("+", "").strip() for td in tr.find_all("td")]
                if not cells or len(cells) != len(sh_cols): continue
                parsed = [cells[0]]
                for val in cells[1:]:
                    clean = val.replace(",", "").replace("%", "")
                    try:
                        num = float(clean)
                        # Ensure shareholding is correctly scaled as percentage
                        parsed.append(num if num > 1 else num * 100.0)
                    except ValueError:
                        parsed.append(0.0)
                sh_rows.append(parsed)
            sh_df = pd.DataFrame(sh_rows, columns=sh_cols).set_index("Metric")
            tables["shareholding"] = sh_df

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
        index_with_yoy.append(f"{label} (YoY/QoQ %)")
    return pd.DataFrame(rows, columns=years, index=index_with_yoy)

if st.button("🚀 Run Valuation Model", type="primary"):
    try:
        meta, tables = fetch_screener_data(url)
        pnl = tables["profit-loss"]
        bs = tables["balance-sheet"]
        cf = tables["cash-flow"]
        quarters = tables.get("quarters", pd.DataFrame())
        shareholding = tables.get("shareholding", pd.DataFrame())
        
        years = list(pnl.columns)
        n = len(years)

        def get_row(df, candidates, default=0.0):
            for label in candidates:
                for idx in df.index:
                    if label.lower() in idx.lower():
                        return np.array(df.loc[idx].astype(float).values, dtype=float)
            return np.full(df.shape[1] if not df.empty else n, default, dtype=float)

        sales = get_row(pnl, ["Sales", "Revenue"])
        expenses = get_row(pnl, ["Expenses"])
        op_profit = get_row(pnl, ["Operating Profit", "Financing Profit"])
        other_inc = get_row(pnl, ["Other Income"])
        depr = get_row(pnl, ["Depreciation"])
        interest = get_row(pnl, ["Interest"])
        pbt = get_row(pnl, ["Profit before tax"])
        tax_pct = get_row(pnl, ["Tax %"])
        pat = get_row(pnl, ["Net Profit"])
        eps = get_row(pnl, ["Basic EPS", "EPS"])

        eff_tax = np.where(tax_pct > 1, tax_pct / 100.0, tax_pct)
        nopat = op_profit * (1.0 - eff_tax)

        eq_cap = get_row(bs, ["Equity Capital"])
        reserves = get_row(bs, ["Reserves"])
        total_equity = eq_cap + reserves
        total_debt = get_row(bs, ["Borrowings", "Debt"])
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

        mcap_val = float(meta.get("Market Cap", 0.0))
        pe_ratio = np.where(pat != 0, mcap_val / pat, 0.0)
        pb_ratio = np.where(total_equity != 0, mcap_val / total_equity, 0.0)
        debt_to_equity = np.where(total_equity != 0, total_debt / total_equity, 0.0)
        op_margin = np.where(sales != 0, op_profit / sales, 0.0)
        roe = np.where(total_equity != 0, pat / total_equity, 0.0)
        roce = np.where((total_assets - other_liab) != 0, op_profit / (total_assets - other_liab), 0.0)
        roic = np.where(invested_cap != 0, nopat / invested_cap, 0.0)
        accrual_ratio = np.where(total_assets != 0, (pat - cfo) / total_assets, 0.0)

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
        mos = ((fair_value - cmp) / fair_value