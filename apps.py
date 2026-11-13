import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

st.set_page_config(page_title="Comprehensive 10-Year Financial Analysis Model", layout="wide")

st.title("📊 Comprehensive Financial Analysis, DCF & Promoter Model")
st.markdown("Analyzes Annual Statements, Quarters, Balance Sheet, Cash Flow, Ratios, Promoter Holding, with YoY, CAGR, Mean, Median, Min, Max, and Multi-Sheet Excel Export.")

url = st.text_input("Screener.in Company URL:", value="https://www.screener.in/company/TCS/consolidated/")

COL1, COL2, COL3, COL4 = st.columns(4)
with COL1:
    wacc = st.slider("WACC / Discount Rate", 0.01, 0.40, 0.125, 0.005)
with COL2:
    stage1_g = st.slider("Stage 1 Growth (Y1-Y5)", -0.10, 0.50, 0.08, 0.005)
with COL3:
    stage2_g = st.slider("Stage 2 Growth (Y6-Y10)", -0.10, 0.40, 0.06, 0.005)
with COL4:
    term_g = st.slider("Terminal Growth Rate (g)", 0.00, 0.15, 0.045, 0.005)

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
    for sec_id in ["profit-loss", "balance-sheet", "cash-flow", "quarters", "shareholding"]:
        sec = soup.find("section", id=sec_id)
        if not sec: continue
        table = sec.find("table")
        if not table: continue
        
        headers_list = [th.get_text(strip=True) for th in table.select("thead th")]
        if not headers_list: continue
        headers_list[0] = "Metric"
        
        rows = []
        for tr in table.select("tbody tr"):
            cells = [td.get_text(strip=True).replace("+", "").strip() for td in tr.find_all("td")]
            if not cells or len(cells) < 2: continue
            
            metric_name = cells[0]
            values = []
            for val in cells[1:]:
                clean = val.replace(",", "").replace("%", "")
                try:
                    num = float(clean)
                    values.append(num / 100.0 if "%" in val else num)
                except ValueError:
                    values.append(0.0)
            
            if len(values) == len(headers_list) - 1:
                rows.append([metric_name] + values)
                
        if rows:
            temp_df = pd.DataFrame(rows, columns=headers_list).set_index("Metric")
            valid_cols = [c for c in temp_df.columns if c.upper() != "TTM"]
            tables[sec_id] = temp_df[valid_cols[-10:]]

    return meta, tables

def compute_comprehensive_metrics(df):
    new_rows = {}
    for idx, row in df.iterrows():
        numeric_row = pd.to_numeric(row, errors='coerce')
        clean_row = numeric_row.dropna()
        new_rows[idx] = row.tolist()
        
        yoy_vals = [np.nan]
        for i in range(1, len(row)):
            val_prev = float(numeric_row.iloc[i-1])
            val_curr = float(numeric_row.iloc[i])
            if val_prev != 0 and not np.isnan(val_prev) and not np.isnan(val_curr):
                yoy_vals.append((val_curr - val_prev) / abs(val_prev) * 100.0)
            else:
                yoy_vals.append(np.nan)
        new_rows[f"{idx} (YoY %)"] = yoy_vals

        if not clean_row.empty and len(clean_row) > 0:
            mean_val = clean_row.mean()
            median_val = clean_row.median()
            min_val = clean_row.min()
            max_val = clean_row.max()
            
            non_zero_clean = clean_row[clean_row != 0]
            if not non_zero_clean.empty:
                first_val = non_zero_clean.iloc[0]
                last_val = non_zero_clean.iloc[-1]
                n_periods = len(non_zero_clean) - 1
                
                if first_val > 0 and last_val > 0 and n_periods > 0:
                    cagr_val = ((last_val / first_val) ** (1.0 / n_periods) - 1.0) * 100.0
                elif first_val < 0 and last_val > 0:
                    cagr_val = 100.0
                elif first_val == 0:
                    cagr_val = np.nan
                else:
                    cagr_val = -100.0 if last_val == 0 else np.nan
            else:
                cagr_val = np.nan
        else:
            mean_val, median_val, min_val, max_val, cagr_val = np.nan, np.nan, np.nan, np.nan, np.nan

        for metric_name, metric_val in [("Mean", mean_val), ("Median", median_val), ("CAGR (%)", cagr_val), ("Min", min_val), ("Max", max_val)]:
            new_rows[f"{idx} ({metric_name})"] = [metric_val] * len(row)

    return pd.DataFrame(new_rows, columns=df.columns).T

def style_financial_table(df):
    def highlight_rows(x):
        df_styled = pd.DataFrame('', index=x.index, columns=x.columns)
        for idx in x.index:
            if "YoY %" in idx:
                df_styled.loc[idx] = 'background-color: #FFF2CC; font-weight: bold;'
            elif any(m in idx for m in ["Mean", "Median", "CAGR", "Min", "Max"]):
                df_styled.loc[idx] = 'background-color: #E2EFDA; font-style: italic;'
        return df_styled
    return df.style.apply(highlight_rows, axis=None).format(lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v))

if st.button("🚀 Run Full Valuation Model", type="primary"):
    try:
        meta, tables = fetch_screener_data(url)
        pnl = tables.get("profit-loss", pd.DataFrame())
        bs = tables.get("balance-sheet", pd.DataFrame())
        cf = tables.get("cash-flow", pd.DataFrame())
        quarters = tables.get("quarters", pd.DataFrame())
        shareholding = tables.get("shareholding", pd.DataFrame())
        
        if pnl.empty:
            st.error("Could not fetch data. Please check the URL or Screener structure.")
        else:
            years = list(pnl.columns)
            n = len(years)

            def get_row_safe(df, candidates, default=0.0):
                if df.empty: return np.full(n, default, dtype=float)
                for label in candidates:
                    for idx in df.index:
                        if label.lower() in idx.lower():
                            return np.array(df.loc[idx].astype(float).values, dtype=float)
                return np.full(n, default, dtype=float)

            sales = get_row_safe(pnl, ["Sales", "Revenue"])
            expenses = get_row_safe(pnl, ["Expenses"])
            op_profit = get_row_safe(pnl, ["Operating Profit", "Financing Profit"])
            other_inc = get_row_safe(pnl, ["Other Income"])
            depr = get_row_safe(pnl, ["Depreciation"])
            interest = get_row_safe(pnl, ["Interest"])
            pbt = get_row_safe(pnl, ["Profit before tax"])
            tax_pct = get_row_safe(pnl, ["Tax %"], default=25.0)
            pat = get_row_safe(pnl, ["Net Profit"])
            eps = get_row_safe(pnl, ["Basic EPS", "EPS"])

            eff_tax = np.where(tax_pct > 1, tax_pct / 100.0, tax_pct)
            nopat = op_profit * (1.0 - eff_tax)

            eq_cap = get_row_safe(bs, ["Equity Capital"])
            reserves = get_row_safe(bs, ["Reserves"])
            total_equity = eq_cap + reserves
            total_debt = get_row_safe(bs, ["Borrowings", "Debt"])
            other_liab = get_row_safe(bs, ["Other Liabilities"])
            total_assets = get_row_safe(bs, ["Total Assets"])
            fix_assets = get_row_safe(bs, ["Fixed Assets", "Net Block"])
            cwip = get_row_safe(bs, ["CWIP"])
            investments = get_row_safe(bs, ["Investments"])
            cash_bank = get_row_safe(bs, ["Cash & Bank"])
            cash_investments = cash_bank + investments
            invested_cap = total_equity + total_debt - cash_investments

            cfo = get_row_safe(cf, ["Cash from Operating Activity"])
            capex = np.zeros(n)
            if len(depr) > 0: capex[0] = depr[0]
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
                    int(total_equity[i]/total_assets[i] > total_equity[i-1]/total_assets[i-1