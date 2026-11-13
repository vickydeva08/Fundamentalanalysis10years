import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

st.set_page_config(page_title="Advanced 10-Year Fundamental Analysis Model", layout="wide")

st.title("📊 Advanced Financial Analysis & Valuation Model")
st.markdown("Analyzes Annual Statements, Quarters, Balance Sheet, Cash Flow, Ratios, Promoter Holding, with Mean, Median, CAGR, Min, Max, and YoY columns.")

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
                        parsed.append(num if num > 1 else num * 100.0)
                    except ValueError:
                        parsed.append(0.0)
                sh_rows.append(parsed)
            sh_df = pd.DataFrame(sh_rows, columns=sh_cols).set_index("Metric")
            tables["shareholding"] = sh_df

    return meta, tables

def compute_financial_metrics(df):
    new_rows = []
    new_index = []
    
    stat_cols = ["Mean", "Median", "CAGR (%)", "Min", "Max"]
    base_columns = list(df.columns)
    full_columns = base_columns + stat_cols
    
    for idx, row in df.iterrows():
        numeric_row = pd.to_numeric(row, errors='coerce')
        clean_row = numeric_row.dropna()
        
        mean_val, median_val, min_val, max_val, cagr_val = np.nan, np.nan, np.nan, np.nan, np.nan
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
        
        main_vals = row.tolist() + [mean_val, median_val, cagr_val, min_val, max_val]
        new_rows.append(main_vals)
        new_index.append(idx)
        
        yoy_vals = [np.nan]
        for i in range(1, len(row)):
            val_prev = float(numeric_row.iloc[i-1])
            val_curr = float(numeric_row.iloc[i])
            if val_prev != 0 and not np.isnan(val_prev) and not np.isnan(val_curr):
                yoy_vals.append((val_curr - val_prev) / abs(val_prev) * 100.0)
            else:
                yoy_vals.append(np.nan)
        
        yoy_numeric = pd.Series(yoy_vals, index=base_columns)
        yoy_clean = yoy_numeric.dropna()
        yoy_mean, yoy_median, yoy_min, yoy_max, yoy_cagr = np.nan, np.nan, np.nan, np.nan, np.nan
        if not yoy_clean.empty:
            yoy_mean = yoy_clean.mean()
            yoy_median = yoy_clean.median()
            yoy_min = yoy_clean.min()
            yoy_max = yoy_clean.max()
            
        yoy_full_vals = yoy_vals + [yoy_mean, yoy_median, yoy_cagr, yoy_min, yoy_max]
        new_rows.append(yoy_full_vals)
        new_index.append(f"{idx} (YoY %)")
        
    return pd.DataFrame(new_rows, index=new_index, columns=full_columns)

def style_financial_table(df):
    def highlight_rows(x):
        df_styled = pd.DataFrame('', index=x.index, columns=x.columns)
        for idx in x.index:
            if "YoY %" in idx:
                df_styled.loc[idx] = 'background-color: #FFF2CC; font-weight: bold;'
        return df_styled
    return df.style.apply(highlight_rows, axis=None).format(lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v))

if st.button("🚀 Run Advanced Valuation Model", type="primary"):
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

        wacc_val = wacc
        stage1_g_val = stage1_g
        stage2_g_val = stage2_g
        term_g_val = term_g

        base_fcff = fcff[-1]
        proj_fcff = []
        curr = base_fcff
        for yr in range(1, 11):
            g = stage1_g_val if yr <= 5 else stage2_g_val
            curr *= (1.0 + g)
            proj_fcff.append(curr)

        pv_fcff = [f / ((1.0 + wacc_val) ** idx) for idx, f in enumerate(proj_fcff, 1)]
        sum_pv_fcf = sum(pv_fcff)
        term_val = proj_fcff[-1] * (1.0 + term_g_val) / (wacc_val - term_g_val)
        pv_tv = term_val / ((1.0 + wacc_val) ** 10)
        enterprise_val = sum_pv_fcf + pv_tv
        net_debt = total_debt[-1] - cash_investments[-1]
        equity_val = enterprise_val - net_debt

        cmp = float(meta.get("Current Price", 0.0))
        mcap = float(meta.get("Market Cap", 0.0))
        shares_cr = (mcap / cmp) if (cmp > 0 and mcap > 0) else (total_equity[-1] / 10.0)
        fair_value = equity_val / shares_cr if shares_cr > 0 else 0.0
        mos = ((fair_value - cmp) / fair_value) if fair_value > 0 else 0.0

        st.success("✅ Advanced Model Generated Successfully!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price (CMP)", f"₹{cmp:,.2f}")
        c2.metric("DCF Fair Value", f"₹{fair_value:,.2f}")
        c3.metric("Margin of Safety", f"{mos:+.1%}")
        c4.metric("10-Yr Cum FCFF", f"₹{np.sum(fcff):,.0f} Cr")

        pnl_base = pd.DataFrame([sales, expenses, op_profit, other_inc, depr, interest, pbt, pat], 
                                index=["Revenue", "Expenses", "Operating EBITDA", "Other Income", "Depreciation", "Interest", "PBT", "Net Profit"], columns=years)
        bs_base = pd.DataFrame([total_debt, total_equity, total_assets, cash_investments], 
                               index=["Total Debt", "Total Equity", "Total Assets", "Cash & Investments"], columns=years)
        cf_base = pd.DataFrame([cfo, capex, fcff], 
                               index=["CFO", "CapEx", "Free Cash Flow (FCFF)"], columns=years)

        pnl_expanded = compute_financial_metrics(pnl_base)
        bs_expanded = compute_financial_metrics(bs_base)
        cf_expanded = compute_financial_metrics(cf_base)
        art_expanded = compute_financial_metrics(quarters) if not quarters.empty else pd.DataFrame()

        ratios_df = pd.DataFrame(
            [pe_ratio, pb_ratio, debt_to_equity, op_margin*100, roe*100, roce*100, roic*100, accrual_ratio, f_scores], 
            index=["P/E Ratio", "P/B Ratio", "Debt-to-Equity", "Operating Margin (%)", "ROE (%)", "ROCE (%)", "ROIC (%)", "Accrual Ratio", "Piotroski F-Score"], 
            columns=years
        )
        
        proj_years = [f"FY{int(years[-1][-2:]) + i}" for i in range(1, 11)]
        dcf_df = pd.DataFrame([proj_fcff, pv_fcff], index=["Projected FCFF", "Present Value"], columns=proj_years)

        tabs = st.tabs(["Income Statement", "Quarterly Results", "Balance Sheet", "Cash Flow", "Promoter Holding", "Ratios & Forensic", "10-Yr DCF Projections"])
        with tabs[0]:
            st.dataframe(style_financial_table(pnl_expanded))
        with tabs[1]:
            if not art_expanded.empty:
                st.dataframe(style_financial_table(art_expanded))
            else:
                st.info("Quarterly data not available.")
        with tabs[2]:
            st.dataframe(style_financial_table(bs_expanded))
        with tabs[3]:
            st.dataframe(style_financial_table(cf_expanded))
        with tabs[4]:
            st.subheader("👥 Shareholding / Promoter Holding Pattern (%)")
            if not shareholding.empty:
                st.dataframe(shareholding.astype(float).style.format("{:,.2f}%"))
            else:
                st.info("Shareholding data not available.")
        with tabs[5]:
            st.dataframe(ratios_df.style.format("{:,.2f}"))
        with tabs[6]:
            st.dataframe(dcf_df.style.format("{:,.2f}"))

        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        navy = "1B365D"
        font_hdr = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        fill_hdr = PatternFill(start_color=navy, end_color=navy, fill_type="solid")
        fill_growth = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        font_growth = Font(name="Calibri", size=11, bold=True)
        align_center = Alignment(horizontal="center", vertical="center")

        def write_expanded_df_to_sheet(sheet, df, title):
            sheet.append([title])
            sheet.append([])
            headers = ["Metric"] + list(df.columns)
            sheet.append(headers)
            for c in range(1, len(headers) + 1):
                cell = sheet.cell(3, c)
                cell.font = font_hdr; cell.fill = fill_hdr; cell.alignment = align_center
            
            for idx, row in df.iterrows():
                row_data = [str(idx)] + [float(val) if isinstance(val, (int, float)) and not np.isnan(val) else str(val) for val in row.values]
                sheet.append(row_data)
                row_num = sheet.max_row
                if "YoY %" in str(idx):
                    for col_num in range(1, len(row_data) + 1):
                        cell = sheet.cell(row_num, col_num)
                        cell.fill = fill_growth; cell.font = font_growth

        sheets_info = [
            ("P&L Statement", pnl_expanded),
            ("Quarterly Results", art_expanded),
            ("Balance Sheet", bs_expanded),
            ("Cash Flow", cf_expanded)
        ]

        for title, df in sheets_info:
            if not df.empty:
                ws = wb.create_sheet(title=title[:31])
                write_expanded_df_to_sheet(ws, df, title)

        excel_buffer = io.BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)
        st.download_button("📥 Download Advanced Excel Model", data=excel_buffer, file_name="Advanced_Financial_Analysis_Model.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error processing URL: {e}")