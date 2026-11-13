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
st.markdown("Analyzes Annual Statements, Quarters, Balance Sheet, Cash Flow, Ratios, Promoter Holding, with YoY, CAGR, Mean, Median, Min, Max, and Multi-Sheet Excel Export.")

url = st.text_input("Screener.in Company URL:", value="https://www.secner.in/company/TCS/consolidated/") # அல்லது சரியான URL கொடுக்கவும்

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
            # Clean columns (remove TTM if present, keep last 10 years)
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
                else:
                    cagr_val = np.nan
            else:
                cagr_val = np.nan
        else:
            mean_val, median_val, min_val, max_val, cagr_val = np