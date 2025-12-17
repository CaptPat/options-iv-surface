import argparse
import sys
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import webbrowser

# --- CONFIGURATION ---
RISK_FREE_RATE = 0.04
VERSION = "0.1.0"

# --- GUI IMPORTS ---
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# --- MATH FUNCTIONS ---

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    try:
        if market_price <= 0 or np.isnan(market_price): return np.nan
        func = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(func, 0.001, 5.0)
    except:
        return np.nan

# --- METADATA & DATA PROCESSING ---

def extract_file_metadata(filepath):
    """
    Reads the first 2 lines of the Fidelity CSV to extract:
    1. Spot Price (Line 1)
    2. Spot Time (Line 1)
    3. Data Time (Line 2)
    """
    meta = {
        'spot': None,
        'spot_time_str': 'Unknown',
        'data_time_str': 'Unknown',
        'analysis_date': None
    }
    
    try:
        with open(filepath, 'r') as f:
            line1 = f.readline().strip() # Spot Info
            line2 = f.readline().strip() # Data Time

        # PARSE LINE 1: Spot line (support minor wording variations)
        # Examples:
        #  - "NVDA ... Last Trade $181.98 as of 12/5/2025 3:36:10 PM"
        #  - "NVDA ... Last Price $181.98 as of ..."
        price_match = (
            re.search(r'Last Trade \$(\d+\.\d+)', line1) or
            re.search(r'Last Price \$(\d+\.\d+)', line1)
        )
        if price_match:
            meta['spot'] = float(price_match.group(1))

        time_match = re.search(r'as of (.+)$', line1)
        if time_match:
            meta['spot_time_str'] = time_match.group(1).strip()

        # PARSE LINE 2: "12/05/2025 03:36:29 PM ET"
        if line2:
            meta['data_time_str'] = line2.strip()
            # Parse date object for calculation (strip timezone " ET" if present)
            clean_date_str = line2.replace(" ET", "").strip()
            try:
                meta['analysis_date'] = datetime.strptime(clean_date_str, '%m/%d/%Y %I:%M:%S %p')
            except ValueError:
                meta['analysis_date'] = datetime.now()
    except Exception as e:
        print(f"Warning: Could not parse header metadata from {filepath}: {e}")

    return meta

def _find_column(columns, must_have=None, any_of=None, exclude=None):
    """Utility to find a column by fuzzy matching tokens in header names.
    - columns: iterable of column names
    - must_have: list of substrings that must all appear (case-insensitive)
    - any_of: list of substrings; at least one must appear
    - exclude: list of substrings that must NOT appear
    Returns the first matching column name or None.
    """
    must_have = [s.lower() for s in (must_have or [])]
    any_of = [s.lower() for s in (any_of or [])]
    exclude = [s.lower() for s in (exclude or [])]
    for col in columns:
        c = str(col).strip().lower()
        if any(ex in c for ex in exclude):
            continue
        if any_of and not any(tok in c for tok in any_of):
            continue
        if must_have and not all(tok in c for tok in must_have):
            continue
        return col
    return None

def _to_numeric(series):
    """Convert a pandas Series to numeric, stripping common formatting.
    Handles dashes, commas, currency symbols. Returns float dtype series."""
    # Replace common non-numeric markers
    s = series.astype(str).str.replace('[,$]', '', regex=True)
    s = s.replace({'—': np.nan, '--': np.nan, '': np.nan})
    return pd.to_numeric(s, errors='coerce')

def schema_report(filepath):
    """Print detailed schema mapping and sample rows."""
    try:
        df = pd.read_csv(filepath, skiprows=2)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    df = df.dropna(how='all')
    cols = list(df.columns)

    exp_col = _find_column(cols, any_of=['expir', 'expiry'])
    strike_col = _find_column(cols, any_of=['strike'])
    call_bid_col = _find_column(cols, must_have=['call', 'bid'])
    call_ask_col = _find_column(cols, must_have=['call', 'ask'])
    put_bid_col = _find_column(cols, must_have=['put', 'bid'])
    put_ask_col = _find_column(cols, must_have=['put', 'ask'])

    # Fallback
    if not call_bid_col or not put_bid_col:
        bid_cols = [c for c in cols if 'bid' in str(c).strip().lower()]
        if len(bid_cols) >= 2:
            if not call_bid_col:
                call_bid_col = bid_cols[0]
            if not put_bid_col:
                put_bid_col = bid_cols[1]
    if not call_ask_col or not put_ask_col:
        ask_cols = [c for c in cols if 'ask' in str(c).strip().lower()]
        if len(ask_cols) >= 2:
            if not call_ask_col:
                call_ask_col = ask_cols[0]
            if not put_ask_col:
                put_ask_col = ask_cols[1]

    print("\nSchema Report")
    print("="*60)
    print(f"File: {os.path.basename(filepath)}")
    print(f"Total rows (after skipping metadata): {len(df)}")
    print(f"Total columns: {len(cols)}")
    print("\nCanonical mapping:")
    print(f"  Expiration Date <- {exp_col}")
    print(f"  Strike          <- {strike_col}")
    print(f"  Call Bid        <- {call_bid_col}")
    print(f"  Call Ask        <- {call_ask_col}")
    print(f"  Put Bid         <- {put_bid_col}")
    print(f"  Put Ask         <- {put_ask_col}")

    # Sample rows for mapped columns
    mapped_cols = [exp_col, strike_col, call_bid_col, call_ask_col, put_bid_col, put_ask_col]
    sample_df = df[[c for c in mapped_cols if c]].head(3)
    print("\nSample rows (first 3):")
    print(sample_df.to_string(index=False))

    # Extra columns
    extras = [c for c in cols if c not in mapped_cols]
    if extras:
        print(f"\nExtra columns (preserved for future use): {len(extras)}")
        print("  " + ", ".join(str(e) for e in extras))

    return True

def validate_csv(filepath, strict=False, threshold=95.0):
    """Validate CSV headers and report detected mappings without running IV analysis."""
    try:
        df = pd.read_csv(filepath, skiprows=2)
    except Exception as e:
        return False, f"Error reading {filepath}: {e}"

    df = df.dropna(how='all')
    cols = list(df.columns)

    exp_col = _find_column(cols, any_of=['expir', 'expiry'])
    strike_col = _find_column(cols, any_of=['strike'])
    call_bid_col = _find_column(cols, must_have=['call', 'bid'])
    call_ask_col = _find_column(cols, must_have=['call', 'ask'])
    put_bid_col = _find_column(cols, must_have=['put', 'bid'])
    put_ask_col = _find_column(cols, must_have=['put', 'ask'])

    # Fallback for Bid/Ask pairs
    if not call_bid_col or not put_bid_col:
        bid_cols = [c for c in cols if 'bid' in str(c).strip().lower()]
        if len(bid_cols) >= 2:
            if not call_bid_col:
                call_bid_col = bid_cols[0]
            if not put_bid_col:
                put_bid_col = bid_cols[1]
    if not call_ask_col or not put_ask_col:
        ask_cols = [c for c in cols if 'ask' in str(c).strip().lower()]
        if len(ask_cols) >= 2:
            if not call_ask_col:
                call_ask_col = ask_cols[0]
            if not put_ask_col:
                put_ask_col = ask_cols[1]

    missing = []
    if not exp_col: missing.append('Expiration Date')
    if not strike_col: missing.append('Strike')
    if not call_bid_col: missing.append('Call Bid')
    if not call_ask_col: missing.append('Call Ask')
    if not put_bid_col: missing.append('Put Bid')
    if not put_ask_col: missing.append('Put Ask')

    print("\nCSV Header Validation Report")
    print("---------------------------------")
    print(f"File: {os.path.basename(filepath)}")
    print("Detected columns:")
    print(f"  Expiration Date -> {exp_col}")
    print(f"  Strike          -> {strike_col}")
    print(f"  Call Bid        -> {call_bid_col}")
    print(f"  Call Ask        -> {call_ask_col}")
    print(f"  Put Bid         -> {put_bid_col}")
    print(f"  Put Ask         -> {put_ask_col}")
    print("Other headers:")
    extras = [c for c in cols if c not in {exp_col, strike_col, call_bid_col, call_ask_col, put_bid_col, put_ask_col}]
    print("  " + ", ".join([str(e) for e in extras]))

    # Numeric conversion stats (only if we have mappings)
    total_rows = len(df)
    def _stat(name, colname):
        if not colname:
            return f"  {name}: (missing)", 0.0
        s = _to_numeric(df[colname])
        valid = int(s.notna().sum())
        pct = (valid/total_rows*100 if total_rows else 0)
        return f"  {name}: valid {valid}/{total_rows} ({pct:.1f}%)", pct

    print("\nNumeric conversion stats:")
    strike_msg, strike_pct = _stat('Strike', strike_col)
    call_bid_msg, call_bid_pct = _stat('Call Bid', call_bid_col)
    call_ask_msg, call_ask_pct = _stat('Call Ask', call_ask_col)
    put_bid_msg, put_bid_pct = _stat('Put Bid', put_bid_col)
    put_ask_msg, put_ask_pct = _stat('Put Ask', put_ask_col)
    
    print(strike_msg)
    print(call_bid_msg)
    print(call_ask_msg)
    print(put_bid_msg)
    print(put_ask_msg)

    # Strict mode: fail if any field below threshold
    if strict:
        min_pct = min(strike_pct, call_bid_pct, call_ask_pct, put_bid_pct, put_ask_pct)
        if min_pct < threshold:
            msg = f"\nSTRICT MODE FAILURE: Minimum validity {min_pct:.1f}% is below threshold {threshold}%"
            print(msg)
            return False, msg

    if missing:
        present = ', '.join([str(c) for c in cols])
        msg = (
            "Missing required columns: " + ", ".join(missing) +
            f".\nFound headers: {present}\n"
            "Tips: Ensure the export includes Strike, Expiration, Call Bid/Ask, Put Bid/Ask."
        )
        print("\n" + msg)
        return False, msg

    return True, "Headers look good."

def process_data(filepath, manual_spot, manual_date, min_dte, max_dte):
    # 1. Extract Metadata from Header
    meta = extract_file_metadata(filepath)

    # 2. Resolve Spot Price (Arg overrides File)
    spot_price = manual_spot if manual_spot is not None else meta['spot']
    if spot_price is None:
        raise ValueError("Could not find Spot Price in CSV header and none provided.")

    # 3. Resolve Analysis Date (Arg overrides File)
    if manual_date:
        if isinstance(manual_date, str):
            analysis_date = datetime.strptime(manual_date, '%Y-%m-%d')
        else:
            analysis_date = manual_date
    else:
        analysis_date = meta['analysis_date'] if meta['analysis_date'] else datetime.now()

    print(f"Processing {filepath}...")
    print(f" > Spot Price: ${spot_price}")
    print(f" > Analysis Date: {analysis_date.strftime('%Y-%m-%d')}")

    # 4. Load CSV Data using header row (skip only first 2 metadata lines)
    try:
        df = pd.read_csv(filepath, skiprows=2)
    except Exception as e:
        raise ValueError(f"Error reading {filepath}: {e}")

    try:
        # Drop entirely blank rows that sometimes appear after headers
        df = df.dropna(how='all')

        # Detect required columns flexibly
        cols = list(df.columns)
        exp_col = _find_column(cols, any_of=['expir', 'expiry'])
        strike_col = _find_column(cols, any_of=['strike'])
        call_bid_col = _find_column(cols, must_have=['call', 'bid'])
        call_ask_col = _find_column(cols, must_have=['call', 'ask'])
        put_bid_col = _find_column(cols, must_have=['put', 'bid'])
        put_ask_col = _find_column(cols, must_have=['put', 'ask'])

        # Fallback: handle Fidelity-style duplicated generic headers (Bid/Ask and Bid.1/Ask.1)
        if not call_bid_col or not put_bid_col:
            bid_cols = [c for c in cols if 'bid' in str(c).strip().lower()]
            if len(bid_cols) >= 2:
                # Assign first occurrence to Call, second to Put if not already mapped
                if not call_bid_col:
                    call_bid_col = bid_cols[0]
                if not put_bid_col:
                    put_bid_col = bid_cols[1]
        if not call_ask_col or not put_ask_col:
            ask_cols = [c for c in cols if 'ask' in str(c).strip().lower()]
            if len(ask_cols) >= 2:
                if not call_ask_col:
                    call_ask_col = ask_cols[0]
                if not put_ask_col:
                    put_ask_col = ask_cols[1]

        missing = []
        if not exp_col: missing.append('Expiration Date (e.g., contains "Expir")')
        if not strike_col: missing.append('Strike')
        if not call_bid_col: missing.append('Call Bid')
        if not call_ask_col: missing.append('Call Ask')
        if not put_bid_col: missing.append('Put Bid')
        if not put_ask_col: missing.append('Put Ask')

        if missing:
            present = ', '.join([str(c) for c in cols])
            raise ValueError(
                "Missing required columns: " + ", ".join(missing) +
                f".\nFound headers: {present}\n"
                "Tips: Ensure the export includes Strike, Expiration, Call Bid/Ask, Put Bid/Ask."
            )

        # Rename to canonical names used downstream
        rename_map = {
            exp_col: 'Expiration Date',
            strike_col: 'Strike',
            call_bid_col: 'Call Bid',
            call_ask_col: 'Call Ask',
            put_bid_col: 'Put Bid',
            put_ask_col: 'Put Ask',
        }
        df = df.rename(columns=rename_map)

        # Keep all extra columns for potential future analysis, but ensure core are numeric
        df['Strike'] = _to_numeric(df['Strike'])
        df['Call Bid'] = _to_numeric(df['Call Bid'])
        df['Call Ask'] = _to_numeric(df['Call Ask'])
        df['Put Bid'] = _to_numeric(df['Put Bid'])
        df['Put Ask'] = _to_numeric(df['Put Ask'])

        # Drop rows without valid strike
        df = df[df['Strike'].notna()]

        # Mid Price
        df['Call Mid'] = (df['Call Bid'] + df['Call Ask']) / 2
        df['Put Mid'] = (df['Put Bid'] + df['Put Ask']) / 2

        # Date parsing (support multiple formats)
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')

        # Calculate DTE
        df['Days to Expiry'] = (df['Expiration Date'] - analysis_date).dt.days
        df['Time to Expiry'] = df['Days to Expiry'] / 365.25

        # Filter
        df = df[df['Days to Expiry'] > 0].copy()
        df = df[(df['Days to Expiry'] >= min_dte) & (df['Days to Expiry'] <= max_dte)]

        if df.empty:
            print(f"Warning: No data found for range {min_dte}-{max_dte} DTE.")
            return df, meta

        # Ensure sorted order within each expiration by Strike (ascending)
        df = df.sort_values(['Days to Expiry', 'Strike']).reset_index(drop=True)

        # Calculate IV
        df['Call IV'] = df.apply(lambda row: implied_volatility(
            row['Call Mid'], spot_price, row['Strike'], row['Time to Expiry'], RISK_FREE_RATE, 'call'), axis=1)
        df['Put IV'] = df.apply(lambda row: implied_volatility(
            row['Put Mid'], spot_price, row['Strike'], row['Time to Expiry'], RISK_FREE_RATE, 'put'), axis=1)

        # Preferred IV selection: Put IV below spot, Call IV at/above spot
        df['IV'] = np.where(df['Strike'] < spot_price, df['Put IV'], df['Call IV'])

        return df, meta

    except Exception as e:
        raise ValueError(f"Data Processing Error: {e}")

# --- CORE LOGIC WRAPPER ---

def run_analysis(input_file, output_file, spot=None, date=None, diff_file=None, 
                 min_dte=1, max_dte=3650, open_in_browser=True, 
                 mode='surface', target_dte_line=30, auto_output=False):
    
    # 1. Process Main
    try:
        df_main, meta_main = process_data(input_file, spot, date, min_dte, max_dte)
    except Exception as e:
        print(f"Error: {e}")
        return False, str(e)

    if df_main.empty:
        return False, "No data found matching criteria."

    # 2. Logic for Diff or Standard
    z_column = 'IV'
    df_plot = df_main
    
    spot_info = f"${meta_main['spot']} as of {meta_main['spot_time_str']}"
    main_title = f"IV Surface as of {meta_main['data_time_str']} (Spot {spot_info})"

    # Only process diff file in surface mode (ignore for line mode)
    if diff_file and mode != 'line':
        print("\n--- Processing Reference File ---")
        try:
            df_ref, meta_ref = process_data(diff_file, None, None, min_dte, max_dte)
            if not df_ref.empty:
                # Check for overlapping expirations
                main_expirations = set(df_main['Expiration Date'].unique())
                ref_expirations = set(df_ref['Expiration Date'].unique())
                overlap = main_expirations.intersection(ref_expirations)
                
                main_only = main_expirations - ref_expirations
                ref_only = ref_expirations - main_expirations
                
                if not overlap:
                    print("WARNING: No overlapping expirations between datasets.")
                    print(f"Main file expirations: {sorted([d.strftime('%Y-%m-%d') for d in main_expirations])}")
                    print(f"Ref file expirations: {sorted([d.strftime('%Y-%m-%d') for d in ref_expirations])}")
                    return False, "No overlapping expirations to compute diff. Files cover different time periods."
                
                print(f"Found {len(overlap)} overlapping expirations. Merging datasets...")
                if main_only:
                    print(f"  Note: {len(main_only)} expiration(s) only in main file (likely expired in ref): {sorted([d.strftime('%m/%d') for d in main_only])}")
                if ref_only:
                    print(f"  Note: {len(ref_only)} expiration(s) only in ref file (new listings): {sorted([d.strftime('%m/%d') for d in ref_only])}")
                df_merge = pd.merge(
                    df_main[['Expiration Date', 'Strike', 'Days to Expiry', 'IV']],
                    df_ref[['Expiration Date', 'Strike', 'IV']],
                    on=['Expiration Date', 'Strike'],
                    suffixes=('', '_ref')
                )
                
                if df_merge.empty:
                    print("WARNING: No matching strike/expiration pairs after merge.")
                    return False, "No matching data points between files after merge."
                
                df_merge['IV_Diff'] = df_merge['IV'] - df_merge['IV_ref']
                # Sort merged dataset by expiration and strike for consistent plotting
                df_merge = df_merge.sort_values(['Days to Expiry', 'Strike']).reset_index(drop=True)
                df_plot = df_merge
                z_column = 'IV_Diff'
                main_title = f"IV Change: {meta_main['data_time_str']} vs {meta_ref['data_time_str']}"
        except Exception as e:
            print(f"Diff File Error: {e}")
            return False, f"Diff processing failed: {e}"

    # 2.5 Auto-adjust output filename if using defaults
    if auto_output:
        if mode == 'line':
            output_file = 'pin_analysis.html'
        elif z_column == 'IV_Diff':
            output_file = 'diff_analysis.html'
        else:
            output_file = 'vol_analysis.html'

    # 3. Filtering Bounds & Mode Selection
    current_spot = meta_main['spot']
    lower_bound = current_spot * 0.5
    upper_bound = current_spot * 1.5
    df_plot = df_plot[(df_plot['Strike'] >= lower_bound) & (df_plot['Strike'] <= upper_bound)].sort_values('Strike')

    # Apply Mode Restriction
    if mode == 'line':
        # Find the unique DTE closest to target_dte_line
        available_dtes = df_plot['Days to Expiry'].unique()
        if len(available_dtes) > 0:
            closest_dte = min(available_dtes, key=lambda x: abs(x - target_dte_line))
            print(f"Line Mode: Filtering to closest DTE {closest_dte} (Target: {target_dte_line})")
            df_plot = df_plot[df_plot['Days to Expiry'] == closest_dte]
            main_title += f" [DTE: {closest_dte}]"

    unique_dtes = df_plot['Days to Expiry'].unique()

    # --- VISUALIZATION ---
    if z_column == 'IV_Diff':
        c_scale = 'RdBu_r'
    else:
        c_scale = 'Viridis'

    if len(unique_dtes) == 1:
        # MODE A: 2D SMILE (Single Expiration)
        dte_val = unique_dtes[0]
        # Sort strictly by Strike for 2D plotting
        df_plot = df_plot.sort_values('Strike')
        
        # Calculate Pin Target (Min IV)
        if not df_plot[z_column].isnull().all():
            min_iv_idx = df_plot[z_column].idxmin()
            min_iv_val = df_plot.loc[min_iv_idx, z_column]
            min_iv_strike = df_plot.loc[min_iv_idx, 'Strike']
            print(f" > PIN TARGET (Min IV): Strike ${min_iv_strike} (IV: {min_iv_val:.4f})")
            pin_text = f"Pin Target: ${min_iv_strike}"
        else:
            min_iv_strike = current_spot
            min_iv_val = 0
            pin_text = "N/A"

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_plot['Strike'], y=df_plot[z_column],
            mode='lines+markers',
            name=f'{dte_val} DTE',
            line=dict(color='firebrick', width=3),
            marker=dict(size=6)
        ))

        fig.add_vline(x=current_spot, line_dash="dash", line_color="green", annotation_text="Spot")

        if pin_text != "N/A":
            fig.add_annotation(
                x=min_iv_strike, y=min_iv_val,
                text=pin_text,
                showarrow=True, arrowhead=1, ax=0, ay=-40,
                bgcolor="yellow", bordercolor="black"
            )

        fig.update_layout(
            title=main_title,
            xaxis_title='Strike Price',
            yaxis_title='Implied Volatility' if z_column=='IV' else 'IV Change',
            width=1000, height=600,
            template='plotly_white'
        )
    
    else:
        # 3D Surface
        iv_matrix = df_plot.pivot_table(index='Strike', columns='Days to Expiry', values=z_column)
        
        # Interpolation
        min_strike, max_strike = int(iv_matrix.index.min()), int(iv_matrix.index.max())
        iv_matrix = iv_matrix.reindex(range(min_strike, max_strike + 1))
        
        min_day, max_day = int(iv_matrix.columns.min()), int(iv_matrix.columns.max())
        iv_matrix = iv_matrix.reindex(columns=range(min_day, max_day + 1))
        
        iv_matrix = iv_matrix.interpolate(method='linear', axis=0, limit_direction='both')
        method_x = 'cubic' if len(unique_dtes) > 3 else 'linear'
        iv_matrix = iv_matrix.interpolate(method=method_x, axis=1, limit_direction='both')

        fig = go.Figure(data=[go.Surface(
            z=iv_matrix.values,
            x=iv_matrix.columns,
            y=iv_matrix.index,
            colorscale=c_scale,
            colorbar_title='IV' if z_column=='IV' else 'Diff',
        )])

        fig.update_layout(
            title=main_title,
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Strike',
                zaxis_title='IV' if z_column=='IV' else 'IV Change'
            ),
            width=1200, height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

    # 4. Save with proper HTML5 metadata
    try:
        # Generate base HTML from Plotly
        fig.write_html(output_file)
        
        # Post-process to add missing HTML5 metadata
        with open(output_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Determine appropriate title based on mode
        if mode == 'line':
            title = f"Pin Analysis - {meta_main['data_time_str']}"
        elif z_column == 'IV_Diff':
            title = f"IV Diff Analysis - {meta_main['data_time_str']}"
        else:
            title = f"Volatility Surface - {meta_main['data_time_str']}"
        
        # Add lang attribute, title, and viewport (charset already exists in Plotly output)
        html_content = html_content.replace(
            '<html>',
            '<html lang="en">'
        )
        html_content = html_content.replace(
            '<head><meta charset="utf-8" />',
            f'<head><meta charset="utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>{title}</title>'
        )
        
        # Write enhanced HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"SUCCESS: Saved to {output_file}")
        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(output_file))
        return True, "Analysis Complete"
    except Exception as e:
        return False, f"Error saving file: {e}"


# --- GUI CLASS ---

class IVSurfaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Options IV Surface Tool")
        
        # --- Frame: Files ---
        file_frame = ttk.LabelFrame(root, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # Input
        ttk.Label(file_frame, text="Input CSV:").grid(row=0, column=0, sticky="w")
        self.entry_input = ttk.Entry(file_frame, width=40)
        self.entry_input.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)

        # Output
        ttk.Label(file_frame, text="Output HTML:").grid(row=1, column=0, sticky="w")
        self.entry_output = ttk.Entry(file_frame, width=40)
        self.entry_output.insert(0, "vol_analysis.html")
        self.entry_output.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Diff
        ttk.Label(file_frame, text="Diff CSV (Opt):").grid(row=2, column=0, sticky="w")
        self.diff_var = tk.StringVar()
        self.diff_var.trace_add('write', self.on_diff_change)
        self.entry_diff = ttk.Entry(file_frame, width=40, textvariable=self.diff_var)
        self.entry_diff.grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_diff).grid(row=2, column=2)

        # --- Frame: Parameters ---
        param_frame = ttk.LabelFrame(root, text="Analysis Parameters", padding="10")
        param_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Dates / DTE
        ttk.Label(param_frame, text="Min DTE:").grid(row=0, column=0, sticky="w")
        self.entry_min_dte = ttk.Entry(param_frame, width=10)
        self.entry_min_dte.insert(0, "1")
        self.entry_min_dte.grid(row=0, column=1, sticky="w")
        
        ttk.Label(param_frame, text="OR Min Date (YYYY-MM-DD):").grid(row=0, column=2, sticky="e")
        self.entry_min_date = ttk.Entry(param_frame, width=15)
        self.entry_min_date.grid(row=0, column=3, padx=5)

        ttk.Label(param_frame, text="Max DTE:").grid(row=1, column=0, sticky="w")
        self.entry_max_dte = ttk.Entry(param_frame, width=10)
        self.entry_max_dte.insert(0, "3650")
        self.entry_max_dte.grid(row=1, column=1, sticky="w")
        
        ttk.Label(param_frame, text="OR Max Date (YYYY-MM-DD):").grid(row=1, column=2, sticky="e")
        self.entry_max_date = ttk.Entry(param_frame, width=15)
        self.entry_max_date.grid(row=1, column=3, padx=5)

        # Scope (Radio)
        ttk.Label(param_frame, text="Mode:").grid(row=2, column=0, sticky="w", pady=10)
        self.mode_var = tk.StringVar(value="surface")
        
        # Sub-frame for radio buttons to keep layout clean
        mode_frame = ttk.Frame(param_frame)
        mode_frame.grid(row=2, column=1, columnspan=3, sticky="w")
        
        r1 = ttk.Radiobutton(mode_frame, text="Surface (All Expirations)", variable=self.mode_var, value="surface", command=self.toggle_mode)
        r1.pack(side="left", padx=5)
        
        r2 = ttk.Radiobutton(mode_frame, text="Line (Single Expiration)", variable=self.mode_var, value="line", command=self.toggle_mode)
        r2.pack(side="left", padx=5)
        
        # Target DTE for Line Mode
        self.lbl_target = ttk.Label(mode_frame, text="Target DTE:")
        self.entry_target = ttk.Entry(mode_frame, width=5)
        self.entry_target.insert(0, "30")
        
        self.toggle_mode() # Set initial state

        # Browser Toggle
        self.browser_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(root, text="Open Result in Browser Automatically", variable=self.browser_var).grid(row=2, column=0, pady=5)

        # --- Actions ---
        btn_frame = ttk.Frame(root, padding="10")
        btn_frame.grid(row=3, column=0)
        
        # Run Calculation
        action_btn = ttk.Button(btn_frame, text="Compute VSM", command=self.run_process, width=15)
        action_btn.pack(side="left", padx=5)

        # Show Existing Result (NEW)
        show_btn = ttk.Button(btn_frame, text="Show Result", command=self.show_result, width=15)
        show_btn.pack(side="left", padx=5)
        
        # Exit
        close_btn = ttk.Button(btn_frame, text="Exit", command=root.quit, width=10)
        close_btn.pack(side="left", padx=5)

    def toggle_mode(self):
        if self.mode_var.get() == 'line':
            self.lbl_target.pack(side="left", padx=(10,2))
            self.entry_target.pack(side="left")
        else:
            self.lbl_target.forget()
            self.entry_target.forget()
        # Unified output filename update based on mode/diff presence
        self.update_output_default()

    def update_output_default(self):
        current = (self.entry_output.get() or '').strip()
        mode = self.mode_var.get()
        diff_present = bool((self.diff_var.get() or '').strip())
        defaults = {'vol_analysis.html', 'diff_analysis.html', 'pin_analysis.html', ''}
        if current in defaults:
            if mode == 'line':
                new_name = 'pin_analysis.html'
            else:
                new_name = 'diff_analysis.html' if diff_present else 'vol_analysis.html'
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, new_name)

    def browse_input(self):
        f = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if f:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, f)

    def browse_output(self):
        f = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML Files", "*.html")])
        if f:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, f)

    def browse_diff(self):
        f = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if f:
            self.diff_var.set(f)
        else:
            # If user cancels and clears, ensure defaults revert
            if not self.entry_diff.get().strip():
                self.diff_var.set('')

    def on_diff_change(self, *args):
        # Update output filename whenever diff field changes
        self.update_output_default()

    def resolve_date_input(self, dte_str, date_str, current_analysis_date, is_max=False):
        """Helper to figure out DTE from either a DTE integer or a Date String"""
        # If specific date string provided, calculate DTE
        if date_str and date_str.strip():
            try:
                target_date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
                delta_days = (target_date - current_analysis_date).days
                return delta_days
            except ValueError:
                messagebox.showerror("Date Error", f"Invalid Date Format: {date_str}. Use YYYY-MM-DD")
                return None
        
        # Otherwise use DTE integer
        if dte_str and dte_str.strip():
            try:
                return int(dte_str)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid DTE Number: {dte_str}")
                return None
        
        return 3650 if is_max else 1

    def run_process(self):
        inp = self.entry_input.get()
        out = self.entry_output.get()
        diff = self.entry_diff.get()
        
        if not inp:
            messagebox.showwarning("Missing Input", "Please select an input CSV file.")
            return

        # We need metadata to resolve dates if user used date fields
        try:
            meta = extract_file_metadata(inp)
            an_date = meta['analysis_date'] if meta['analysis_date'] else datetime.now()
        except:
            an_date = datetime.now()

        # Resolve Min/Max
        min_dte = self.resolve_date_input(self.entry_min_dte.get(), self.entry_min_date.get(), an_date, is_max=False)
        max_dte = self.resolve_date_input(self.entry_max_dte.get(), self.entry_max_date.get(), an_date, is_max=True)
        
        if min_dte is None or max_dte is None: return 

        target_line = 30
        if self.mode_var.get() == 'line':
            try:
                target_line = int(self.entry_target.get())
            except:
                messagebox.showwarning("Input Error", "Invalid Target DTE for Line Mode.")
                return

        # Run
        success, msg = run_analysis(
            input_file=inp,
            output_file=out,
            diff_file=diff if diff else None,
            min_dte=min_dte,
            max_dte=max_dte,
            open_in_browser=self.browser_var.get(),
            mode=self.mode_var.get(),
            target_dte_line=target_line,
            auto_output=False  # GUI manages output filename explicitly
        )

        if success:
            if not self.browser_var.get():
                messagebox.showinfo("Success", "Analysis Generated Successfully (Saved to disk).")
        else:
            messagebox.showerror("Error", msg)

    def show_result(self):
        """Open the current output file in the browser without re-calculating."""
        out_file = self.entry_output.get()
        if os.path.exists(out_file):
            webbrowser.open('file://' + os.path.realpath(out_file))
        else:
            messagebox.showerror("File Not Found", f"The file '{out_file}' does not exist.\nPlease Run Compute VSM first.")


# --- MAIN ENTRY POINT ---

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate IV Surface/Smile from Fidelity Data.')
    parser.add_argument('-i', '--input', type=str, help='Input CSV (Current Data)')
    parser.add_argument('-o', '--output', type=str, default='vol_analysis.html', help='Output HTML filename')
    parser.add_argument('-s', '--spot', type=float, default=None, help='Override Spot Price')
    parser.add_argument('--date', type=str, default=None, help='Override Analysis Date YYYY-MM-DD')
    parser.add_argument('--diff', type=str, default=None, help='Reference CSV for Diff Chart')
    parser.add_argument('--min-dte', type=int, default=1, help='Min Days to Expiration')
    parser.add_argument('--max-dte', type=int, default=3650, help='Max Days to Expiration')
    parser.add_argument('--mode', type=str, default='surface', choices=['surface', 'line'], help='Analysis mode: surface (3D/multi-expiry) or line (2D single expiry)')
    parser.add_argument('--target-dte', type=int, default=30, help='Target DTE for line mode (finds closest match)')
    parser.add_argument('--validate', action='store_true', help='Validate CSV headers and exit')
    parser.add_argument('--list-columns', action='store_true', help='List CSV column headers and exit')
    parser.add_argument('--schema-report', action='store_true', help='Show detailed schema mapping and sample rows')
    parser.add_argument('--strict', action='store_true', help='Enable strict validation (fails if numeric validity < threshold)')
    parser.add_argument('--threshold', type=float, default=95.0, help='Minimum validity percentage for strict mode (default: 95.0)')
    return parser.parse_args()

def main():
    # Detect if run from CLI with args or needs GUI
    if len(sys.argv) > 1:
        # CLI MODE
        args = parse_arguments()
        if not args.input:
            print("Error: -i / --input is required in CLI mode.")
            sys.exit(1)

        # Schema report mode
        if args.schema_report:
            ok = schema_report(args.input)
            if args.diff:
                print("\n" + "="*60)
                print("Diff/Reference file schema:")
                print("="*60)
                ok2 = schema_report(args.diff)
                ok = ok and ok2
            sys.exit(0 if ok else 1)

        # List-only mode
        if args.list_columns:
            try:
                df_tmp = pd.read_csv(args.input, skiprows=2)
                df_tmp = df_tmp.dropna(how='all')
                print("\nCSV Columns:")
                for c in df_tmp.columns:
                    print(f" - {c}")
            except Exception as e:
                print(f"Error reading {args.input}: {e}")
                sys.exit(1)
            if args.diff:
                print("\nDiff/Reference file columns:")
                try:
                    df_tmp2 = pd.read_csv(args.diff, skiprows=2)
                    df_tmp2 = df_tmp2.dropna(how='all')
                    for c in df_tmp2.columns:
                        print(f" - {c}")
                except Exception as e:
                    print(f"Error reading {args.diff}: {e}")
                    sys.exit(1)
            sys.exit(0)

        # Validation-only mode
        if args.validate:
            ok, msg = validate_csv(args.input, strict=args.strict, threshold=args.threshold)
            # If diff provided, validate that too
            if args.diff:
                print("\nValidating diff/reference file...")
                ok2, msg2 = validate_csv(args.diff, strict=args.strict, threshold=args.threshold)
                ok = ok and ok2
                msg = msg if ok else (msg2 if not ok2 else msg)
            if not ok:
                print(msg)
                sys.exit(1)
            else:
                if args.strict:
                    print(f"\nValidation succeeded (strict mode, threshold: {args.threshold}%).")
                else:
                    print("\nValidation succeeded.")
                sys.exit(0)
            
        # Check if output was explicitly provided or using default
        auto_output = args.output == 'vol_analysis.html'
        
        success, msg = run_analysis(
            input_file=args.input,
            output_file=args.output,
            spot=args.spot,
            date=args.date,
            diff_file=args.diff,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            open_in_browser=True,
            mode=args.mode,
            target_dte_line=args.target_dte,
            auto_output=auto_output
        )
        if not success:
            print(msg)
            sys.exit(1)
    else:
        # GUI MODE
        if not GUI_AVAILABLE:
            print("Error: Tkinter not installed/supported. Please use CLI arguments.")
            sys.exit(1)
        
        root = tk.Tk()
        app = IVSurfaceGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()
