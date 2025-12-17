# NVDA Volatility Surface Analysis Tool

## Project Overview
Python-based options analytics tool for visualizing implied volatility (IV) surfaces and smiles from Fidelity CSV exports. Generates interactive 3D surfaces (Plotly) and 2D "smile" charts, with differential analysis capabilities for tracking IV changes over time. Designed for NVDA (NVIDIA) options but adaptable to any equity.

## Core Architecture

### Current Version
- **`plot_vol_surface_v3.py`** - Production version with dual CLI/GUI modes
- Features: "Show Result" button, flexible date/DTE input, target DTE selection for line mode
- Historical note: Evolved from v1→v2→v3 (earlier versions removed; compare against n-1 for future versions)

### Data Flow
```
Fidelity CSV → extract_file_metadata() → process_data() → run_analysis() → Plotly HTML
    (header parsing)      (IV calculation)      (visualization)
```

## Fidelity CSV Format Requirements

The tool expects Fidelity-specific format with:
- **Line 1**: Spot price metadata - `"NVDA - NVIDIA CORP Last Trade $185.27 as of 12/8/2025 1:42:48 PM"`
- **Line 2**: Data timestamp - `"12/08/2025 01:42:58 PM ET"`
- **Line 3**: Column headers (18 columns: Calls + Strike + Puts with Greeks)
- **Line 4+**: Options chain data

CSV parsing: `skiprows=3` then standardize to 18-column layout in `process_data()`.

## Key Functions & Conventions

### Black-Scholes Implementation
- `black_scholes_price()`: Standard BS formula (no dividends)
- `implied_volatility()`: Root-finding via `scipy.optimize.brentq` with bounds [0.001, 5.0]
- **IV Selection Logic**: Use Put IV for strikes < spot, Call IV for strikes >= spot (reduces bid-ask bias)
- Risk-free rate: `RISK_FREE_RATE = 0.04` (global constant)

### Metadata Extraction
`extract_file_metadata()` parses header lines with regex:
- Spot price: `r'Last Trade \$(\d+\.\d+)'`
- Timestamps: `r'as of (.+)$'` and `'%m/%d/%Y %I:%M:%S %p'`
- Returns dict with `spot`, `spot_time_str`, `data_time_str`, `analysis_date`

### Data Processing Pipeline
1. **Load**: `pd.read_csv(filepath, skiprows=3)` → slice to 18 columns
2. **Clean**: Convert numeric fields, drop invalid strikes, skip blank row after header (`df.iloc[1:]`)
3. **Calculate**: Mid prices `(Bid + Ask) / 2`, DTE from `(Expiration - analysis_date).dt.days`, Time to expiry `DTE / 365.25`
4. **IV Computation**: Apply `implied_volatility()` row-wise for calls/puts
5. **Filter**: Min/Max DTE, strike bounds `[0.5*spot, 1.5*spot]`

### Visualization Modes
- **3D Surface** (default): Multiple expirations → pivot table → interpolate → `go.Surface()` with Viridis colorscale
- **2D Smile**: Single expiration → `go.Scatter()` with **Pin Target** annotation (strike with minimum IV)
- **Line Mode** (v3): Filter to single closest DTE via `target_dte_line` parameter
- **Diff Mode**: Merge two datasets → plot `IV_Diff` with RdBu_r colorscale (red=increase, blue=decrease)

## Development Environment

### Conda Environment
Active environment: `VolMapStudy` (visible in terminal context)
```powershell
conda activate P:\anaconda3\envs\VolMapStudy
```

### Dependencies
```python
pandas, numpy, scipy, plotly, tkinter (GUI), argparse, datetime, webbrowser
```

## CLI Usage Examples

```bash
# Basic surface
python plot_vol_surface_v3.py -i options.csv -o output.html

# With DTE filtering
python plot_vol_surface_v3.py -i options.csv --min-dte 7 --max-dte 90

# Diff mode (change analysis)
python plot_vol_surface_v3.py -i today.csv --diff yesterday.csv -o diff_analysis.html

# Override spot/date
python plot_vol_surface_v3.py -i options.csv -s 180.50 --date 2025-12-10
```

## GUI Mode (v3)
Run without arguments to launch Tkinter GUI with:
- File browser for input/diff CSVs
- Min/Max DTE inputs (numeric or YYYY-MM-DD date strings)
- Mode selector: Surface vs. Line (single expiration)
- Target DTE field for line mode (finds closest match)
- "Compute VSM" button → calls `run_analysis()` with GUI params
- "Show Result" button → reopens last generated HTML without recalculation

## Common Development Tasks

### Adding New Visualization Features
Modify `run_analysis()` after line `# --- VISUALIZATION ---`. Check `z_column` to handle both IV and IV_Diff modes. Reference current spot via `meta_main['spot']`.

### Adjusting IV Calculation
Edit `implied_volatility()` bounds or `RISK_FREE_RATE`. Changes affect all modes globally.

### Debugging CSV Parse Errors
Enable verbose output in `extract_file_metadata()`. Check if Fidelity format changed (common for header regex mismatches).

### Interpolation Settings
3D mode uses linear for strikes, cubic for DTE (if >3 expirations). Tuned in `run_analysis()` after `iv_matrix.reindex()`.

## Testing & Regression

### Test Data Organization
- **Primary test files**: `testdata01.csv`, `testdata02.csv` - canonical datasets for quick validation
- **Extended test suite**: `fidelityATPExport-options*.csv` - real historical exports for comprehensive testing
  - Naming convention: `fidelityATPExport-options[DATE][TIME].csv` (e.g., `options20251210close.csv`)
  - Organized chronologically for temporal diff analysis testing

### Regression Testing Workflow
```bash
# Validate against primary test data
python plot_vol_surface_v3.py -i testdata01.csv -o test_output01.html
python plot_vol_surface_v3.py -i testdata02.csv -o test_output02.html

# Test diff mode with temporal pairs
python plot_vol_surface_v3.py -i fidelityATPExport-options20251210close.csv \
    --diff fidelityATPExport-options20251210noon.csv -o diff_test.html

# For future versions: compare new version vs v3 (current) outputs
```

When modifying core functions (`black_scholes_price`, `implied_volatility`, `process_data`), capture console output from test files as baseline before changes, then verify outputs remain consistent after modifications.

## Output Files
- HTML visualizations: `vol_analysis.html`, `diff_analysis.html`, `options2series.html`
- Auto-opened in browser via `webbrowser.open('file://...')`
- Self-contained Plotly charts (no external dependencies)

## Pin Target Calculation (2D Mode)
When single expiration detected:
```python
min_iv_idx = df_plot[z_column].idxmin()
min_iv_strike = df_plot.loc[min_iv_idx, 'Strike']
```
Annotates strike with lowest IV (gamma concentration point). Logged to console and displayed on chart.

## Important Quirks
- **Date handling**: GUI supports both DTE integers and date strings. Resolved via `resolve_date_input()` helper.
- **Empty diff failures**: Diff mode silently falls back to single-file mode if reference file fails.
- **Title formatting**: Uses `<br>` for multi-line titles in Plotly (not `\n`).
- **Column slicing**: Always take first 18 columns after skip - extra columns ignored.

## Mode Logic & UI Behavior (v3 Implementation Details)

### CLI vs GUI Auto-Detection
- **CLI mode triggered**: When `sys.argv` length > 1 (any arguments provided)
- **GUI mode triggered**: When script runs without arguments (requires `tkinter`)
- Single codebase handles both via `main()` dispatcher function

### Date/DTE Resolution Pattern
GUI supports flexible input via `resolve_date_input()`:
- If date field contains YYYY-MM-DD string → calculate DTE as delta from analysis date
- Else use numeric DTE field directly
- Min default: 1, Max default: 3650 (10 years)
- Used for both time bounds in analysis

### Error Handling in Diff Mode
```python
# Diff file parsing wrapped in try-except
# Silently continues with single-file mode if merge fails
# No exception propagation - user still gets valid output
```

## Implementation Patterns to Preserve

### IV Calculation Pattern
The tool uses **strike-dependent IV selection** to minimize bid-ask spread noise:
```python
df['IV'] = np.where(df['Strike'] < spot_price, df['Put IV'], df['Call IV'])
```
This pattern appears in both `process_data()` and visualization filtering. When adding new features, respect this OTM-preference convention.

### Plotly Chart Type Selection Logic
- **Single DTE** (1 expiration) → 2D Scatter (smile curve) with pin target annotation
- **Multiple DTEs** (2+ expirations) → 3D Surface with interpolation
- Interpolation: linear for strikes (always), cubic for DTE columns (if 3+ expirations available)
- Colorscale selection: `Viridis` for IV, `RdBu_r` for diffs (red=IV up, blue=IV down)

### Header Parsing Strategy
Fidelity format is rigid; regex patterns are specific:
- Spot: `r'Last Trade \$(\d+\.\d+)'` (requires $ literal)
- Time in line 1: `r'as of (.+)$'` captures everything after "as of"
- Line 2 stripped of " ET" suffix before datetime parsing
- Any deviation (e.g., "Last Price" instead of "Last Trade") breaks extraction

## Extension Points for Future Development

1. **New Visualization Modes**: Add to `mode` parameter in `run_analysis()`, insert logic after `# --- VISUALIZATION ---` comment
2. **Additional IV Models**: Replace `black_scholes_price()` implementation; `implied_volatility()` remains wrapper-agnostic
3. **Different Colorscales**: Update `c_scale` variable in visualization block; consider diverging for diffs
4. **CSV Format Support**: Extend `extract_file_metadata()` with conditional logic for different broker formats
5. **GUI Enhancements**: Modify `IVSurfaceGUI` class; preserve `resolve_date_input()` pattern for date handling consistency
