# Options IV Surface Tool

Flexible parsing and visualization for options IV surfaces and smiles from Fidelity ATP CSV exports. Works for any ticker as long as the export format matches the expected headers. Sample data in this repo uses NVDA, but the code is ticker-agnostic.

## Installation

```powershell
# Clone the repository
git clone https://github.com/CaptPat/options-iv-surface.git
cd options-iv-surface

# Create and activate conda environment (recommended)
conda create -n VolMapStudy python=3.11
conda activate VolMapStudy

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```powershell
# GUI mode (no arguments)
python plot_vol_surface.py

# CLI: Generate volatility surface
python plot_vol_surface.py -i testdata01.csv -o vol_analysis.html

# CLI: 2D smile with pin target
python plot_vol_surface.py -i testdata01.csv --mode line --target-dte 7

# CLI: Diff analysis
python plot_vol_surface.py -i testdata01.csv --diff testdata02.csv
```

## Header-Aware CSV Parsing

The tool reads the first two metadata lines for spot and timestamps, then uses the table header row to detect columns. It supports common Fidelity patterns and minor variations.

### Required Fields

- Expiration Date: header containing expir or expiry
- Strike: header containing strike
- Call Bid / Call Ask
- Put Bid / Put Ask

### Flexible Detection

- If headers explicitly include call/put with bid/ask, they are used.
- Otherwise, the first Bid/Ask are mapped to Calls, and the second Bid.1/Ask.1 to Puts (Fidelity-style).
- Extra columns (e.g., Delta, Gamma, Vega, Theta, Rho) are preserved and available for future analysis.

### Spot & Timestamp Parsing

- Spot price: accepts Last Trade $... or Last Price $... on line 1
- Timestamps: line 1 (as of ...) and line 2 (e.g., MM/DD/YYYY hh:mm:ss AM ET)

## Usage

```powershell
# List headers (input and optional diff)
python plot_vol_surface.py --list-columns -i fidelityATPExport-options20251210close.csv --diff fidelityATPExport-options20251210noon.csv

# Schema report (detailed mapping + sample rows)
python plot_vol_surface.py --schema-report -i fidelityATPExport-options20251210close.csv

# Validate headers only
python plot_vol_surface.py --validate -i fidelityATPExport-options20251210close.csv

# Strict validation (fail if any field < 95% numeric validity)
python plot_vol_surface.py --validate --strict -i fidelityATPExport-options20251210close.csv

# Custom threshold for strict mode
python plot_vol_surface.py --validate --strict --threshold 98.0 -i fidelityATPExport-options20251210close.csv

# Generate surface
python plot_vol_surface.py -i fidelityATPExport-options20251210close.csv -o vol_analysis.html

# Diff mode
python plot_vol_surface.py -i fidelityATPExport-options20251210close.csv `
  --diff fidelityATPExport-options20251210noon.csv -o diff_analysis.html
```

## Error Handling

If required fields are missing, the tool prints:

- Which fields are missing
- Which headers were found
- Tips to correct the export (include Strike, Expiration, Call Bid/Ask, Put Bid/Ask)

Validation also reports numeric conversion stats for core fields (valid counts/percentages) to help spot formatting issues.

## Validation Modes

- **--list-columns**: Quick list of all CSV headers
- **--schema-report**: Detailed view of canonical mappings, sample rows, and extra columns
- **--validate**: Header detection + numeric conversion stats
- **--strict**: Fail validation if any core field has < threshold% valid numeric values (default: 95%)
  - Useful for CI/CD pipelines and batch processing to catch malformed exports early

## Notes

- Date parsing uses pandas.to_datetime for Expiration Date.
- IV selection logic prefers Put IV below spot and Call IV at/above spot.
- Charts auto-open unless disabled via GUI; CLI always opens on success.
