# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 — 2025-12-17
- Initial public release.
- Generic naming: tool is ticker-agnostic and works with any Fidelity-format options CSV export.
- Line mode fix: Diff CSV is fully ignored when plotting the single-expiration smile and pin target.
- GUI enhancement: Output filename auto-switches based on mode and Diff CSV presence (surface → vol/diff; line → pin).
- CI: Basic GitHub Actions workflow to lint and validate sample CSV headers.
- Docs: README updated with Modes section and correct clone URL.
