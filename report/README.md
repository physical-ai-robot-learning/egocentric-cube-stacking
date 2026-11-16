# report

The technical report: **one current document**, `report_en.pdf` (15 pages, IEEEtran, version 5.2) and its
Korean twin `report_ko.pdf` (15 pages). Older and partial documents are in `archive/`.

**Status:** current (v5.2, 5 November 2026: v5 with the code map updated to the refactored layout; no result changed)

The LaTeX sources and the figure scripts are part of the source code, which is available on request
([Code availability](../README.md#code-availability)).

## How the numbers are checked

`analysis/verify_claims.py` (included with the code on request) recomputes every quantitative claim from the records in
`results/` and the calibrations in `configs/`, and checks that the report text contains each value: the unified report
`report_en.tex` (section `v5_unified`: every value required by the full-length v4 documents, retired values absent,
attribution sentences), the v4 full and short documents in `archive/`, and `README.md` + `docs/results.md` +
`docs/history.md`. Output: `736 claims checked, 0 failed.` The Korean report is a translation of the English one and is
not checked separately.

## Layout

- `report_en.pdf`, `report_ko.pdf` — the technical report (v5.2)
- `archive/` — v4 documents (Part I stacking, Part II approach, methods; short and full-length, EN/KO) and the v3 single report

## Provenance

v2 (2026-09-30) → v3 (10-04) → v4 split documents (10-07, 4.1 on 10-12) → v5 one report (10-12) → v5.1 paths after the
11-01 restructure → v5.2 (11-05) code map for the refactored layout. The short v4 documents and the methods document moved
from `report/` to `report/archive/` on 2026-11-05. Git tags `report-v2-2026-09-30` and `report-v3-2026-10-04`.
