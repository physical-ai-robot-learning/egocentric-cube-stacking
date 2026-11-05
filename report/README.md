# report

The technical report: **one current document**, `report_en.tex` / `report_en.pdf` (15 pages, IEEEtran, version 5.2) and its
Korean twin `report_ko.tex` / `report_ko.pdf` (15 pages). Older and partial documents are in `archive/`.

**Status:** current (v5.2, 5 November 2026: v5 with the code map updated to the refactored layout; no result changed)

## Inputs → Outputs

In: the LaTeX sources and `figures/`. Out: the PDFs (committed).

## Build

```bash
cd report
tectonic report_en.tex
tectonic report_ko.tex            # xeCJK with the macOS fonts AppleMyungjo / Apple SD Gothic Neo
python3 figures/make_figures.py && python3 figures/make_figures_v3.py && python3 figures/make_fig10_handumi_mods.py   # rebuild figures
python3 ../analysis/v4_r30_ablation.py && python3 ../analysis/v4_hra_summary.py                                  # Figs. 8-9
cd archive && tectonic part1_stacking_en.tex      # archived documents build from archive/ (figures via ../figures/)
```

## How the numbers are checked

`python analysis/verify_claims.py` (run in CI) recomputes every quantitative claim from the records in `results/` and the
calibrations in `configs/`, and checks that the report text contains each value: the unified report `report_en.tex`
(section `v5_unified`: every value required by the full-length v4 documents, retired values absent, attribution sentences),
the v4 full and short documents in `archive/`, and `README.md` + `docs/results.md` + `docs/history.md`. Output: `736 claims
checked, 0 failed.`; per-claim table with `--md analysis/out/claim_audit.md`. The Korean report is a translation of the
English one and is not checked separately.

## Layout

- `report_en.tex/.pdf`, `report_ko.tex/.pdf` — the technical report (v5.2)
- `figures/` — figure PDFs/PNGs and the scripts that make them
- `archive/` — v4 documents (Part I stacking, Part II approach, methods; short and full-length, EN/KO) and the v3 single report

## Provenance

v2 (2026-09-30) → v3 (10-04) → v4 split documents (10-07, 4.1 on 10-12) → v5 one report (10-12) → v5.1 paths after the
11-01 restructure → v5.2 (11-05) code map for the refactored layout. The short v4 documents and the methods document moved
from `report/` to `report/archive/` on 2026-11-05. Git tags `report-v2-2026-09-30` and `report-v3-2026-10-04`.
