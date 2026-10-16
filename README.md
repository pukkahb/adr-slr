# Genomic and Computational Approaches to Antimalarial Drug Resistance: Analysis Code

Python pipeline and extracted dataset underlying the systematic review
*"Genomic and Computational Approaches to Antimalarial Drug Resistance: A Systematic Review"*.
The code reproduces every figure, table, and descriptive statistic reported in the manuscript
from the extracted study-level dataset (77 included studies).

> **Citation:** Temitayo Ogundimu, Emmanuella Matumamboh, Jerry Emmanuel, Jelili Oyelade. Genomic and Computational Approaches to Antimalarial Drug Resistance:
> A Systematic Review. [In Review Stage, 2026].

## What this repository does

- Cleans and normalises the free-text extraction fields (genes, drugs, regions, software, methods)
- Applies an explicit, rule-based classification of machine-learning use (predictive/trained models only)
- Derives bioinformatics-pipeline reporting status directly from each study's free-text pipeline description
- Generates all figures (publication trend, choropleth map, collaboration chord diagram, detection-method
  treemap, reproducibility indicators, region-stratified proportions, etc.)
- Reports region-stratified methodological/reproducibility indicators as descriptive proportions with
  95% Wilson confidence intervals (no pooling)
- Runs supplementary checks: chi-squared tests across region strata, a binary geographic classification
  sensitivity analysis, and a machine-learning definition of sensitivity analysis

## Repository structure

```
.
├── scripts/
│   ├── config.py                  # DPI, output paths, plotting style
│   ├── clean.py                   # data loading, normalisation, ML and pipeline-status rules
│   ├── charts.py                  # reusable chart functions
│   ├── map_fig.py                 # choropleth map with macro-region rings
│   ├── chord_fig.py               # sample-origin vs lead-author-region chord diagram
│   ├── descriptive_proportions.py # Wilson-CI stratified proportion plots
│   ├── analysis.py                # main script: figures + descriptive statistics
│   ├── sensitivity_analysis.py    # chi-squared tests and sensitivity analyses
│   └── tables_export.py           # exports all summary tables as CSV
├── data/
│   └── adr.csv                    # extracted dataset (77 studies)
├── tables/                        # summary tables (CSV)
├── figures/                       # figures (PNG)
├── world.geojson
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

1. **Set paths.** Open `scripts/config.py` and set `FIG_DIR` and `TAB_DIR` to your local output folders.
   Then update the input and output file paths near the top of `scripts/analysis.py`,
   `scripts/sensitivity_analysis.py`, and `scripts/map_fig.py` (`WORLD_GEOJSON`) to match your machine.
   Figure resolution is controlled by `DPI` in `config.py`.
2. **Run the main analysis** (generates figures and `df_clean.pkl`):
   ```bash
   python3 scripts/analysis.py
   ```
3. **Export tables and run the supplementary statistical checks:**
   ```bash
   python3 scripts/tables_export.py
   ```
   This also runs `sensitivity_analysis.py`, so that the chi-squared
   and sensitivity analyses are generated and written to CSV.

The country boundary file used for the map is downloaded automatically on the first run
(Natural Earth, 110m). If there is no internet access, the map is skipped, and all other outputs are unaffected.

## Key classification rules (documented in `scripts/clean.py`)

- **Machine learning** is counted only where a model was trained to map features to a labelled or predicted
  outcome (e.g. random forest, SVM, neural networks, trained pathogenicity predictors). Unsupervised
  ordination/clustering (PCA, PCoA, hierarchical clustering, DAPC, ADMIXTURE) and descriptive regression
  are classified as statistical methods.
- **Pipeline reporting** is "clearly described" when the free-text pipeline field contains a real description,
  "no" when it reads "not reported", and "not applicable" when no computational pipeline exists.
- **Region strata** for the stratified analysis are mutually exclusive: *Africa-based* if any African
  region is recorded, otherwise *Global / multi-region* if the region field reads "Global", otherwise
  *Non-Africa single-region*.

## Data

`data/adr.csv` contains one row per included study (77 rows). Fields cover study
characteristics, species, drugs, genes, detection and analysis methods, software, machine-learning use,
and data/code/pipeline availability. Free-text fields are extracted and normalised by `clean.py`.

## License

Code is released under the [MIT License](LICENSE). The extracted dataset in `data/` is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

Temitayo Ogundimu, Covenant University Bioinformatics Research [temitayo.ogundimupgs@stu.cu.edu.ng].
