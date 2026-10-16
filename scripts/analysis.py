"""
Data analysis script, combining all other scripts accordingly.
"""
import os
import re
import numpy as np
import pandas as pd
from collections import Counter

from config import DPI, FIG_DIR, TAB_DIR, VIRIDIS
from clean import load_clean, split_multi
from charts import hbar, lollipop, donut, treemap, line_trend, stacked_bar
from map_fig import make_map
from chord_fig import make_chord
from descriptive_proportions import descriptive_strata_plot

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def counter_from_lists(series_of_lists):
    c = Counter()
    for lst in series_of_lists:
        for item in lst:
            c[item] += 1
    return c

# ================================================================
df = load_clean('/users/chiomab/adr/data/adr.csv')
N = len(df)
print(f'Total studies included in quantitative synthesis: N = {N}  |  DPI = {DPI}')

results = {'N': N}

# =====================================================================
# LAYER 1: DESCRIPTIVE & BIBLIOMETRIC
# =====================================================================

yr_counts = df['pub_year'].value_counts().sort_index()
line_trend(yr_counts.index.astype(int), yr_counts.values, 'fig01_year_trend.png',
           xlabel='Publication Year', ylabel='Number of Studies', cumulative=True)
results['year_range'] = (int(df['pub_year'].min()), int(df['pub_year'].max()))
results['peak_year'] = (int(yr_counts.idxmax()), int(yr_counts.max()))

# ------------------------------------------------------------------------------------------ #
journal_counts = df['journal'].value_counts()
hbar(list(journal_counts.items()), 'fig02_top_journals.png', top=10)
results['n_journals'] = df['journal'].nunique()
results['top_journal'] = journal_counts.index[0], int(journal_counts.iloc[0])

# ------------------------------------------------------------------------------------------ #
map_fname = make_map(df, 'fig03_map_choropleth.png')
macro_counts = counter_from_lists(df['macro_region_list'])
africa_n = df['is_africa'].sum()
results['africa_n'] = int(africa_n)
results['africa_pct'] = round(100*africa_n/N, 1)
results['lead_africa_n'] = int(df['lead_is_africa'].sum())
results['lead_africa_pct'] = round(100*df['lead_is_africa'].sum()/N, 1)

# ------------------------------------------------------------------------------------------ #
chord_fname, chord_matrix = make_chord(df, 'fig04_chord_collaboration.png')
chord_matrix.to_csv(f'{TAB_DIR}/table_chord_matrix.csv')

# ------------------------------------------------------------------------------------------ #
def design_bucket(label):
    l = label.lower()
    if 'surveillance' in l: return 'Surveillance (field / epidemiology)'
    if 'computational-only' in l or 'in silico' in l: return 'Computational-only (in silico)'
    if 'methodology' in l: return 'Methodology / pipeline development'
    if 'experimental' in l or 'laboratory' in l or 'in vitro' in l: return 'Experimental / laboratory (in vitro-ex vivo)'
    if 'cross-sectional' in l: return 'Cross-sectional / molecular epidemiology'
    if 'randomized' in l: return 'Randomized controlled trial'
    if 'population genomics' in l: return 'Population genomics'
    if 'retrospective' in l or 'historical' in l or 'paleogenomic' in l: return 'Retrospective / historical genomic study'
    if 'mathematical modeling' in l or 'computational modelling' in l or 'computational biology' in l: return 'Mathematical / computational modelling'
    return 'Other'
design_bucketed = Counter()
for lst in df['study_design_list']:
    seen = set(design_bucket(x) for x in lst) if lst else {'Not reported'}
    for b in seen:
        design_bucketed[b] += 1

mc_counts = df['multi_country_clean'].value_counts()
results['multicountry_yes_pct'] = round(100*mc_counts.get('Yes',0)/N,1)

# ------------------------------------------------------------------------------------------ #
results['n_methodology_only'] = int(df['is_methodology_only'].sum())
results['n_hmm_dcifer_cite'] = int(df['software_list'].apply(
    lambda lst: any(t.lower() in ('hmmibd','dcifer','moire','real mccoil','the real mccoil')
                     for t in [x.lower() for x in lst])).sum())
# =====================================================================
# LAYER 2: GENOMIC & BIOLOGICAL
# =====================================================================

species_counter = counter_from_lists(df['species_list'])
def species_bucket(s):
    s = s.lower()
    if 'falciparum' in s: return 'P. falciparum'
    if 'vivax' in s: return 'P. vivax'
    if 'knowlesi' in s: return 'P. knowlesi'
    if 'malariae' in s: return 'P. malariae'
    if 'ovale' in s: return 'P. ovale'
    if 'yoelii' in s: return 'Others'
    if 'berghei' in s: return 'Others'
    if 'chabaudi' in s: return 'Others'
    return s
species_bucketed = Counter()
for lst in df['species_list']:
    seen = set(species_bucket(x) for x in lst) if lst else set()
    for b in seen:
        species_bucketed[b] += 1
donut(species_bucketed.most_common(), 'fig05_species.png', legend_title='Plasmodium Species')
results['pf_pct'] = round(100*species_bucketed.get('P. falciparum',0)/N,1)
results['pv_pct'] = round(100*species_bucketed.get('P. vivax',0)/N,1)

gene_counter = counter_from_lists(df['gene_list'])
drug_counter = counter_from_lists(df['drug_list'])

drugclass_counter = counter_from_lists(df['drug_class_list'])
donut(drugclass_counter.most_common(), 'fig06_drug_class.png', top=5, legend_title='Drug Class')

# ------------------------------------------------------------------------------------------ #
def detect_bucket(label):
    l = label.lower()
    if 'whole-genome' in l or 'wgs' in l: return 'Whole-genome sequencing (WGS)'
    if 'sanger' in l: return 'Sanger sequencing'
    if 'nanopore' in l or 'ont' in l: return 'Nanopore sequencing'
    if 'illumina' in l or 'miseq' in l or 'novaseq' in l: return None  # folded into scope elsewhere
    if 'amplicon' in l or 'ampseq' in l: return 'Targeted amplicon sequencing'
    if 'rflp' in l: return 'PCR-RFLP'
    if 'pcr' in l and 'nested' in l: return 'Nested PCR'
    if 'qpcr' in l or 'rt-qpcr' in l: return 'qPCR / RT-qPCR'
    if 'microscopy' in l: return 'Microscopy'
    if 'single-cell' in l or 'scrna' in l: return 'Single-cell sequencing'
    if 'microarray' in l: return 'Microarray'
    return label

def classify_detection(tok_list):
    joined = ' ; '.join(tok_list).lower()
    buckets = set()
    scope_found = False
    for t in tok_list:
        l = t.lower()
        if 'whole-genome' in l or 'wgs' in l or 'whole genome amplification' in l:
            buckets.add('Whole-genome sequencing (WGS)'); scope_found = True
        elif 'amplicon' in l or 'ampseq' in l or 'molecular inversion probes' in l or 'mips' in l:
            buckets.add('Targeted amplicon sequencing'); scope_found = True
        elif 'sanger' in l:
            buckets.add('Sanger sequencing (single-locus)'); scope_found = True
        elif 'rflp' in l:
            buckets.add('PCR-RFLP'); scope_found = True
        elif 'nested' in l and 'pcr' in l:
            buckets.add('Nested PCR'); scope_found = True
        elif 'qpcr' in l or 'rt-qpcr' in l:
            buckets.add('qPCR / RT-qPCR'); scope_found = True
        elif 'microscopy' in l:
            buckets.add('Microscopy'); scope_found = True
        elif 'single-cell' in l or 'scrna' in l:
            buckets.add('Single-cell sequencing'); scope_found = True
        elif 'microarray' in l:
            buckets.add('Microarray'); scope_found = True
        elif 'nanopore' in l or 'ont' in l:
            buckets.add('Nanopore sequencing'); scope_found = True
        elif 'illumina' in l or 'miseq' in l or 'novaseq' in l or 'hiseq' in l or 'nextseq' in l:
            pass
        else:
            buckets.add(t)
    if not scope_found and ('illumina' in joined or 'miseq' in joined or 'novaseq' in joined):
        buckets.add('Illumina sequencing (scope not specified)')
    return buckets

detect_bucketed = Counter()
for lst in df['detection_list']:
    seen = classify_detection(lst) if lst else set()
    for b in seen:
        detect_bucketed[b] += 1

feature_counter = counter_from_lists(df['feature_list'])
def feat_bucket(l):
    ll = l.lower()
    if 'snp' in ll: return 'SNPs'
    if 'haplotype' in ll and 'micro' not in ll: return 'Haplotypes'
    if 'microhaplotype' in ll: return 'Microhaplotypes'
    if 'cnv' in ll or 'copy number' in ll: return 'Copy number variants (CNVs)'
    if 'genomic region' in ll: return 'Genomic regions'
    if 'gene-level' in ll: return 'Gene-level features'
    return l
feat_bucketed = Counter()
for lst in df['feature_list']:
    seen = set(feat_bucket(x) for x in lst) if lst else set()
    for b in seen:
        feat_bucketed[b] += 1
# =====================================================================
# LAYER 3: COMPUTATIONAL & MODELLING
# =====================================================================
def cat_bucket(l):
    ll = l.lower()
    if 'genomic surveillance' in ll: return 'Genomic surveillance'
    if 'population genomics' in ll: return 'Population genomics'
    if 'machine learning' in ll: return 'Machine learning'
    if 'computational modelling' in ll: return 'Computational modelling'
    if 'candidate gene' in ll: return 'Candidate gene study'
    if 'methodology' in ll or 'pipeline' in ll: return 'Methodology / pipeline development'
    if 'clinical trial' in ll: return 'Clinical trial'
    if 'paleogenomics' in ll: return 'Paleogenomics'
    if 'transcriptomic' in ll: return 'Transcriptomics'
    if 'molecular epidemiology' in ll: return 'Molecular epidemiology'
    if 'mixed' in ll: return 'Mixed / multi-category'
    return 'Other'
cat_bucketed = Counter()
for lst in df['study_category_list']:
    seen = set(cat_bucket(x) for x in lst) if lst else set()
    for b in seen:
        cat_bucketed[b] += 1

# ------------------------------------------------------------------------------------------ #
def analysis_bucket(l):
    ll = l.lower()
    if 'neighbor-joining' in ll or 'neighbour-joining' in ll or 'neighbor joining' in ll:
        return 'Tree-based clustering (e.g. neighbour-joining)'
    if 'phylogenetic' in ll:
        return 'Model-based phylogenetic analysis'
    if 'haplotype analysis' in ll: return 'Haplotype analysis'
    if 'allele frequency' in ll: return 'Allele frequency analysis'
    if 'machine learning' in ll: return 'Machine learning'
    if 'regression' in ll: return 'Regression modelling'
    if 'population differentiation' in ll or 'fst' in ll: return 'Population differentiation (Fst)'
    if 'selection scan' in ll or 'ihs' in ll: return 'Selection scans'
    if 'neutrality test' in ll: return 'Neutrality tests'
    if 'relatedness' in ll or 'ibd' in ll: return 'Relatedness / IBD analysis'
    if 'moi' in ll or 'multiplicity' in ll: return 'Multiplicity of infection (MOI)'
    if 'descriptive statistics' in ll: return 'Descriptive statistics'
    if 'gwas' in ll: return 'GWAS'
    return l
analysis_bucketed = Counter()
for lst in df['analysis_list']:
    seen = set(analysis_bucket(x) for x in lst) if lst else set()
    for b in seen:
        analysis_bucketed[b] += 1

# ------------------------------------------------------------------------------------------ #
ml_model_counter = counter_from_lists(df[df['ml_used_clean']=='Yes']['ml_model_list'])

df['period'] = pd.cut(df['pub_year'], bins=[2000,2015,2020,2023,2027],
                       labels=['\u2264 2015','2016\u20132020','2021\u20132023','2024\u20132026'])
ml_by_period = df.groupby('period', observed=True)['ml_used_clean'].apply(lambda s: (s=='Yes').mean()*100)
line_trend(ml_by_period.index.astype(str), ml_by_period.values, 'fig07_ml_trend.png',
           xlabel='Publication Period', ylabel='% of Studies Using ML', cumulative=False)

CANON_TOOL = {}
raw_counter = counter_from_lists(df['software_list'])
merged_counter = Counter()
for tool, cnt in raw_counter.items():
    key = re.sub(r'\s+v?\d[\d.]*$', '', tool).strip().lower()
    if key not in CANON_TOOL:
        CANON_TOOL[key] = tool
    merged_counter[CANON_TOOL[key]] += cnt
software_counter = merged_counter
results['n_unique_tools'] = len(software_counter)

# ------------------------------------------------------------------------------------------ #
repro_df = pd.DataFrame({
    'Raw data available': df['raw_data_clean'].value_counts(),
    'Code available': df['code_clean'].value_counts(),
    'Pipeline clearly reported': df['pipeline_clean'].replace(
        {'Yes (clearly described)': 'Yes'}).value_counts(),
}).fillna(0).astype(int)
repro_df = repro_df.reindex(['Yes','Not applicable','No','Not reported']).fillna(0).astype(int)
colors_map = {'Yes': VIRIDIS(0.85), 'Not applicable': VIRIDIS(0.55), 'No': VIRIDIS(0.30), 'Not reported': VIRIDIS(0.08)}
stacked_bar(repro_df.columns.tolist(), {s: repro_df.loc[s].values for s in repro_df.index}, colors_map,
            'fig08_reproducibility.png')

results['raw_data_yes_pct'] = round(100*(df['raw_data_clean']=='Yes').mean(),1)
results['code_yes_pct'] = round(100*(df['code_clean']=='Yes').mean(),1)
results['pipeline_yes_pct'] = round(100*(df['pipeline_clean']=='Yes (clearly described)').mean(),1)

# =====================================================================
# DESCRIPTIVE STRATIFIED PROPORTIONS (replaces pooled meta-analysis)
# =====================================================================
def region_group(row):
    if row['is_africa']: return 'Africa-based samples'
    if 'Global / Multi-region' in row['macro_region_list']: return 'Global / multi-region dataset'
    return 'Non-Africa (single region)'
df['region_group'] = df.apply(region_group, axis=1)

order = ['Africa-based samples', 'Non-Africa (single region)', 'Global / multi-region dataset']
INDICATORS = {
    'ml_yes': ('ml_used_clean', 'Machine Learning Adoption', 'fig09_desc_ml.png'),
    'code_yes': ('code_clean', 'Open-Code Availability', 'fig10_desc_code.png'),
    'rawdata_yes': ('raw_data_clean', 'Raw Genomic Data Availability', 'fig11_desc_rawdata.png'),
    'pipeline_yes': ('pipeline_clean', 'Clearly Reported Bioinformatics Pipeline', 'fig12_desc_pipeline.png'),
}
desc_summary_rows = []
for key, (col, title, fname) in INDICATORS.items():
    events, ns, labels = [], [], []
    for g in order:
        sub = df[df['region_group'] == g]
        n = len(sub)
        if col == 'pipeline_clean':
            ev = int((sub[col] == 'Yes (clearly described)').sum())
        else:
            ev = int((sub[col] == 'Yes').sum())
        events.append(ev); ns.append(n); labels.append(f'{g} (n={n})')
    if col == 'pipeline_clean':
        overall_ev = int((df[col] == 'Yes (clearly described)').sum())
    else:
        overall_ev = int((df[col] == 'Yes').sum())
    rows = descriptive_strata_plot(events, ns, labels, fname, xlab=title,
                                     overall_events=overall_ev, overall_n=N)
    for r in rows:
        r['Indicator'] = title
        desc_summary_rows.append(r)

desc_summary = pd.DataFrame(desc_summary_rows)[['Indicator','Stratum','n','N','Proportion (%)','CI lower','CI upper']]
desc_summary.to_csv(f'{TAB_DIR}/table_descriptive_proportions.csv', index=False)

print('All figures regenerated.')
print(results)

df.to_pickle('/users/chiomab/adr/data/df_clean.pkl')
df.to_csv('/users/chiomab/adr/tables/adr_cleaned.csv')

