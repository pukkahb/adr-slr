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
# --- 1.1 Publication trend ---
yr_counts = df['pub_year'].value_counts().sort_index()
line_trend(yr_counts.index.astype(int), yr_counts.values, 'fig01_year_trend.png',
           xlabel='Publication Year', ylabel='Number of Studies', cumulative=True)
results['year_range'] = (int(df['pub_year'].min()), int(df['pub_year'].max()))
results['peak_year'] = (int(yr_counts.idxmax()), int(yr_counts.max()))

# --- 1.2 Top journals ---
journal_counts = df['journal'].value_counts()
hbar(list(journal_counts.items()), 'fig02_top_journals.png', top=10)
results['n_journals'] = df['journal'].nunique()
results['top_journal'] = journal_counts.index[0], int(journal_counts.iloc[0])

# --- 1.3 + 1.4 map ---
map_fname = make_map(df, 'fig03_map_choropleth.png')
macro_counts = counter_from_lists(df['macro_region_list'])
africa_n = df['is_africa'].sum()
results['africa_n'] = int(africa_n)
results['africa_pct'] = round(100*africa_n/N, 1)
results['lead_africa_n'] = int(df['lead_is_africa'].sum())
results['lead_africa_pct'] = round(100*df['lead_is_africa'].sum()/N, 1)

# --- 1.5 chord ---
chord_fname, chord_matrix = make_chord(df, 'fig04_chord_collaboration.png')
chord_matrix.to_csv(f'{TAB_DIR}/table_chord_matrix.csv')

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

drugclass_counter = counter_from_lists(df['drug_class_list'])
donut(drugclass_counter.most_common(), 'fig06_drug_class.png', top=5, legend_title='Drug Class')

# =====================================================================
# LAYER 3: COMPUTATIONAL & MODELLING
# =====================================================================
# --- 3.1 ML usage: RECLASSIFIED per explicit predictive-model rule (see clean.py) ---
ml_model_counter = counter_from_lists(df[df['ml_used_clean']=='Yes']['ml_model_list'])

df['period'] = pd.cut(df['pub_year'], bins=[2000,2015,2020,2023,2027],
                       labels=['\u2264 2015','2016\u20132020','2021\u20132023','2024\u20132026'])
ml_by_period = df.groupby('period', observed=True)['ml_used_clean'].apply(lambda s: (s=='Yes').mean()*100)
line_trend(ml_by_period.index.astype(str), ml_by_period.values, 'fig07_ml_trend.png',
           xlabel='Publication Period', ylabel='% of Studies Using ML', cumulative=False)

# --- 3.2 Reproducibility: pipeline_clean is RE-DERIVED (see clean.py) ---
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
