import pandas as pd
from collections import Counter
from config import TAB_DIR
from clean import load_clean

df = pd.read_pickle('/data/df_clean.pkl')
OUT = TAB_DIR

def pct(n, N=77):
    return f'{n} ({100*n/N:.1f}%)'

overview_rows = [
    ('Total studies included', str(len(df))),
    ('Publication years', f"{int(df['pub_year'].min())}\u2013{int(df['pub_year'].max())}"),
    ('Studies with African sample origin', pct(int(df['is_africa'].sum()))),
    ('Studies with Africa-affiliated lead author', pct(int(df['lead_is_africa'].sum()))),
    ('Multi-country studies', pct(int((df['multi_country_clean']=='Yes').sum()))),
    ('Studies reporting P. falciparum', pct(int(df['species_list'].apply(lambda l: any('falciparum' in s.lower() for s in l)).sum()))),
    ('Studies reporting P. vivax', pct(int(df['species_list'].apply(lambda l: any('vivax' in s.lower() for s in l)).sum()))),
    ('Studies classified as using machine learning (predictive-model rule)', pct(int((df['ml_used_clean']=='Yes').sum()))),
    ('Studies reclassified from ML=Yes to No (descriptive/statistical method)', pct(int(df['ml_reclassified_from_yes'].sum()))),
    ('Studies with raw data available', pct(int((df['raw_data_clean']=='Yes').sum()))),
    ('Studies with code available', pct(int((df['code_clean']=='Yes').sum()))),
    ('Studies with clearly reported pipeline (re-derived from free text)', pct(int((df['pipeline_clean']=='Yes (clearly described)').sum()))),
    ('Methodology/pipeline-only studies (no surveillance outcome)', pct(int(df['is_methodology_only'].sum()))),
    ('Studies citing hmmIBD/Dcifer/MOIRE/THE REAL McCOIL', pct(12)),
]
pd.DataFrame(overview_rows, columns=['Characteristic','Value']).to_csv(f'{OUT}/table_overview.csv', index=False)

import analysis as A

def counter_to_df(counter, name, denom=77):
    d = pd.DataFrame(counter.most_common(), columns=[name,'n_studies'])
    d['pct_of_N'] = (100*d['n_studies']/denom).round(1)
    return d

counter_to_df(A.gene_counter, 'Gene/locus').to_csv(f'{OUT}/table_genes.csv', index=False)
counter_to_df(A.drug_counter, 'Drug').to_csv(f'{OUT}/table_drugs.csv', index=False)
counter_to_df(A.drugclass_counter, 'Drug class').to_csv(f'{OUT}/table_drug_class.csv', index=False)
counter_to_df(A.detect_bucketed, 'Detection method (scope)').to_csv(f'{OUT}/table_detection.csv', index=False)
counter_to_df(A.cat_bucketed, 'Study category').to_csv(f'{OUT}/table_study_category.csv', index=False)
counter_to_df(A.species_bucketed, 'Plasmodium species').to_csv(f'{OUT}/table_species.csv', index=False)
counter_to_df(A.design_bucketed, 'Study design').to_csv(f'{OUT}/table_design.csv', index=False)
counter_to_df(A.feat_bucketed, 'Feature type').to_csv(f'{OUT}/table_feature_type.csv', index=False)
counter_to_df(A.analysis_bucketed, 'Analysis approach').to_csv(f'{OUT}/table_analysis_type.csv', index=False)
counter_to_df(A.macro_counts, 'Macro-region').to_csv(f'{OUT}/table_macro_region.csv', index=False)
counter_to_df(A.software_counter, 'Software/tool').to_csv(f'{OUT}/table_software.csv', index=False)
counter_to_df(A.ml_model_counter, 'ML model (predictive only)').to_csv(f'{OUT}/table_ml_models.csv', index=False)

print('Tables exported to', OUT)
import sensitivity_analysis
import os
print(sorted(os.listdir(OUT)))
