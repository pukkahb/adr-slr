"""
Supplementary statistical checks referenced in Methods §2.8 and Results §3.5:

  1. Chi-squared test of independence for each of the four methodological/
     reproducibility indicators (ML use, code availability, raw-data
     availability, pipeline-reporting clarity) across the three region
     strata (Africa-based / non-Africa single-region / global-multi-region).

  2. A sensitivity analysis repeating all four indicators under an
     alternative, coarser binary geographic classification (Africa-based
     vs. all other studies combined), to check whether the descriptive
     pattern reported is robust to the specific regional grouping chosen.
"""
import pandas as pd
from scipy import stats
from descriptive_proportions import wilson_ci

TAB_DIR = '/users/chiomab/adr/tables'

df = pd.read_pickle('/users/chiomab/adr/data/df_clean.pkl')
 
def region_group_3way(row):
    if row['is_africa']:
        return 'Africa-based'
    if 'Global / Multi-region' in row['macro_region_list']:
        return 'Global/multi-region'
    return 'Non-Africa single-region'
df['region_group_3way'] = df.apply(region_group_3way, axis=1)
df['region_group_binary'] = df['is_africa'].map(
    {True: 'Africa-based', False: 'Non-Africa (incl. global/multi-region)'})
 
INDICATORS = {
    'ML use': ('ml_used_clean', 'Yes'),
    'Code available': ('code_clean', 'Yes'),
    'Raw data available': ('raw_data_clean', 'Yes'),
    'Pipeline clearly reported': ('pipeline_clean', 'Yes (clearly described)'),
}
 
# ---------------------------------------------------------------------
# 1. Chi-squared test of independence (3-way region classification)
# ---------------------------------------------------------------------
order_3way = ['Africa-based', 'Non-Africa single-region', 'Global/multi-region']
chi2_rows = []
for name, (col, yes_val) in INDICATORS.items():
    flag = df[col] == yes_val
    ct = pd.crosstab(df['region_group_3way'], flag).reindex(index=order_3way)
    ct = ct.reindex(columns=[True, False], fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    chi2_rows.append({
        'Indicator': name,
        'χ²': round(chi2, 3),
        'df': dof,
        'p-value': round(p, 4),
    })
chi2_df = pd.DataFrame(chi2_rows)
chi2_df.to_csv(f'{TAB_DIR}/table_chisquared_region.csv', index=False)
print('=== Chi-squared test of independence across 3 region strata ===')
print(chi2_df.to_string(index=False))
print()
 
# ---------------------------------------------------------------------
# 2. Sensitivity analysis: alternative binary geographic classification
# ---------------------------------------------------------------------
sens_rows = []
for name, (col, yes_val) in INDICATORS.items():
    row = {'Indicator': name}
    for grouping_name, grouping_col, order in [
        ('3-way (primary)', 'region_group_3way', order_3way),
        ('Binary (sensitivity)', 'region_group_binary',
         ['Africa-based', 'Non-Africa (incl. global/multi-region)']),
    ]:
        for g in order:
            sub = df[df[grouping_col] == g]
            n = len(sub)
            ev = int((sub[col] == yes_val).sum())
            p, lo, hi = wilson_ci(ev, n)
            row[f'{grouping_name}: {g}'] = f'{ev}/{n} = {100*p:.1f}% [{100*lo:.1f}-{100*hi:.1f}%]'
    sens_rows.append(row)
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(f'{TAB_DIR}/table_geographic_sensitivity.csv', index=False)
print('=== Geographic classification sensitivity analysis ===')
print(sens_df.to_string(index=False))
 
# ---------------------------------------------------------------------
# 3. ML classification sensitivity analysis (broad vs narrow definition)
# ---------------------------------------------------------------------
raw_yes = df['ml_used'].astype(str).str.lower().str.startswith('yes')
narrow_yes = df['ml_used_clean'] == 'Yes'
ml_sens = pd.DataFrame([
    {'Definition': 'Broad (original extraction, includes descriptive/ordination/clustering methods)',
     'ML Usage': int(raw_yes.sum()), 'N': len(df), '%': round(100*raw_yes.sum()/len(df), 1)},
    {'Definition': 'Narrow (predictive-model-only rule, used throughout manuscript)',
     'ML Usage': int(narrow_yes.sum()), 'N': len(df), '%': round(100*narrow_yes.sum()/len(df), 1)},
])
ml_sens.to_csv(f'{TAB_DIR}/table_ml_definition_sensitivity.csv', index=False)
print()
print('=== ML classification sensitivity analysis ===')
print(ml_sens.to_string(index=False))

