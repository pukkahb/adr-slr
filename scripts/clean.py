"""
Cleaning / normalisation utilities for the extracted ADR systematic review
dataset (N=77 studies).
"""
import re
import pandas as pd
import numpy as np

NA_TOKENS = {
    'not reported', 'not applicable', 'na', 'n/a', 'none', 'nan', '',
    'not explicitly named', 'unknown', 'not specified',
}

def _clean_token(t):
    t = t.strip()
    t = t.strip(' .')
    return t

def clean_free_text(cell):
    """Pass-through for free-text fields (key_findings, limitations)."""
    if pd.isna(cell):
        return cell
    return str(cell).strip()

def split_multi(cell, drop_na=True):
    """Split a semicolon/comma-delimited free-text cell into clean tokens,
    respecting parenthesis nesting so that "Population structure analysis
    (PCA, ADMIXTURE, DAPC)" is kept as ONE token rather than being split on
    the commas inside the parentheses into broken fragments like
    "Admixture)"."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    parts = []
    buf = []
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ';' and depth == 0:
            parts.append(''.join(buf)); buf = []
        elif ch == ',' and depth == 0:
            # only split on a comma at depth 0 if not between two digits
            # (keeps "1,234" and similar numbers intact)
            prev_digit = i > 0 and s[i-1].isdigit()
            next_digit = i+1 < len(s) and s[i+1].isdigit()
            if prev_digit and next_digit:
                buf.append(ch)
            else:
                parts.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append(''.join(buf))
    out = []
    for p in parts:
        p = _clean_token(p)
        if not p:
            continue
        if drop_na and p.lower() in NA_TOKENS:
            continue
        out.append(p)
    return out

# ---------------------------------------------------------------------------
# Re-derived pipeline-reporting status
# ---------------------------------------------------------------------------
def derive_pipeline_status(bioinformatics_pipeline_cell):
    """
    Classify pipeline-reporting clarity directly from the free-text
    bioinformatics_pipeline field, rather than trusting the (artifactual)
    pipeline_reporting_status column.

    Rule (stated explicitly for transparency / reproducibility):
      - "Not applicable"  -> field is exactly 'Not applicable' (no
        bioinformatics pipeline exists for this study, e.g. pure wet-lab
        phenotyping with no sequence-analysis component)
      - "No"               -> field is 'Not reported' / missing / empty
      - "Yes (clearly described)" -> field contains a real, non-trivial
        description (e.g. a named tool, or a step-chain such as
        "BWA-MEM -> GATK -> SnpEff")
    """
    if pd.isna(bioinformatics_pipeline_cell):
        return 'No'
    s = str(bioinformatics_pipeline_cell).strip()
    s_low = s.lower()
    if s_low in ('not applicable', 'n/a', 'na'):
        return 'Not applicable'
    if s_low in ('not reported', 'unknown', 'not specified', ''):
        return 'No'
    return 'Yes (clearly described)'

# ---------------------------------------------------------------------------
# ML classification rule
# ---------------------------------------------------------------------------
# Predictive/trained ML: a model fit to map input features to a labelled or
# predicted outcome, used for classification, regression-as-prediction, or
# sequence/structure prediction. Excludes: unsupervised ordination/clustering
# used descriptively (PCA, PCoA, hierarchical clustering, DAPC, ADMIXTURE),
# and classical inferential statistics (logistic regression used for
# association/inference rather than out-of-sample prediction), which are
# retained under "analysis_type" as statistical/population-genetic methods,
# not machine learning.
ML_PREDICTIVE_KEYWORDS = [
    'random forest', 'support vector', 'svm', 'svc', 'neural network', 'cnn',
    'rnn', 'lstm', 'deep learning', 'deep neural', 'gradient boosting',
    'xgboost', 'decision tree', 'k-nearest', 'knn', 'naive bayes',
    'auto-weka', 'gaussian process', 'ensemble', 'bagging', 'sift',
    'mutpred', 'polyphen', 'clair3', 'alexnet', 'deepsweep',
]
ML_DESCRIPTIVE_EXCLUDE = [
    'principal component', 'principal coordinates', 'pcoa', 'hierarchical',
    'agglomerative clustering', 'dapc', 'admixture', 'logistic regression',
    'linear model', 'lasso regression', 'spline',
]

def classify_ml(ml_used_raw, ml_model_cell):
    """Returns 'Yes' only for studies using a genuinely predictive/trained
    model per ML_PREDICTIVE_KEYWORDS; reclassifies purely descriptive
    ordination/clustering or classical regression to 'No' even if the
    original extraction marked them 'Yes'."""
    if pd.isna(ml_model_cell):
        model_text = ''
    else:
        model_text = str(ml_model_cell).lower()
    has_predictive = any(k in model_text for k in ML_PREDICTIVE_KEYWORDS)
    only_descriptive = (not has_predictive) and any(k in model_text for k in ML_DESCRIPTIVE_EXCLUDE)
    raw_yes = str(ml_used_raw).strip().lower().startswith('yes')
    if raw_yes and has_predictive:
        return 'Yes'
    if raw_yes and only_descriptive:
        return 'No (descriptive/statistical method, reclassified)'
    if raw_yes:
        return 'Yes'  
    return 'No'


def canonical_region(token):
    return token.strip()

MACRO_MAP = {
    'East Africa': 'Africa', 'Central Africa': 'Africa', 'West Africa': 'Africa',
    'South Africa': 'Africa', 'North Africa': 'Africa', 'North-East Africa': 'Africa',
    'Sub-Saharan Africa': 'Africa', 'Africa (unspecified)': 'Africa',
    'South-East Asia': 'Asia', 'South Asia': 'Asia', 'East Asia': 'Asia',
    'West Asia': 'Asia', 'Central Asia': 'Asia', 'Asia (unspecified)': 'Asia',
    'Oceania': 'Oceania',
    'South America': 'Americas', 'Central America': 'Americas', 'North America': 'Americas',
    'Caribbean': 'Americas',
    'Northern Europe': 'Europe', 'Southern Europe': 'Europe', 'Western Europe': 'Europe',
    'Eastern Europe': 'Europe', 'Europe (unspecified)': 'Europe',
    'Global': 'Global / Multi-region', 'Not applicable': 'Not applicable',
}

def macro_region(region_token):
    r = canonical_region(region_token)
    return MACRO_MAP.get(r, r)

GENE_MAP = {
    'k13': 'Kelch13 (K13)', 'kelch13': 'Kelch13 (K13)', 'pfkelch13': 'Kelch13 (K13)', 'pfk13': 'Kelch13 (K13)',
    'crt': 'CRT', 'crt-o': 'CRT', 'mdr1': 'MDR1', 'dhfr': 'DHFR', 'dhps': 'DHPS', 'dhodh': 'DHODH',
    'coronin': 'Coronin', 'ubp1': 'UBP1', 'ap2u': 'AP2-mu', 'ap2-mu': 'AP2-mu',
    'ama1': 'AMA1', 'msp1': 'MSP1', 'msp2': 'MSP2', 'msp3': 'MSP3', 'glurp': 'GLURP',
    'csp': 'CSP', 'trap': 'TRAP', 'hrp2': 'HRP2/3', 'hrp2/3': 'HRP2/3', 'hrp3': 'HRP2/3',
    'ldh': 'LDH', 'gch1': 'GCH1', 'atp6': 'ATPase6 (ATP6)', 'pm2': 'Plasmepsin 2/3 (PM2/3)',
    'pm3': 'Plasmepsin 2/3 (PM2/3)', 'plasmepsin2': 'Plasmepsin 2/3 (PM2/3)', 'plasmepsin3': 'Plasmepsin 2/3 (PM2/3)',
    'exo': 'Exonuclease (Exo-E415G)', 'eba175': 'EBA-175', 'dbp': 'DBP/RBP family',
    'dbp1': 'DBP/RBP family', 'ebp': 'DBP/RBP family', 'rbp2b': 'DBP/RBP family',
}

def canonical_gene(token):
    t = token.strip()
    t_low = t.lower()
    t_low = re.sub(r'\(.*?\)', '', t_low)
    t_low = re.sub(r'^p[fvk]', '', t_low)
    t_low = re.sub(r'[\s\-_]', '', t_low).strip()
    for key, label in GENE_MAP.items():
        key_norm = key.replace('-', '').replace(' ', '')
        if t_low == key_norm:
            return label
    m = re.match(r'^(p[fvc])([a-z0-9]+)$', t, re.IGNORECASE)
    if m and ' ' not in t:
        return m.group(1)[0].upper() + m.group(1)[1].lower() + m.group(2).upper()
    if re.match(r'^[a-z]+$', t) and ' ' not in t:
        return t.capitalize()
    return t.strip()

def yesno_bucket(cell):
    if pd.isna(cell):
        return 'Not reported'
    s = str(cell).strip().lower()
    if s.startswith('yes'):
        return 'Yes'
    if s.startswith('no') and 'not' not in s[:2]:
        return 'No'
    if s in ('not reported', 'not specified', 'unknown'):
        return 'Not reported'
    if s == 'not applicable':
        return 'Not applicable'
    return 'Not reported'


DRUG_CANON = {
    'sulfadoxine-pyrimethamine (sp)': 'Sulfadoxine-Pyrimethamine',
    'sulfadoxine-pyrimethamine': 'Sulfadoxine-Pyrimethamine',
    'artesunate / sulfadoxine-pyrimethamine (as/sp)': 'Artesunate-Sulfadoxine-Pyrimethamine',
}
def canonical_drug(token):
    t = token.strip()
    return DRUG_CANON.get(t.lower(), t)

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    
    if 'inclusion' in df.columns:
        df = df[df['inclusion'].astype(str).str.strip().str.lower() == 'yes'].reset_index(drop=True)

    df['pub_year'] = pd.to_numeric(df['year'], errors='coerce')

    df['region_list'] = df['region'].apply(lambda x: [canonical_region(t) for t in split_multi(x)])
    df['macro_region_list'] = df['region_list'].apply(lambda lst: sorted(set(macro_region(t) for t in lst)))
    df['is_africa'] = df['macro_region_list'].apply(lambda lst: 'Africa' in lst)

    df['lead_region_list'] = df['lead_author_affiliation_region'].apply(
        lambda x: [canonical_region(t) for t in split_multi(x)])
    df['lead_macro_region_list'] = df['lead_region_list'].apply(
        lambda lst: sorted(set(macro_region(t) for t in lst)))
    df['lead_is_africa'] = df['lead_macro_region_list'].apply(lambda lst: 'Africa' in lst)

    def clean_species(cell):
        toks = split_multi(cell)
        return [t.replace('P. ', 'Plasmodium ').strip() for t in toks]
    df['species_list'] = df['plasmodium_species'].apply(clean_species)

    df['gene_list'] = df['gene'].apply(lambda x: [canonical_gene(t) for t in split_multi(x)])
    df['drug_list'] = df['drug_name'].apply(lambda x: [canonical_drug(t) for t in split_multi(x)])
    df['drug_class_list'] = df['drug_class'].apply(split_multi)
    df['detection_list'] = df['detection_method'].apply(split_multi)
    df['analysis_list'] = df['analysis_type'].apply(split_multi)
    df['feature_list'] = df['feature_type'].apply(split_multi)
    df['software_list'] = df['software_tools'].apply(split_multi)
    df['study_category_list'] = df['study_category'].apply(split_multi)
    df['study_design_list'] = df['study_design'].apply(split_multi)

    # ---- ML: explicit re-classification rule ----
    df['ml_classified'] = df.apply(lambda r: classify_ml(r['ml_used'], r['ml_model']), axis=1)
    df['ml_used_clean'] = df['ml_classified'].apply(lambda x: 'Yes' if x == 'Yes' else 'No')
    df['ml_reclassified_from_yes'] = df['ml_classified'].str.startswith('No (descriptive')
    df['ml_model_list'] = df['ml_model'].apply(split_multi)

    # ---- reproducibility indicators ----
    df['raw_data_clean'] = df['raw_data_status'].apply(yesno_bucket)
    df['code_clean'] = df['code_status'].apply(yesno_bucket)

    # pipeline status: RE-DERIVED from free text, not trusted from extraction
    df['pipeline_clean'] = df['bioinformatics_pipeline'].apply(derive_pipeline_status)
    df['pipeline_status_was_reextracted'] = True  # flag for methods-section transparency

    df['multi_country_clean'] = df['multi_country'].apply(
        lambda x: 'Yes' if str(x).lower().startswith('yes') else ('No' if str(x).lower().startswith('no') else 'Not reported')
    )

    df['key_findings'] = df['key_findings'].apply(clean_free_text)
    df['limitations'] = df['limitations'].apply(clean_free_text)

    def extract_n(cell):
        if pd.isna(cell):
            return np.nan
        m = re.search(r'([\d,]{2,})', str(cell))
        if m:
            try:
                return int(m.group(1).replace(',', ''))
            except ValueError:
                return np.nan
        return np.nan
    df['sample_size_n'] = df['sample_size'].apply(extract_n)

    # ---- flag studies whose ONLY content is methodology/pipeline development
    def is_methodology_only(cat_list, design_list):
        cats = ' '.join(cat_list).lower()
        design = ' '.join(design_list).lower()
        is_method_cat = 'methodology' in cats or 'pipeline' in cats
        is_surveillance = 'surveillance' in cats or 'surveillance' in design
        return is_method_cat and not is_surveillance
    df['is_methodology_only'] = df.apply(
        lambda r: is_methodology_only(r['study_category_list'], r['study_design_list']), axis=1)

    return df

if __name__ == '__main__':
    df = load_clean('/data/adr.csv')
    print('N =', len(df))
    print('ML yes (reclassified):', (df['ml_used_clean']=='Yes').sum(), '/ raw Yes was:',
          (df['ml_used'].astype(str).str.lower().str.startswith('yes')).sum())
    print('Reclassified from Yes to No:', df['ml_reclassified_from_yes'].sum())
    print('Pipeline status (re-derived):')
    print(df['pipeline_clean'].value_counts())
    print('Methodology-only studies:', df['is_methodology_only'].sum())
