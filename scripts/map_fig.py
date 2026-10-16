"""
Map figure function and properties.
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import DPI, FIG_DIR, VIRIDIS

WORLD_GEOJSON = '/users/chiomab/adr/world.geojson'

COUNTRY_ISO3 = {
    'Bangladesh': 'BGD', 'Brazil': 'BRA', 'Cambodia': 'KHM', 'Cameroon': 'CMR',
    'Central African Republic': 'CAF', 'China': 'CHN', 'Colombia': 'COL',
    'Democratic Republic of Congo': 'COD', 'Democratic Republic of the Congo': 'COD',
    'Democratic Republic of the Congo (DRC)': 'COD', 'Ethiopia': 'ETH',
    'France': 'FRA', 'France (imported from Africa)': 'FRA', 'French Guiana': 'GUF', 'Gabon': 'GAB',
    'Gambia': 'GMB', 'Germany': 'DEU', 'Ghana': 'GHA', 'Guinea-Bissau': 'GNB',
    'Guyana': 'GUY', 'Honduras': 'HND', 'India': 'IND', 'Indonesia': 'IDN',
    'Italy': 'ITA', 'Kenya': 'KEN', 'Laos': 'LAO', 'Madagascar': 'MDG',
    'Malawi': 'MWI', 'Malaysia': 'MYS', 'Mali': 'MLI', 'Mozambique': 'MOZ',
    'Myanmar': 'MMR', 'Netherlands': 'NLD', 'Nicaragua': 'NIC', 'Nigeria': 'NGA',
    'Pakistan': 'PAK', 'Panama': 'PAN', 'Papua New Guinea': 'PNG', 'Peru': 'PER',
    'Portugal': 'PRT', 'Republic of Congo': 'COG', 'Rwanda': 'RWA', 'Senegal': 'SEN',
    'South Korea': 'KOR', 'Spain': 'ESP', 'Sudan': 'SDN', 'Tanzania': 'TZA',
    'Thailand': 'THA', 'UK': 'GBR', 'UK (Software Development)': 'GBR', 'USA': 'USA',
    'Uganda': 'UGA', 'Vietnam': 'VNM', 'Zambia': 'ZMB', 'Burkina Faso': 'BFA',
    'Zimbabwe': 'ZWE', 'Ivory Coast': 'CIV', "Cote d'Ivoire": 'CIV', 'Angola': 'AGO',
    'Djibouti': 'DJI', 'Eritrea': 'ERI', 'Somalia': 'SOM',
}


def ensure_world_geojson(path):
    if os.path.exists(path):
        return path
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"WARNING: could not download world boundaries file ({e}). Skipping choropleth.")
        return None


def country_study_counts(df):
    from clean import split_multi
    counts = {}
    for cell in df['country']:
        toks = split_multi(cell)
        iso_seen = set()
        for t in toks:
            if t.startswith('Global') or t.startswith('Not applicable'):
                continue
            iso = COUNTRY_ISO3.get(t)
            if iso:
                iso_seen.add(iso)
        for iso in iso_seen:
            counts[iso] = counts.get(iso, 0) + 1
    return counts


def make_map(df, fname='fig03_map_choropleth.png'):
    world_path = ensure_world_geojson(WORLD_GEOJSON)
    if world_path is None:
        return None
    world = gpd.read_file(world_path)

    iso_counts = country_study_counts(df)
    cc = pd.Series(iso_counts, name='n_studies').rename_axis('ISO_A3').reset_index()
    world = world.merge(cc, on='ISO_A3', how='left')
    world['n_studies'] = world['n_studies'].fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7.4))
    zero = world[world['n_studies'] == 0]
    nonzero = world[world['n_studies'] > 0]

    zero.plot(ax=ax, color='#f2f2f0', edgecolor='#c9c9c9', linewidth=0.4)
    if len(nonzero):
        vmax = nonzero['n_studies'].max()
        nonzero.plot(ax=ax, column='n_studies', cmap=VIRIDIS, edgecolor='#c9c9c9', linewidth=0.4,
                     vmin=1, vmax=vmax, legend=True,
                     legend_kwds={'label': 'Number of Studies (Country of Sample Origin)',
                                  'shrink': 0.55, 'orientation': 'vertical', 'pad': 0.01})

    ax.set_xlim(-135, 165)
    ax.set_ylim(-58, 78)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/{fname}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return fname

