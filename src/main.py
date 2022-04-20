import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gp
import os
import numpy as np
import json
import math
import plotly.express as px
import sqlite3
# import kaleido
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fiona
pd.set_option('display.max_columns',10)

os.chdir('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/sources/')


def calculate_aggregated_patterns(patterns_poi):
    '''
        Aggregates patterns into census tract level, scales home origins to the number of devices, and
        creates a closed system by removing visits from outside of the study area.

    '''

    patterns_aggr = patterns_poi.loc[:, ['GEOID', 'date_range_start', 'raw_visit_counts', 'raw_visitor_counts']].groupby(['GEOID', 'date_range_start']).sum()

    patterns_aggr['visitor_home_ct'] = np.nan
    patterns_aggr['visitor_home_ct_count'] = np.nan

    patterns_aggr.reset_index(inplace=True)

    # Merge all home locations and group_by tract
    j = 0
    for ct in patterns_poi['GEOID'].unique():
        print(ct)
        for w in patterns_poi['date_range_start'].unique():
            homes = patterns_poi[(patterns_poi['GEOID'] == ct)&(patterns_poi['date_range_start'] == w)]
            aux = pd.DataFrame(columns=['index', 'count'])
            for idx, row in homes.iterrows():
                a = pd.DataFrame.from_dict(json.loads(row['visitor_home_cbgs']), orient='index').reset_index()
                a.rename({0: 'count'}, axis=1, inplace=True)
                aux = pd.concat([aux, a], axis=0)

            aux.set_index('index', inplace=True)
            aux.index = aux.index.str[:-1]

            aux['count'] = aux['count'].astype(int)
            #         aux['count'] = aux['count'] * (patterns_aggr.loc[ct, 'raw_visitor_counts'] / aux['count'].sum()).round()

            mask = (patterns_aggr['GEOID']==ct)&(patterns_aggr['date_range_start']==w)
            patterns_aggr.loc[mask, 'visitor_home_ct_count'] = aux['count'].sum()
            patterns_aggr.loc[mask, 'visitor_home_ct'] = json.dumps(aux.groupby('index').sum().to_dict()['count'])
            j = j + 1

    return patterns_aggr

def calculate_average_patterns(df):
    '''
        Aggregates patterns into census tract level, scales home origins to the number of devices, and
        creates a closed system by removing visits from outside of the study area.

    '''

    patterns_avg = df.loc[:, ['GEOID', 'raw_visit_counts', 'raw_visitor_counts']].groupby(['GEOID']).mean()

    patterns_avg['visitor_home_ct_avg'] = np.nan

    patterns_avg.reset_index(inplace=True)

    # Merge all home locations and group_by tract
    j = 0
    for ct in df['GEOID'].unique():
        homes = df[df['GEOID'] == ct]
        aux = pd.DataFrame(columns=['index', 'avg'])
        for idx, row in homes.iterrows():
            a = pd.DataFrame.from_dict(json.loads(row['visitor_home_ct']), orient='index').reset_index()
            a.rename({0: 'avg'}, axis=1, inplace=True)
            aux = pd.concat([aux, a], axis=0)

        aux['avg'] = aux['avg'].astype(int)
        #         aux['count'] = aux['count'] * (patterns_aggr.loc[ct, 'raw_visitor_counts'] / aux['count'].sum()).round()

        mask = patterns_avg['GEOID']==ct
        patterns_avg.loc[mask, 'visitor_home_ct_avg'] = json.dumps(aux.groupby('index').mean().to_dict()['avg'])
        j = j + 1

    return patterns_avg

county_fips = ['42027', '42033', '42035', '42119', '42087', '42061', '42013']
county_nms = ['Centre County', 'Clearfield County', 'Clinton County', 'Union County', 'Mifflin County', 'Huntingdon County', 'Blair County']

centre_student_tractcode=[
"42027012000",
"42027012100",
"42027012200",
"42027012400",
"42027012500",
"42027012600"
]

centre_nonstudent_tractcode = [
 '42027010100',
 '42027010200',
 '42027010300',
 '42027010400',
 '42027010500',
 '42027010600',
 '42027010700',
 '42027010800',
 '42027010900',
 '42027011000',
 '42027011100',
 '42027011201',
 '42027011300',
 '42027011400',
 '42027011501',
 '42027011502',
 '42027011600',
 '42027011702',
 '42027011800',
 '42027011901',
 '42027011902',
 '42027012700',
 '42027012800',
 '42027981202']

# County populations
cols = ['ALT0E001', 'GEOID', 'STATE', 'COUNTY']
pop = pd.read_csv('census/attributes/nhgis0003_ds244_20195_tract.csv', usecols=cols, encoding="ISO-8859-1")

mask = (pop['STATE'] == 'Pennsylvania') & (pop['COUNTY'].isin(county_nms))
pop = pop.loc[mask, :]
pop['GEOID'] = pop['GEOID'].str[7:]

# Does not exist in SafeGraph
pop = pop[pop['GEOID']!='42027012300']

pop.loc[pop['COUNTY']=='Centre County', 'FIPS'] = county_fips[0]
pop.loc[pop['COUNTY']=='Clearfield County', 'FIPS'] = county_fips[1]
pop.loc[pop['COUNTY']=='Clinton County', 'FIPS'] = county_fips[2]
pop.loc[pop['COUNTY']=='Union County', 'FIPS'] = int(county_fips[3])
pop.loc[pop['COUNTY']=='Mifflin County', 'FIPS'] = county_fips[4]
pop.loc[pop['COUNTY']=='Huntingdon County', 'FIPS'] = county_fips[5]
pop.loc[pop['COUNTY']=='Blair County', 'FIPS'] = county_fips[6]

pop['FIPS'] = pop['FIPS'].astype('int')
pop_county = pop.groupby(['COUNTY', 'FIPS']).sum().reset_index()

pop['GEOID'] = pop['GEOID'].astype('str')
mask = pop['GEOID'].isin(centre_student_tractcode)
pop.loc[mask, ['Cohort']] = 'Centre County Student'
mask = pop['GEOID'].isin(centre_nonstudent_tractcode)
pop.loc[mask, ['Cohort']] = 'Centre County Community'
mask = pop['Cohort'].isna()
pop.loc[mask, ['Cohort']] = pop.loc[mask, 'COUNTY']

pop_cohort = pop.groupby(['Cohort']).sum().reset_index()[['Cohort', 'ALT0E001']]
pop_cohort.rename({'ALT0E001':'Population'}, axis=1, inplace=True)
aux = pd.DataFrame([['Centre County', (129294 + 30494)]], columns=list(['Cohort', 'Population']))
pop_cohort = pop_cohort.append(aux, ignore_index=True)

# Here we read all patterns
path = "PA_counties_weekly_patterns_20180101-20220124/PA_counties_weekly_patterns/"
dirs = os.listdir(path)

df = pd.DataFrame(columns=['placekey','parent_placekey','location_name','street_address','city','region','postal_code',
                           'iso_country_code','safegraph_brand_ids','brands','date_range_start', 'date_range_end',
                           'raw_visit_counts','raw_visitor_counts','visits_by_day','visits_by_each_hour','poi_cbg',
                           'visitor_home_cbgs','visitor_home_aggregation','visitor_daytime_cbgs','visitor_country_of_origin',
                           'distance_from_home','median_dwell','bucketed_dwell_times','related_same_day_brand',
                           'related_same_week_brand','device_type','carrier_name','County'])
for folder in dirs:
    p = path + folder + '/'
    f = os.listdir(p)
    for item in f:
        print(item)
        df = pd.concat([df, pd.read_csv(p + item)], axis=0)
df = pd.read_csv('all.csv', low_memory=False)
df.drop('Unnamed: 0', axis=1, inplace=True)

# Here, we need POI data for the desired county:
census_tracts = gp.read_file('census/gis/ct_counties.shp')
naics = pd.read_csv("PA_counties_weekly_patterns_20180101-20220124/PA_POI_20220108.csv")
naics = gp.GeoDataFrame(naics, geometry=gp.points_from_xy(naics.longitude, naics.latitude))
naics.set_crs('EPSG:4326', inplace=True)
naics.to_crs(census_tracts.crs, inplace=True)
naics_gdf = gp.sjoin(naics, census_tracts.loc[:, ['GEOID', 'geometry']], how='left')
naics_gdf = naics_gdf[~naics_gdf['GEOID'].isna()]

naics_sub = naics_gdf[['placekey', 'top_category', 'sub_category', 'naics_code', 'GEOID']]

patterns_poi = df.merge(naics_sub,on='placekey')

# Count pois
ct_poi = patterns_poi.loc[:, ['GEOID', 'raw_visit_counts']].groupby('GEOID').count()
ct_poi.rename({'raw_visit_counts':'poi_counts'}, axis=1, inplace=True)

# Businesses:
cols = ['NAICS Code', 'May Continue Physical Operations']
bs = pd.read_csv("essential businesses2.csv", usecols=cols)
bs = bs.loc[bs['May Continue Physical Operations']=='Yes', ['NAICS Code']]

# Pre-pandemic: ‘2019-09-01’ to ‘2020-03-09’: 191 days
mask = (patterns_poi['date_range_start']>='2019-09-01')&(patterns_poi['date_range_start']<='2020-03-09')
patterns_poi.loc[mask, 'Period'] = 'Pre-pandemic'
# Social Distancing: ‘2020-03-10’ to ‘2020-09-01’: 175 days
mask = (patterns_poi['date_range_start']>='2020-03-10')&(patterns_poi['date_range_start']<='2020-09-01')
patterns_poi.loc[mask, 'Period'] = 'Social distancing'
# Post Social Distancing 1: ‘2020-09-01’ to ‘2021-05-15’: 256 days
mask = (patterns_poi['date_range_start']>='2020-09-01')&(patterns_poi['date_range_start']<='2021-05-15')
patterns_poi.loc[mask, 'Period'] = 'Post-pandemic 1'
# Post Social Distancing 2: ‘2021-05-15’ t o ‘2022-01-25’: 255 days
mask = (patterns_poi['date_range_start']>='2021-05-15')&(patterns_poi['date_range_start']<='2022-01-25')
patterns_poi.loc[mask, 'Period'] = 'Post-pandemic 2'

patterns_poi = patterns_poi[(~patterns_poi['Period'].isna())&(patterns_poi['poi_cbd'].astype(str).str[:5]=='42027')]

all_patterns = patterns_poi
essential_patterns = patterns_poi[(patterns_poi['naics_code'].astype(str).str[:4].isin(bs['NAICS Code'].astype(str)))|(patterns_poi['naics_code'].astype(str).str[:6].isin(bs['NAICS Code'].astype(str)))]
non_essential_patterns = patterns_poi[(~patterns_poi['naics_code'].astype(str).str[:4].isin(bs['NAICS Code'].astype(str)))&(~patterns_poi['naics_code'].astype(str).str[:6].isin(bs['NAICS Code'].astype(str)))]

# # Calculate aggregated patterns
all_patterns_aggr = calculate_aggregated_patterns(all_patterns)
all_patterns_aggr.to_csv('all_patterns_aggr.csv')
essential_patterns_aggr = calculate_aggregated_patterns(essential_patterns)
essential_patterns_aggr.to_csv('essential_patterns_aggr.csv')
non_essential_patterns_aggr = calculate_aggregated_patterns(non_essential_patterns)
non_essential_patterns_aggr.to_csv('non_essential_patterns_aggr.csv')

all_patterns_aggr['GEOID'] = all_patterns_aggr['GEOID'].astype(str)
essential_patterns_aggr['GEOID'] = essential_patterns_aggr['GEOID'].astype(str)
non_essential_patterns_aggr['GEOID'] = non_essential_patterns_aggr['GEOID'].astype(str)

all_patterns_aggr['County'] = all_patterns_aggr['GEOID'].astype(str).str[:5]
fig = px.bar(all_patterns_aggr.groupby(['County', 'date_range_start']).sum().reset_index(), x="date_range_start", y="raw_visitor_counts", title='All businesses', color='County')
fig.show()

essential_patterns_aggr['County'] = essential_patterns_aggr['GEOID'].astype(str).str[:5]
fig = px.bar(essential_patterns_aggr.groupby(['County', 'date_range_start']).sum().reset_index(), x="date_range_start", y="raw_visitor_counts", title='Essential businesses', color='County')
fig.show()

non_essential_patterns_aggr['County'] = non_essential_patterns_aggr['GEOID'].astype(str).str[:5]
fig = px.bar(non_essential_patterns_aggr.groupby(['County', 'date_range_start']).sum().reset_index(), x="date_range_start", y="raw_visitor_counts", title='None-essential businesses', color='County')
fig.show()

# Business Closures:
# Red: 40 days (28 March–7 May)
# Yellow: 20 days (8 May –28 May)
# Green: 29 May-now

# patterns_avg = calculate_average_patterns(patterns_poi)
# patterns_avg.to_csv('patterns_avg.csv', index=False)

patterns_poi =pd.read_csv('patterns_avg.csv')

# Distances
distances = gp.GeoDataFrame(census_tracts.set_index('GEOID').centroid)
distances = distances[distances.index.isin(df_pre_avg.index)]
distances = gp.GeoDataFrame(distances)
distances.rename({0:'geometry'}, axis=1, inplace=True)
distances.to_crs(census_tracts.crs, inplace=True)

dists = pd.DataFrame(index = list(distances.index), columns = list(distances.index))
for r, row in distances.iterrows():
    for c, col in distances.iterrows():
        dists.loc[r, c] = row['geometry'].distance(col['geometry'])

dist = dists.stack().reset_index()
dist.columns = ['Origin', 'Destination', 'Distance']

# Calculate od matrices
od_list = pd.DataFrame(columns=['Period', 'Origin', 'Destination', 'Flow'])
for p in patterns_poi['Period'].unique():
    aux0 = patterns_poi[patterns_poi['Period']==p]
    od = pd.DataFrame(index=list(aux0.GEOID), columns=list(aux0.GEOID), dtype='float')
    aux0.set_index('GEOID', inplace=True)
    for idx, row in aux0.iterrows():
        if len(json.loads(row['visitor_home_cbg_avg'])) != 0:
            aux1 = pd.DataFrame.from_dict(json.loads(aux0.loc[idx, 'visitor_home_cbg_avg']), orient='index')
            aux1 = aux1[aux1.index.str[:5].isin(county_fips)]
            aux1.rename({0: 'avg'}, axis=1, inplace=True)
            aux1.index = aux1.index.astype('int64')
            mask = od.index.isin([int(i) for i in aux1.index])
            od.loc[mask, idx] = aux1.loc[:,'avg']

    od = od.fillna(0)
    aux2 = od.stack().reset_index()
    aux2.columns = ['Origin', 'Destination', 'Flow']

    aux2 = aux2.merge(pop, left_on='Origin', right_on='GEOID', how='left')
    aux2.rename({'STATE': 'Origin state', 'COUNTY': 'Origin county', 'ALT0E001': 'Origin population',
                    'FIPS': 'Origin FIPS','Cohort': 'Origin Cohort'}, axis=1, inplace=True)
    aux2.drop(['GEOID'], axis=1, inplace=True)

    aux2 = aux2.merge(pop, left_on='Destination', right_on='GEOID', how='left')
    aux2.rename(
        {'STATE': 'Destination state', 'COUNTY': 'Destination county', 'ALT0E001': 'Destination population',
         'FIPS': 'Destination FIPS', 'Cohort': 'Destination Cohort'}, axis=1, inplace=True)
    aux2.drop(['GEOID'], axis=1, inplace=True)

    aux2 = aux2.merge(ct_poi, left_on='Origin', right_on=ct_poi.index, how='left')
    aux2.rename({'poi_counts': 'Origin poi counts'}, axis=1, inplace=True)

    aux2 = aux2.merge(ct_poi, left_on='Destination', right_on=ct_poi.index, how='left')
    aux2.rename({'poi_counts': 'Destination poi counts'}, axis=1, inplace=True)

    mask = (aux2['Destination Cohort'] != 'Centre County Community') & (
                aux2['Destination Cohort'] != 'Centre County Student')
    aux2.loc[mask, 'Destination Cohort 2'] = 'Outside Centre County'
    mask = (aux2['Origin Cohort'] != 'Centre County Community') & (
                aux2['Origin Cohort'] != 'Centre County Student')
    aux2.loc[mask, 'Origin Cohort 2'] = 'Outside Centre County'
    mask = aux2['Destination Cohort 2'].isna()
    aux2.loc[mask, 'Destination Cohort 2'] = aux2.loc[mask, 'Destination Cohort']
    mask = aux2['Origin Cohort 2'].isna()
    aux2.loc[mask, 'Origin Cohort 2'] = aux2.loc[mask, 'Origin Cohort']

    aux2['Period']=p

    od_list = pd.concat([od_list, aux2], axis=0)

od_list.to_csv('od_list.csv', index=False)
od_list = pd.read_csv('od_list.csv')




