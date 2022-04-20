import pandas as pd
import geopandas as gp
import os
import numpy as np
import json
import plotly.express as px
import ast

pd.set_option('display.max_columns', 10)

os.chdir('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/sources/')

county_fips = ['42027', '42033', '42035', '42119', '42087', '42061', '42013']
county_nms = ['Centre County', 'Clearfield County', 'Clinton County', 'Union County', 'Mifflin County',
              'Huntingdon County', 'Blair County']

centre_student_tractcode = [
    "420270120001",
    "420270120002",
    "420270120003",
    "420270120004",
    "420270120005",
    "420270121001",
    "420270121002",
    "420270121003",
    "420270121004",
    "420270122001",
    "420270122002",
    "420270122003",
    "420270124001",
    "420270124002",
    "420270124003",
    "420270125001",
    "420270125002",
    "420270126001",
    "420270126002",
]

# County populations
cols = ['ALT0E001', 'GEOID', 'STATE', 'COUNTY']
pop = pd.read_csv('census/attributes/nhgis0003_ds244_20195_blck_grp.csv', usecols=cols, encoding="ISO-8859-1")

mask = (pop['STATE'] == 'Pennsylvania') & (pop['COUNTY'].isin(county_nms))
pop = pop.loc[mask, :]
pop['GEOID'] = pop['GEOID'].str[7:]

# Does not exist in SafeGraph
# pop = pop[pop['GEOID'].str[:-1] != '42027012300']

pop.loc[pop['COUNTY'] == 'Centre County', 'FIPS'] = county_fips[0]
pop.loc[pop['COUNTY'] == 'Clearfield County', 'FIPS'] = county_fips[1]
pop.loc[pop['COUNTY'] == 'Clinton County', 'FIPS'] = county_fips[2]
pop.loc[pop['COUNTY'] == 'Union County', 'FIPS'] = int(county_fips[3])
pop.loc[pop['COUNTY'] == 'Mifflin County', 'FIPS'] = county_fips[4]
pop.loc[pop['COUNTY'] == 'Huntingdon County', 'FIPS'] = county_fips[5]
pop.loc[pop['COUNTY'] == 'Blair County', 'FIPS'] = county_fips[6]

pop['FIPS'] = pop['FIPS'].astype('int')
pop_county = pop.groupby(['COUNTY', 'FIPS']).sum().reset_index()

pop['GEOID'] = pop['GEOID'].astype('str')
mask = pop['GEOID'].isin(centre_student_tractcode)
pop.loc[mask, ['Cohort']] = 'Centre County Student'
mask = (~pop['GEOID'].isin(centre_student_tractcode)) & (pop['COUNTY'] == 'Centre County')
pop.loc[mask, ['Cohort']] = 'Centre County Community'
mask = pop['Cohort'].isna()
pop.loc[mask, ['Cohort']] = pop.loc[mask, 'COUNTY']

pop_cohort = pop.groupby(['Cohort']).sum().reset_index()[['Cohort', 'ALT0E001']]
pop_cohort.rename({'ALT0E001': 'Population'}, axis=1, inplace=True)
aux = pd.DataFrame([['Centre County', (129294 + 30494)]], columns=list(['Cohort', 'Population']))
pop_cohort = pop_cohort.append(aux, ignore_index=True)

# # Here we read all patterns
# path = "PA_counties_weekly_patterns_20180101-20220124/PA_counties_weekly_patterns/"
# dirs = os.listdir(path)
#
# df = pd.DataFrame(columns=['placekey','parent_placekey','location_name','street_address','city','region','postal_code',
#                            'iso_country_code','safegraph_brand_ids','brands','date_range_start', 'date_range_end',
#                            'raw_visit_counts','raw_visitor_counts','visits_by_day','visits_by_each_hour','poi_cbg',
#                            'visitor_home_cbgs','visitor_home_aggregation','visitor_daytime_cbgs','visitor_country_of_origin',
#                            'distance_from_home','median_dwell','bucketed_dwell_times','related_same_day_brand',
#                            'related_same_week_brand','device_type','carrier_name','County'])
# for folder in dirs:
#     p = path + folder + '/'
#     f = os.listdir(p)
#     for item in f:
#         print(item)
#         df = pd.concat([df, pd.read_csv(p + item)], axis=0)
#
# df.to_csv('all.csv', index=False)
df = pd.read_csv('all.csv', low_memory=False)

# Here, we need POI data for the desired county:
blockgroups = gp.read_file('census/gis/bg_counties.shp')
naics = pd.read_csv("PA_counties_weekly_patterns_20180101-20220124/PA_POI_20220108.csv")
naics = gp.GeoDataFrame(naics, geometry=gp.points_from_xy(naics.longitude, naics.latitude))
naics.set_crs('EPSG:4326', inplace=True)
naics.to_crs(blockgroups.crs, inplace=True)
naics_gdf = gp.sjoin(naics, blockgroups.loc[:, ['GEOID', 'geometry']], how='left')
naics_gdf = naics_gdf[~naics_gdf['GEOID'].isna()]

naics_sub = naics_gdf[['placekey', 'top_category', 'sub_category', 'naics_code', 'GEOID']]

patterns_poi = df.merge(naics_sub, on='placekey')

# Count pois
ct_poi = patterns_poi.loc[:, ['GEOID', 'raw_visit_counts']].groupby('GEOID').count()
ct_poi.rename({'raw_visit_counts': 'poi_counts'}, axis=1, inplace=True)

# # Calculate aggregated patterns
patterns_poi['GEOID'] = patterns_poi['GEOID'].astype(str)

patterns_poi['County'] = patterns_poi['GEOID'].astype(str).str[:5]
aux = patterns_poi.groupby(['County', 'date_range_start']).sum().reset_index()
aux1 = aux[aux['County']=='42027']

# Business Closures:
# Red: 40 days (28 March–7 May)
# Yellow: 20 days (8 May –28 May)
# Green: 29 May-now

# Pre-pandemic: ‘2019-10-10’ to ‘2020-03-09’: 191 days
mask = (patterns_poi['date_range_start'] >= '2019-10-10') & (patterns_poi['date_range_start'] <= '2020-03-09')
patterns_poi.loc[mask, 'Period'] = 'Pre-pandemic'
# Social Distancing: ‘2020-03-09’ to ‘2020-09-01’: 175 days
mask = (patterns_poi['date_range_start'] >= '2020-03-09') & (patterns_poi['date_range_start'] <= '2020-09-01')
patterns_poi.loc[mask, 'Period'] = 'Social distancing'
# Post Social Distancing 1: ‘2020-09-01’ to ‘2021-05-15’: 256 days
mask = (patterns_poi['date_range_start'] >= '2020-09-01') & (patterns_poi['date_range_start'] <= '2021-05-15')
patterns_poi.loc[mask, 'Period'] = 'Post-pandemic 1'
# Post Social Distancing 2: ‘2021-05-15’ t o ‘2022-01-25’: 255 days
mask = (patterns_poi['date_range_start'] >= '2021-05-15') & (patterns_poi['date_range_start'] <= '2022-01-25')
patterns_poi.loc[mask, 'Period'] = 'Post-pandemic 2'

patterns_poi = patterns_poi[~patterns_poi['Period'].isna()]
patterns_poi = patterns_poi[patterns_poi['GEOID'].astype(str).str[:5] == '42027']

for idx, row in patterns_poi.iterrows():
    patterns_poi.loc[idx, 'visitor_home_cbgs_sum'] = sum(json.loads(row['visitor_home_cbgs']).values())

mask = patterns_poi['GEOID'].isin(centre_student_tractcode)
patterns_poi.loc[mask, 'cohort'] = 'Student'
mask = patterns_poi['GEOID'].isna()
patterns_poi.loc[mask, 'cohort'] = 'Community'

patterns_poi_cbg = patterns_poi.groupby(['date_range_start', 'cohort']).sum().reset_index()[['date_range_start', 'cohort', 'raw_visit_counts', 'raw_visitor_counts', 'visitor_home_cbgs_sum']]

fig = px.line(patterns_poi_cbg, x="date_range_start",
             y="raw_visit_counts", title='raw_visit_counts', color='cohort')
fig.add_vrect(x0='2019-10-10', x1='2020-03-09')
fig.add_annotation(x='2019-12-15', y=90000, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=90000, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=90000, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=90000, text="Post pandemic 2",showarrow=False)
fig.show()

fig = px.line(patterns_poi_cbg, x="date_range_start",
             y="raw_visitor_counts", title='raw_visitor_counts', color='cohort')
fig.add_vrect(x0='2019-10-10', x1='2020-03-09')
fig.add_annotation(x='2019-12-15', y=90000, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=90000, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=90000, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=90000, text="Post pandemic 2",showarrow=False)
fig.show()

fig = px.line(patterns_poi_cbg, x="date_range_start",
             y="visitor_home_cbgs_sum", title='visitor_home_cbgs_sum', color='cohort')
fig.add_vrect(x0='2019-10-10', x1='2020-03-09')
fig.add_annotation(x='2019-12-15', y=90000, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=90000, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=90000, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=90000, text="Post pandemic 2",showarrow=False)
fig.show()