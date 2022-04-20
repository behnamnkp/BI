import pandas as pd
import geopandas as gp
import os
import numpy as np
import json
import plotly.express as px
import ast
import statsmodels.api as sm

pd.set_option('display.max_columns', 10)

os.chdir('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/sources/')

def calculate_average_patterns(patterns_poi):

    """
        Aggregates patterns into cbg level and averages them over the periods.
        :param patterns_poi: The raw foot traffic data
        :return: Aggreagted, averaged patterns
    """

    patterns_avg = patterns_poi.loc[:,
                     ['Period', 'GEOID', 'date_range_start', 'raw_visit_counts', 'raw_visitor_counts']].groupby(
        ['Period', 'date_range_start', 'GEOID']).sum().reset_index()
    patterns_avg = patterns_avg.loc[:,
                     ['Period', 'GEOID', 'date_range_start', 'raw_visit_counts', 'raw_visitor_counts']].groupby(
        ['Period', 'GEOID']).mean().reset_index()

    patterns_avg['visitor_home_cbg_avg'] = np.nan

    # Merge all home locations
    for p in patterns_poi['Period'].unique():
        for ct in patterns_poi['GEOID'].unique():
            aux1 = pd.DataFrame(columns=['index', 'avg'])
            for d in patterns_poi['date_range_start'].unique():
                print(p + ', ' + d + ', ' + str(ct))
                homes = patterns_poi[(patterns_poi['Period'] == p) & (patterns_poi['date_range_start'] == d) & (patterns_poi['GEOID'] == ct)]
                if homes.shape[0] != 0:
                    aux = pd.DataFrame(columns=['index', 'count'])
                    for idx, row in homes.iterrows():
                        a = pd.DataFrame.from_dict(json.loads(row['visitor_home_cbgs']), orient='index').reset_index()
                        if a.shape[0] != 0:
                            a.rename({0: 'count'}, axis=1, inplace=True)
                            aux = pd.concat([aux, a], axis=0)

                    aux['count'] = aux['count'].astype(int)
                    aux = aux[aux['index'].astype(str).str[:5]=='42027']
                    aux = aux.groupby('index').sum()
                    aux.reset_index(inplace=True)
                    aux.rename({'count': 'avg'}, inplace=True, axis=1)
                    aux1 = pd.concat([aux1, aux], axis=0)

            if aux1.shape[0] != 0:
                mask = (patterns_avg['Period'] == p) & (patterns_avg['GEOID'] == ct)
                patterns_avg.loc[mask, 'visitor_home_cbg_avg'] = json.dumps(aux1.groupby('index').mean().to_dict()['avg'])

    return patterns_avg

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
mask = patterns_poi['poi_cbg'].astype('int64').astype(str).isin(centre_student_tractcode)
patterns_poi.loc[mask, 'cbg_cohort'] = 'Centre-student'
mask = (patterns_poi['County'] == '42027') & (~patterns_poi['poi_cbg'].astype('int64').astype(str).isin(centre_student_tractcode))
patterns_poi.loc[mask, 'cbg_cohort'] = 'Centre-community'
mask = (patterns_poi['cbg_cohort'].isna())
patterns_poi.loc[mask, 'cbg_cohort'] = patterns_poi.loc[mask, 'County']

aux = patterns_poi.groupby(['cbg_cohort', 'date_range_start']).sum().reset_index()
# aux1 = aux[aux['County']=='42027']

m = aux.raw_visitor_counts.max()
fig = px.line(aux, x="date_range_start",
             y="raw_visitor_counts", title='Visitor counts', color='cbg_cohort')
fig.add_vrect(x0='2019-10-10', x1='2020-03-09')
fig.add_annotation(x='2019-12-01', y=m, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=m, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=m, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=m, text="Post pandemic 2",showarrow=False)
fig.show()

# Business Closures:
# Red: 40 days (28 March–7 May)
# Yellow: 20 days (8 May –28 May)
# Green: 29 May-now

# Pre-pandemic: ‘2019-10-10’ to ‘2020-03-09’:
mask = (patterns_poi['date_range_start'] >= '2019-10-10') & (patterns_poi['date_range_start'] <= '2020-03-09')
patterns_poi.loc[mask, 'Period'] = 'Pre-pandemic'
# Social Distancing: ‘2020-03-10’ to ‘2020-09-01’:
mask = (patterns_poi['date_range_start'] >= '2020-03-10') & (patterns_poi['date_range_start'] <= '2020-05-25')
patterns_poi.loc[mask, 'Period'] = 'Social distancing'
# Post Social Distancing 1: ‘2020-09-01’ to ‘2021-05-15’:
mask = (patterns_poi['date_range_start'] >= '2020-05-25') & (patterns_poi['date_range_start'] <= '2021-05-15')
patterns_poi.loc[mask, 'Period'] = 'Post-pandemic 1'
# Post Social Distancing 2: ‘2021-05-15’ t o ‘2022-01-25’:
mask = (patterns_poi['date_range_start'] >= '2021-05-15') & (patterns_poi['date_range_start'] <= '2022-01-25')
patterns_poi.loc[mask, 'Period'] = 'Post-pandemic 2'

patterns_poi = patterns_poi[~patterns_poi['Period'].isna()]
patterns_poi = patterns_poi[patterns_poi['GEOID'].astype(str).str[:5] == '42027']

# patterns_avg = calculate_average_patterns(patterns_poi)
# patterns_avg.to_csv('patterns_avg.csv', index=False)

patterns_poi = pd.read_csv('patterns_avg.csv')

# Distances
fishnet_centre_j = gp.read_file('census/gis/fishnet_potential_residential.shp')
fishnet_centre_j['X1'] = fishnet_centre_j.centroid.x
fishnet_centre_j['Y1'] = fishnet_centre_j.centroid.y

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

scipy_cdist = cdist(fishnet_centre_j[['X1', 'Y1']], fishnet_centre_j[['X1', 'Y1']], metric='euclidean')

pop['GEOID'] = pop['GEOID'].astype('int64')
ct_poi.index = ct_poi.index.astype('int64')
fishnet_centre_j['GEOID'] = fishnet_centre_j['GEOID'].astype('int64')

scipy_cdist = pd.DataFrame(scipy_cdist)
scipy_cdist.columns = [str(i) for i in fishnet_centre_j['GEOID']]
scipy_cdist.index = [str(i) for i in fishnet_centre_j['GEOID']]

# Calculate od matrices
od_list = pd.DataFrame(columns=['Period', 'Origin', 'Destination', 'Flow'])
for p in patterns_poi['Period'].unique():
    aux0 = patterns_poi[patterns_poi['Period'] == p]
    od = pd.DataFrame(index=list(aux0.GEOID), columns=list(aux0.GEOID), dtype='float')
    aux0.set_index('GEOID', inplace=True)
    for idx, row in aux0.iterrows():
        if ~row.isna().any():
            aux1 = pd.DataFrame.from_dict(json.loads(aux0.loc[idx, 'visitor_home_cbg_avg']), orient='index')
            aux1 = aux1[aux1.index.str[:5].isin(county_fips)]
            aux1.rename({0: 'avg'}, axis=1, inplace=True)
            aux1.index = aux1.index.astype('int64')
            mask = od.index.isin([int(i) for i in aux1.index])
            od.loc[mask, idx] = aux1.loc[:, 'avg']

    od = od.fillna(0)
    aux2 = od.stack().reset_index()
    aux2.columns = ['Origin', 'Destination', 'Flow']

    aux2 = aux2.merge(pop, left_on='Origin', right_on='GEOID', how='left')
    aux2.rename({'STATE': 'Origin state', 'COUNTY': 'Origin county', 'ALT0E001': 'Origin population',
                 'FIPS': 'Origin FIPS', 'Cohort': 'Origin Cohort'}, axis=1, inplace=True)
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

    aux2['Period'] = p

    od_list = pd.concat([od_list, aux2], axis=0)

# Here we sample from origin location
od_list['Distance'] = np.nan
od_list['Flow'] = od_list['Flow'].astype('int64').round()
for p in od_list['Period'].unique():
    for o in od_list['Origin Cohort 2'].unique():
        mask = (od_list['Period'] == p) & (od_list['Origin Cohort 2'] == o)
        aux = od_list[mask]
        j = 0
        for idx, row in aux.iterrows():
            print(p + ', ' + o + ', ' + str(aux.shape[0] - j))
            aux.loc[idx, 'Distance'] = str(list(
                scipy_cdist.loc[str(row['Origin']), str(row['Destination'])].iloc[0, :].sample(int(row['Flow']), replace=True)))
            j = j + 1
        od_list.loc[mask, 'Distance'] = aux['Distance']

# Distances
bgs = gp.read_file('census/gis/bg_counties.shp')

distances = bgs[['GEOID']].astype('int64')
distances['geometry'] = gp.GeoDataFrame(bgs.centroid)
distances = distances[distances['GEOID'].isin(od_list['Origin'])]
distances = gp.GeoDataFrame(distances)
distances.rename({0: 'geometry'}, axis=1, inplace=True)
distances = distances.set_crs(bgs.crs)
distances.set_index('GEOID', inplace=True)

dists = pd.DataFrame(index=list(distances.index), columns=list(distances.index))
for r, row in distances.iterrows():
    for c, col in distances.iterrows():
        dists.loc[r, c] = row['geometry'].distance(col['geometry'])

dist = dists.stack().reset_index()
dist.columns = ['Origin', 'Destination', 'Distance']

for idx, row in od_list.iterrows():
    if row['Flow']!=0:
        od_list.loc[idx, 'Distance_avg'] = np.mean(ast.literal_eval(row['Distance']))
    else:
        od_list.loc[idx, 'Distance_avg'] = np.array(dist.loc[(dist['Origin']==row['Origin'])&(dist['Destination']==row['Destination']), 'Distance'])[0]

od_list.to_csv('od_list.csv', index=False)
od_list = pd.read_csv('od_list.csv')

# Gravity model (separate cohort and pandemic period):
data = od_list
mask = data['Flow'] == 0
data.loc[mask, 'Flow'] = 1
mask = data['Origin population'] == 0
data.loc[mask, 'Origin population'] = 1
mask = data['Destination population'] == 0
data.loc[mask, 'Destination population'] = 1
mask = data['Origin poi counts'] == 0
data.loc[mask, 'Origin poi counts'] = 1
mask = data['Destination poi counts'] == 0
data.loc[mask, 'Destination poi counts'] = 1
mask = data['Distance_avg'] == 0
data.loc[mask, 'Distance_avg'] = 1

data['log Flow'] = np.log(data['Flow'])
data['log Origin population'] = np.log(data['Origin population'])
data['log Destination population'] = np.log(data['Destination population'])
data['log Origin poi counts'] = np.log(data['Origin poi counts'])
data['log Destination poi counts'] = np.log(data['Destination poi counts'])
data['log Distance_avg'] = np.log(data['Distance_avg'])
data = pd.concat([data, pd.get_dummies(data['Period'])], axis=1)
data = pd.concat([data, pd.get_dummies(data['Origin Cohort 2'])], axis=1)

X = data[['log Origin population', 'log Destination population', 'log Origin poi counts', 'log Destination poi counts',
          'log Distance_avg', 'Pre-pandemic', 'Social distancing', 'Post-pandemic 2', 'Centre County Student']]
X = sm.add_constant(X)
y = data['log Flow']

results = sm.OLS(y, X).fit()
print(results.summary())

# Gravity model (combine cohort and pandemic period):
data = od_list
mask = data['Flow'] == 0
data.loc[mask, 'Flow'] = 1
mask = data['Origin population'] == 0
data.loc[mask, 'Origin population'] = 1
mask = data['Destination population'] == 0
data.loc[mask, 'Destination population'] = 1
mask = data['Origin poi counts'] == 0
data.loc[mask, 'Origin poi counts'] = 1
mask = data['Destination poi counts'] == 0
data.loc[mask, 'Destination poi counts'] = 1
mask = data['Distance_avg'] == 0
data.loc[mask, 'Distance_avg'] = 1

data['log Flow'] = np.log(data['Flow'])
data['log Origin population'] = np.log(data['Origin population'])
data['log Destination population'] = np.log(data['Destination population'])
data['log Origin poi counts'] = np.log(data['Origin poi counts'])
data['log Destination poi counts'] = np.log(data['Destination poi counts'])
data['log Distance_avg'] = np.log(data['Distance_avg'])
data['period_community'] = data['Period'] + ', ' + data['Origin Cohort 2']
data = pd.concat([data, pd.get_dummies(data['period_community'])], axis=1)

correlations = data[['Flow', 'Origin population', 'Destination population', 'Origin poi counts', 'Destination poi counts', 'Distance_avg']].corr()
correlations.to_csv('correlations_model1.csv')

X = data[['log Origin population', 'log Destination population', 'log Origin poi counts', 'log Destination poi counts',
          'log Distance_avg', 'Pre-pandemic, Centre County Student',
          'Social distancing, Centre County Community', 'Social distancing, Centre County Student',
          'Post-pandemic 1, Centre County Community', 'Post-pandemic 1, Centre County Student',
          'Post-pandemic 2, Centre County Community', 'Post-pandemic 2, Centre County Student',
          ]]
X = sm.add_constant(X)
y = data['log Flow']

results = sm.OLS(y, X).fit()
print(results.summary())

# Distance friction
centre_distances = pd.DataFrame(columns=['Period', 'Origin Cohort 2', 'Distance'])
for p in od_list['Period'].unique():
    for o in od_list['Origin Cohort 2'].unique():
        mask = (od_list['Period'] == p) & (od_list['Origin Cohort 2'] == o)
        aux = od_list[mask]
        aux_dist = []
        for idx, row in aux.iterrows():
            if row['Distance'] != []:
                aux_dist = aux_dist + ast.literal_eval(row['Distance'])

        aux_dist = [float(i) / 1000 for i in aux_dist]
        aux_dist = pd.DataFrame(aux_dist)
        aux_dist.rename({0: 'Distance'}, inplace=True, axis=1)
        aux_dist['Period'] = p
        aux_dist['Origin Cohort 2'] = o
        centre_distances = pd.concat([centre_distances, aux_dist], axis=0)

st = centre_distances[centre_distances['Origin Cohort 2'] == 'Centre County Student']
fig = px.histogram(st, x='Distance', color='Period', opacity=0.7, nbins=200, barmode='overlay',
                   title='Visits from student CBGs to Centre County by distance')
fig.update_layout(
    font_family="Courier New",
    yaxis_title="Flow",
    xaxis_title="Distance",
    font=dict(
        family="Courier New, monospace",
        size=20,
        color="RebeccaPurple"
    ),
)
fig.show()

cm = centre_distances[centre_distances['Origin Cohort 2'] == 'Centre County Community']
fig = px.histogram(cm, x='Distance', color='Period', opacity=0.7, nbins=200, barmode='overlay',
                   title='Visits from community CBGs to Centre County by distance')
fig.update_layout(
    font_family="Courier New",
    yaxis_title="Flow",
    xaxis_title="Distance",
    font=dict(
        family="Courier New, monospace",
        size=25,
        color="RebeccaPurple"
    ), )
fig.show()

