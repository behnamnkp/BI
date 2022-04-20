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

# Here we read all patterns
path = "social_distancing_PA_counties/"
dirs = os.listdir(path)

df = pd.DataFrame({'origin_census_block_group': pd.Series(dtype=str) ,'date_range_start': pd.Series(dtype=str) ,'date_range_end': pd.Series(dtype=str) ,
                   'device_count': pd.Series(dtype=int) ,'distance_traveled_from_home': pd.Series(dtype=int) ,
             'bucketed_distance_traveled': pd.Series(dtype=str) ,'median_dwell_at_bucketed_distance_traveled': pd.Series(dtype=str) ,
             'completely_home_device_count': pd.Series(dtype=int) ,'median_home_dwell_time': pd.Series(dtype=pd.Series(dtype=int)) ,'bucketed_home_dwell_time': pd.Series(dtype=str) ,
             'at_home_by_each_hour': pd.Series(dtype=str) , 'part_time_work_behavior_devices': pd.Series(dtype=int) ,'full_time_work_behavior_devices': pd.Series(dtype=int) ,
             'destination_cbgs': pd.Series(dtype=str) ,'delivery_behavior_devices': pd.Series(dtype=int) ,'median_non_home_dwell_time': pd.Series(dtype=int) ,'candidate_device_count': pd.Series(dtype=int) ,
             'bucketed_away_from_home_time': pd.Series(dtype=str) ,'median_percentage_time_home': pd.Series(dtype=int) ,'bucketed_percentage_time_home': pd.Series(dtype=str), 'cohort':pd.Series(dtype=str)})

for folder in dirs:
    if '.zip' not in folder and 'centre' in folder:
        p = path + folder + '/'
        f = os.listdir(p)
        for item in f:
            if '_nonstudent' in item:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'non-student cbg'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)
            elif '_student' in item:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'student cbg'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)

df.to_csv('all_social_distancing_data.csv', index=False)
df = pd.read_csv('all_social_distancing_data.csv', low_memory=False)

# Pre-pandemic: ‘2019-09-01’ to ‘2020-03-09’: 191 days
mask = (df['date_range_start'] >= '2019-09-01') & (df['date_range_start'] <= '2020-03-09')
df.loc[mask, 'Period'] = 'Pre-pandemic'
# Social Distancing: ‘2020-03-09’ to ‘2020-09-01’: 175 days
mask = (df['date_range_start'] >= '2020-03-09') & (df['date_range_start'] <= '2020-09-01')
df.loc[mask, 'Period'] = 'Social distancing'
# Post Social Distancing 1: ‘2020-09-01’ to ‘2021-05-15’: 256 days
mask = (df['date_range_start'] >= '2020-09-01') & (df['date_range_start'] <= '2021-05-15')
df.loc[mask, 'Period'] = 'Post-pandemic 1'
# Post Social Distancing 2: ‘2021-05-15’ t o ‘2022-01-25’: 255 days
mask = (df['date_range_start'] >= '2021-05-15') & (df['date_range_start'] <= '2022-01-25')
df.loc[mask, 'Period'] = 'Post-pandemic 2'

df = df[~df['Period'].isna()]

df_c = df.groupby(['date_range_start', 'cohort']).sum()[['device_count', 'distance_traveled_from_home', 'completely_home_device_count', 'median_home_dwell_time',
                                      'median_non_home_dwell_time', 'candidate_device_count', 'median_percentage_time_home']].reset_index()

# df_c['date_range_start'] = pd.to_datetime(df_c['date_range_start'], utc=True)
# df_c['date_range_start'] = pd.to_datetime(df_c["date_range_start"].dt.strftime('%Y-%m-%d'))
# df_c.set_index('date_range_start', inplace = True)
# df_w = df_c.resample('W').mean().reset_index()

m = df_c['device_count'].max()
fig = px.line(df_c.reset_index(), x="date_range_start",
             y="device_count", title='device_count', color='cohort')
fig.add_vrect(x0='2019-09-01', x1='2020-03-09')
fig.add_annotation(x='2019-12-01', y=m, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=m, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=m, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=m, text="Post pandemic 2",showarrow=False)
fig.show()

m = df_c['distance_traveled_from_home'].max()
fig = px.line(df_c.reset_index(), x="date_range_start",
             y="distance_traveled_from_home", title='distance_traveled_from_home', color='cohort')
fig.add_vrect(x0='2019-09-01', x1='2020-03-09')
fig.add_annotation(x='2019-12-01', y=m, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=m, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=m, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=m, text="Post pandemic 2",showarrow=False)
fig.update_layout(
    yaxis_title="Y Axis Title",)
fig.show()

m = df_c['completely_home_device_count'].max()
fig = px.line(df_c.reset_index(), x="date_range_start",
             y="completely_home_device_count", title='completely_home_device_count', color='cohort')
fig.add_vrect(x0='2019-09-01', x1='2020-03-09')
fig.add_annotation(x='2019-12-01', y=m, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=m, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=m, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=m, text="Post pandemic 2",showarrow=False)
fig.show()

m = df_c['median_home_dwell_time'].max()
fig = px.line(df_c.reset_index(), x="date_range_start",
             y="median_home_dwell_time", title='median_home_dwell_time', color='cohort')
fig.add_vrect(x0='2019-09-01', x1='2020-03-09')
fig.add_annotation(x='2019-12-01', y=m, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=m, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=m, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=m, text="Post pandemic 2",showarrow=False)
fig.show()

m = df_c['median_percentage_time_home'].max()
fig = px.line(df_c.reset_index(), x="date_range_start",
             y="median_percentage_time_home", title='median_percentage_time_home', color='cohort')
fig.add_vrect(x0='2019-09-01', x1='2020-03-09')
fig.add_annotation(x='2019-12-01', y=m, text="Pre-pandemic",showarrow=False)
fig.add_vrect(x0='2020-03-10', x1='2020-09-01')
fig.add_annotation(x='2020-06-10', y=m, text="Social distancing",showarrow=False)
fig.add_vrect(x0='2020-09-01', x1='2021-05-15')
fig.add_annotation(x='2021-01-01', y=m, text="Post pandemic 1",showarrow=False)
fig.add_vrect(x0='2021-05-15', x1='2022-01-25')
fig.add_annotation(x='2021-10-01', y=m, text="Post pandemic 2",showarrow=False)
fig.show()