import pandas as pd
import geopandas as gp
import os
import numpy as np
import json
import plotly.express as px
import ast
from epiweeks import Week, Year
import datetime
from ast import literal_eval

def population(names, fips, student_tracts):

    # County populations
    cols = ['ALT0E001', 'GEOID', 'STATE', 'COUNTY']
    pop = pd.read_csv('census/attributes/nhgis0003_ds244_20195_blck_grp.csv', usecols=cols, encoding="ISO-8859-1")

    mask = (pop['STATE'] == 'Pennsylvania') & (pop['COUNTY'].isin(names))
    pop = pop.loc[mask, :]
    pop['GEOID'] = pop['GEOID'].str[7:]

    # Does not exist in SafeGraph
    # pop = pop[pop['GEOID'].str[:-1] != '42027012300']

    pop.loc[pop['COUNTY'] == 'Centre County', 'FIPS'] = fips[0]
    pop.loc[pop['COUNTY'] == 'Clearfield County', 'FIPS'] = fips[1]
    pop.loc[pop['COUNTY'] == 'Clinton County', 'FIPS'] = fips[2]
    pop.loc[pop['COUNTY'] == 'Union County', 'FIPS'] = int(fips[3])
    pop.loc[pop['COUNTY'] == 'Mifflin County', 'FIPS'] = fips[4]
    pop.loc[pop['COUNTY'] == 'Huntingdon County', 'FIPS'] = fips[5]
    pop.loc[pop['COUNTY'] == 'Blair County', 'FIPS'] = fips[6]

    pop['FIPS'] = pop['FIPS'].astype('int')
    pop_county = pop.groupby(['COUNTY', 'FIPS']).sum().reset_index()

    pop['GEOID'] = pop['GEOID'].astype('str')
    mask = pop['GEOID'].isin(student_tracts)
    pop.loc[mask, ['Cohort']] = 'Centre County Student'
    mask = (~pop['GEOID'].isin(student_tracts)) & (pop['COUNTY'] == 'Centre County')
    pop.loc[mask, ['Cohort']] = 'Centre County Community'
    mask = pop['Cohort'].isna()
    pop.loc[mask, ['Cohort']] = pop.loc[mask, 'COUNTY']

    pop_cohort = pop.groupby(['Cohort']).sum().reset_index()[['Cohort', 'ALT0E001']]
    pop_cohort.rename({'ALT0E001': 'Population'}, axis=1, inplace=True)
    aux = pd.DataFrame([['Centre County', (129294 + 30494)]], columns=list(['Cohort', 'Population']))
    pop_cohort = pop_cohort.append(aux, ignore_index=True)

    return pop_cohort

def read_patterns(path):

    dirs = os.listdir(path)

    df = pd.DataFrame(
        columns=['placekey', 'parent_placekey', 'location_name', 'street_address', 'city', 'region', 'postal_code',
                 'iso_country_code', 'safegraph_brand_ids', 'brands', 'date_range_start', 'date_range_end',
                 'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day', 'visits_by_each_hour', 'poi_cbg',
                 'visitor_home_cbgs', 'visitor_home_aggregation', 'visitor_daytime_cbgs', 'visitor_country_of_origin',
                 'distance_from_home', 'median_dwell', 'bucketed_dwell_times', 'related_same_day_brand',
                 'related_same_week_brand', 'device_type', 'carrier_name', 'County'])
    for folder in dirs:
        p = path + folder + '/'
        f = os.listdir(p)
        for item in f:
            print(item)
            df = pd.concat([df, pd.read_csv(p + item)], axis=0)

    return df

def read_pois():

    # Here, we need POI data for the desired county:
    blockgroups = gp.read_file('census/gis/bg_counties.shp')
    naics = pd.read_csv("PA_counties_weekly_patterns_20180101-20220124/PA_POI_20220108.csv")
    naics = gp.GeoDataFrame(naics, geometry=gp.points_from_xy(naics.longitude, naics.latitude))
    naics.set_crs('EPSG:4326', inplace=True)
    naics.to_crs(blockgroups.crs, inplace=True)
    naics_gdf = gp.sjoin(naics, blockgroups.loc[:, ['GEOID', 'geometry']], how='left')
    naics_gdf = naics_gdf[~naics_gdf['GEOID'].isna()]

    naics_sub = naics_gdf[['placekey', 'top_category', 'sub_category', 'naics_code', 'GEOID']]

    return naics_sub

def calculate_weekly_cases(data):

    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

    data_w = pd.DataFrame(columns=data.columns)
    idx = 0
    for y in years:
        for week in Year(y).iterweeks():
            s = week.startdate()
            e = week.enddate()
            mask = (data['Date'] >= str(s)) & (data['Date'] <= str(e))
            a = data.loc[mask, 'Weekly New Cases'].sum()
            data_w.loc[idx, 'Date'] = s
            data_w.loc[idx, 'Weekly New Cases'] = a
            idx = idx + 1

    return data_w

def read_social_distancing():

    path = "social_distancing_PA_counties/"
    dirs = os.listdir(path)

    df = pd.DataFrame({'origin_census_block_group': pd.Series(dtype=str), 'date_range_start': pd.Series(dtype=str),
                       'date_range_end': pd.Series(dtype=str),
                       'device_count': pd.Series(dtype=int), 'distance_traveled_from_home': pd.Series(dtype=int),
                       'bucketed_distance_traveled': pd.Series(dtype=str),
                       'median_dwell_at_bucketed_distance_traveled': pd.Series(dtype=str),
                       'completely_home_device_count': pd.Series(dtype=int),
                       'median_home_dwell_time': pd.Series(dtype=pd.Series(dtype=int)),
                       'bucketed_home_dwell_time': pd.Series(dtype=str),
                       'at_home_by_each_hour': pd.Series(dtype=str),
                       'part_time_work_behavior_devices': pd.Series(dtype=int),
                       'full_time_work_behavior_devices': pd.Series(dtype=int),
                       'destination_cbgs': pd.Series(dtype=str), 'delivery_behavior_devices': pd.Series(dtype=int),
                       'median_non_home_dwell_time': pd.Series(dtype=int),
                       'candidate_device_count': pd.Series(dtype=int),
                       'bucketed_away_from_home_time': pd.Series(dtype=str),
                       'median_percentage_time_home': pd.Series(dtype=int),
                       'bucketed_percentage_time_home': pd.Series(dtype=str), 'cohort': pd.Series(dtype=str)})

    for folder in dirs:
        if '.zip' not in folder and 'centre' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                if '_nonstudent' in item:
                    a = pd.read_csv(p + item, header=None)
                    a = a.iloc[:, :20]
                    a['cohort'] = 'Community cbg'
                    print(item)
                    a.columns = df.columns
                    df = pd.concat([df, a], axis=0)
                elif '_student' in item:
                    a = pd.read_csv(p + item, header=None)
                    a = a.iloc[:, :20]
                    a['cohort'] = 'Student cbg'
                    print(item)
                    a.columns = df.columns
                    df = pd.concat([df, a], axis=0)
        elif '.zip' not in folder and 'union' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                if 'student' not in item:
                    a = pd.read_csv(p + item, header=None)
                    a = a.iloc[:, :20]
                    a['cohort'] = 'Union County'
                    print(item)
                    a.columns = df.columns
                    df = pd.concat([df, a], axis=0)
        elif '.zip' not in folder and 'blair' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'Blair County'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)
        elif '.zip' not in folder and 'clearfield' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'Clearfield County'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)
        elif '.zip' not in folder and 'clinton' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'Clinton County'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)
        elif '.zip' not in folder and 'huntingdon' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'Huntingdon County'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)
        elif '.zip' not in folder and 'mifflin' in folder:
            p = path + folder + '/'
            f = os.listdir(p)
            for item in f:
                a = pd.read_csv(p + item, header=None)
                a = a.iloc[:, :20]
                a['cohort'] = 'Mifflin County'
                print(item)
                a.columns = df.columns
                df = pd.concat([df, a], axis=0)

    return df

def read_home_pattern(county_fips, county_nms, centre_student_tractcode):

    path = "Home panel summaries/home_panel_summary_PA_counties/"
    dirs = os.listdir(path)

    df = pd.DataFrame({'date_range_start': pd.Series(dtype=str), 'date_range_end': pd.Series(dtype=str),
                       'region': pd.Series(dtype=str),'iso_country_code': pd.Series(dtype=str),
                       'census_block_group': pd.Series(dtype=str),
                       'number_devices_residing': pd.Series(dtype=str),'number_devices_primary_daytime': pd.Series(dtype=str)})
    for f in dirs:
        if 'head' not in f and 'centre' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns, dtype=object)
            mask = a['census_block_group'].isin(centre_student_tractcode)
            a.loc[mask, 'cohort'] = 'Centre County Student'
            mask = a['cohort'].isna()
            a.loc[mask, 'cohort'] = 'Centre County Community'
            df = pd.concat([df, a], axis=0)
        elif 'head' not in f and 'union' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns)
            a['cohort'] = 'Union County'
            df = pd.concat([df, a], axis=0)
        elif 'head' not in f and 'blair' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns)
            a['cohort'] = 'Blair County'
            df = pd.concat([df, a], axis=0)
        elif 'head' not in f and 'clearfield' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns)
            a['cohort'] = 'Clearfield County'
            df = pd.concat([df, a], axis=0)
        elif 'head' not in f and 'clinton' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns)
            a['cohort'] = 'Clinton County'
            df = pd.concat([df, a], axis=0)
        elif 'head' not in f and 'huntingdon' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns)
            a['cohort'] = 'Huntingdon County'
            df = pd.concat([df, a], axis=0)
        elif 'head' not in f and 'mifflin' in f:
            a = pd.read_csv(path + f, header=None, names=df.columns)
            a['cohort'] = 'Mifflin County'
            df = pd.concat([df, a], axis=0)

    df['number_devices_primary_daytime'] = df['number_devices_primary_daytime'].astype(int)
    df['number_devices_residing'] = df['number_devices_residing'].astype(int)
    df = df.groupby(['cohort', 'date_range_start']).sum().reset_index()

    path = "Home panel summaries/normalization_stats/"
    dirs = os.listdir(path)

    df2 = pd.DataFrame({'year': pd.Series(dtype=str), 'month': pd.Series(dtype=str),
                       'day': pd.Series(dtype=str),'region': pd.Series(dtype=str),
                       'iso_country_code': pd.Series(dtype=str),
                       'total_visits': pd.Series(dtype=str),'total_home_visitors': pd.Series(dtype=str),
                       'total_home_visits': pd.Series(dtype=str), 'total_home_visitors': pd.Series(dtype=str)})

    for f in dirs:
        a = pd.read_csv(path + f)
        a = a[a['region']=='pa']
        a['total_visits'] = a['total_visits'].astype(int)
        a['total_home_visitors'] = a['total_home_visitors'].astype(int)
        a['date'] = pd.to_datetime(a[["year", "month", "day"]])
        a.rename({'total_visits':'state_total_visits', 'total_home_visitors':'state_total_home_visitors'}, axis=1, inplace=True)
        df['date_range_start'] = pd.to_datetime(df['date_range_start'], format='%Y-%m-%d', utc=True).dt.date
        mask = df['date_range_start'] == a.iloc[0, -1]
        df.loc[mask, 'state_total_visits'] = a[['state_total_visits', 'state_total_home_visitors']].mean()['state_total_visits']
        df.loc[mask, 'state_total_home_visitors'] = a[['state_total_visits', 'state_total_home_visitors']].mean()['state_total_home_visitors']

    df['state_total_home_visitors'] = df['state_total_home_visitors'].astype(int)

    df['number_devices_residing_norm'] = df['number_devices_residing'].astype(int) / df['state_total_home_visitors']

    df['date_range_start'] = pd.to_datetime(df['date_range_start'])

    mask = (df['date_range_start'] >= '2018-01-01') & (df['date_range_start'] <= '2018-05-06')
    df.loc[mask, 'semester'] = 'Spring 2018'
    mask = (df['date_range_start'] >= '2018-05-07') & (df['date_range_start'] <= '2018-08-11')
    df.loc[mask, 'semester'] = 'Summer 2018'
    mask = (df['date_range_start'] >= '2018-08-12') & (df['date_range_start'] <= '2018-12-15')
    df.loc[mask, 'semester'] = 'Fall 2018'
    mask = (df['date_range_start'] >= '2018-12-16') & (df['date_range_start'] <= '2019-01-01')
    df.loc[mask, 'semester'] = 'End of year 2018'
    mask = (df['date_range_start'] >= '2019-01-02') & (df['date_range_start'] <= '2019-05-05')
    df.loc[mask, 'semester'] = 'Spring 2019'
    mask = (df['date_range_start'] >= '2019-05-06') & (df['date_range_start'] <= '2019-08-10')
    df.loc[mask, 'semester'] = 'Summer 2019'
    mask = (df['date_range_start'] >= '2019-08-11') & (df['date_range_start'] <= '2019-12-21')
    df.loc[mask, 'semester'] = 'Fall 2019'
    mask = (df['date_range_start'] >= '2019-12-22') & (df['date_range_start'] <= '2020-01-01')
    df.loc[mask, 'semester'] = 'End of year 2019'
    mask = (df['date_range_start'] >= '2020-01-02') & (df['date_range_start'] <= '2020-05-10')
    df.loc[mask, 'semester'] = 'Spring 2020'
    mask = (df['date_range_start'] >= '2020-05-11') & (df['date_range_start'] <= '2020-08-15')
    df.loc[mask, 'semester'] = 'Summer 2020'
    mask = (df['date_range_start'] >= '2020-08-16') & (df['date_range_start'] <= '2020-12-19')
    df.loc[mask, 'semester'] = 'Fall 2020'
    mask = (df['date_range_start'] >= '2020-12-20') & (df['date_range_start'] <= '2021-01-01')
    df.loc[mask, 'semester'] = 'End of year 2020'
    mask = (df['date_range_start'] >= '2021-01-02') & (df['date_range_start'] <= '2021-05-09')
    df.loc[mask, 'semester'] = 'Spring 2021'
    mask = (df['date_range_start'] >= '2021-05-10') & (df['date_range_start'] <= '2021-08-14')
    df.loc[mask, 'semester'] = 'Summer 2021'
    mask = (df['date_range_start'] >= '2021-08-15') & (df['date_range_start'] <= '2021-12-18')
    df.loc[mask, 'semester'] = 'Fall 2021'
    mask = (df['date_range_start'] >= '2021-12-19') & (df['date_range_start'] <= '2022-01-01')
    df.loc[mask, 'semester'] = 'End of year 2021'
    mask = (df['date_range_start'] >= '2022-01-02') & (df['date_range_start'] <= '2022-05-08')
    df.loc[mask, 'semester'] = 'Spring 2022'
    mask = (df['date_range_start'] >= '2022-05-09') & (df['date_range_start'] <= '2022-08-13')
    df.loc[mask, 'semester'] = 'Summer 2022'

    avg = df.groupby(['cohort', 'semester']).apply(lambda x: x.mean()).reset_index()
    med = df.groupby(['cohort', 'semester']).apply(lambda x: x.median()).reset_index()

    for idx, row in avg.iterrows():
        mask = (df['cohort'] == row['cohort'])&(df['semester'] == row['semester'])
        df.loc[mask, 'avg_number_devices_residing'] = row['number_devices_residing']
    for idx, row in med.iterrows():
        mask = (df['cohort'] == row['cohort'])&(df['semester'] == row['semester'])
        df.loc[mask, 'med_number_devices_residing'] = row['number_devices_residing']

    for idx, row in avg.iterrows():
        mask = (df['cohort'] == row['cohort']) & (df['semester'] == row['semester'])
        df.loc[mask, 'avg_number_devices_residing_norm'] = row['number_devices_residing_norm']
    for idx, row in med.iterrows():
        mask = (df['cohort'] == row['cohort']) & (df['semester'] == row['semester'])
        df.loc[mask, 'med_number_devices_residing_norm'] = row['number_devices_residing_norm']

    return df

def state_home_devices():

    path = "Home panel summaries/normalization_stats/"
    dirs = os.listdir(path)

    df2 = pd.DataFrame({'year': pd.Series(dtype=str), 'month': pd.Series(dtype=str),
                       'day': pd.Series(dtype=str),'region': pd.Series(dtype=str),
                       'iso_country_code': pd.Series(dtype=str),
                       'total_visits': pd.Series(dtype=str),'total_home_visitors': pd.Series(dtype=str),
                       'total_home_visits': pd.Series(dtype=str), 'total_home_visitors': pd.Series(dtype=str)})

    for f in dirs:
        a = pd.read_csv(path + f)
        a = a[a['region'] == 'pa']
        a['total_visits'] = a['total_visits'].astype(int)
        a['total_home_visitors'] = a['total_home_visitors'].astype(int)
        a['date'] = pd.to_datetime(a[["year", "month", "day"]])
        df2 = pd.concat([df2, a], axis=0)

    return df2

def read_flu(county_names):

    # Use the following code in R to prepare county level flu cases
    # library(MMWRweek)
    # library(tidyverse)
    #
    # df <- read.csv("C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/sources/Influenza/County/Flu_county_2022_7_12.csv")
    # df$MMWRYR <- as.numeric(df$MMWRYR)
    # df$MMWRWK <- as.numeric(df$MMWRWK)
    # df$date = NULL
    #
    # for (x in 1:nrow(df)) {
    #
    #   df[x, 5] = MMWRweek2Date(MMWRyear = df[x, 4], MMWRweek = df[x, 3], MMWRday = NULL)
    #
    # }
    #
    # write.csv(df, "C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/sources/Influenza/County/Flu_county_date_2022_7_12.csv")

    f = pd.DataFrame(columns=['VisitDate', 'Result', 'LabName', 'Specimen Type'])
    for i in range(8):
        a = pd.read_excel('Influenza/PA/Bharti Exten flu positive lab.xlsx', sheet_name = i)
        f = pd.concat([f, a], axis=0)

    f['Date'] = pd.to_datetime(f['VisitDate'])
    f['Date'] = pd.to_datetime(f["Date"].dt.strftime('%Y-%m-%d'))
    f.drop(['VisitDate', 'LabName', 'Specimen Type'], axis=1, inplace=True)
    f.rename({'Result': 'Weekly New Cases'}, axis=1, inplace=True)

    f = f.groupby('Date').count().reset_index()
    f = calculate_weekly_cases(f)
    f['Cohort'] = 'Centre County Student'
    f['Date'] = pd.to_datetime(f['Date'])

    f = f[f['Date']>='2015-10-25']

    ff = pd.read_csv('Influenza/County/Flu_county_date_2022_7_12.csv')
    ff.drop('Unnamed: 0', axis=1, inplace=True)

    mask = ff['RESPJURIS'] == 'CENTRE'
    ff.loc[mask, 'RESPJURIS'] = county_names[0]
    mask = ff['RESPJURIS'] == 'CLEARFIELD'
    ff.loc[mask, 'RESPJURIS'] = county_names[1]
    mask = ff['RESPJURIS'] == 'CLINTON'
    ff.loc[mask, 'RESPJURIS'] = county_names[2]
    mask = ff['RESPJURIS'] == 'UNION'
    ff.loc[mask, 'RESPJURIS'] = county_names[3]
    mask = ff['RESPJURIS'] == 'MIFFLIN'
    ff.loc[mask, 'RESPJURIS'] = county_names[4]
    mask = ff['RESPJURIS'] == 'HUNTINGDON'
    ff.loc[mask, 'RESPJURIS'] = county_names[5]
    mask = ff['RESPJURIS'] == 'BLAIR'
    ff.loc[mask, 'RESPJURIS'] = county_names[6]

    mask = ff['Reported_casesi'] == '*'
    ff.loc[mask, 'Reported_casesi'] = np.random.randint(1, 4, ff.loc[mask].shape[0])
    ff['Reported_casesi'] = ff['Reported_casesi'].astype(int)

    ff['date'] = pd.to_datetime(ff['date'])
    ff.drop(['MMWRWK', 'MMWRYR'], axis=1, inplace=True)
    ff.rename({'RESPJURIS': 'Cohort'}, axis=1, inplace=True)
    ff.rename({'Reported_casesi': 'Weekly New Cases', 'date': 'Date'}, axis=1, inplace=True)

    ff.set_index('Date', inplace=True)
    f.set_index('Date', inplace=True)
    sc = f.join(ff, lsuffix='_caller', how='outer')
    sc = sc[(sc['Cohort']=='Centre County')|(sc['Cohort'].isna())]
    sc = sc.fillna(0)
    sc['Weekly New Cases'] = sc['Weekly New Cases'] - sc['Weekly New Cases_caller']

    fff = pd.DataFrame(sc['Weekly New Cases'])
    fff['Cohort'] = 'Centre County Community'

    flu = pd.concat([f, ff, fff], axis=0).reset_index()
    flu['Date'] = pd.to_datetime(flu['Date'])

    flu = flu[flu['Date'] <= '2022-02-14']

    # For students, the rest is reported weekly
    # We need to check for these two weeks
    flu = flu.append({'Date':datetime.datetime.strptime('2022-02-20 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':1, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-02-27 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':1, 'Cohort':'Centre County Student'}, ignore_index=True)
    # Week ending 3/12- 1 case Flu A
    # Week ending 3/19-2 cases Flu A
    # Week ending 3/26-1 case Flu A
    # Week ending 4/2-1 case Flu A
    # Week ending 4/9 -4 cases Flu A
    # Week ending 4/16-2 cases Flu A
    flu = flu.append({'Date':datetime.datetime.strptime('2022-03-06 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':1, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-03-13 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':2, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-03-20 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':1, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-03-27 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':1, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-04-03 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':4, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-04-10 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':2, 'Cohort':'Centre County Student'}, ignore_index=True)
    # Weekly Flu Cases:
    # 4/17 = 1
    # 4/25 = 8
    # 5/1 = 3
    # 5/8 - 6/26 = 0
    flu = flu.append({'Date':datetime.datetime.strptime('2022-04-17 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':1, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-04-24 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':8, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-05-01 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':3, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-05-08 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-05-15 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-05-22 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-05-29 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-06-05 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-06-12 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-06-19 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)
    flu = flu.append({'Date':datetime.datetime.strptime('2022-06-26 00:00:00',"%Y-%m-%d %H:%M:%S"),
                      'Weekly New Cases':0, 'Cohort':'Centre County Student'}, ignore_index=True)

    mask = (flu['Date'] >= '2015-08-01') & (flu['Date'] <= '2016-06-30')
    flu.loc[mask, 'Season'] = '2015'
    mask = (flu['Date']>='2016-08-01')&(flu['Date']<='2017-06-30')
    flu.loc[mask, 'Season'] = '2016'
    mask = (flu['Date']>='2017-08-01')&(flu['Date']<='2018-06-30')
    flu.loc[mask, 'Season'] = '2017'
    mask = (flu['Date']>='2018-08-01')&(flu['Date']<='2019-06-30')
    flu.loc[mask, 'Season'] = '2018'
    mask = (flu['Date']>='2019-08-01')&(flu['Date']<='2020-06-30')
    flu.loc[mask, 'Season'] = '2019'
    mask = (flu['Date']>='2020-08-01')&(flu['Date']<='2021-06-30')
    flu.loc[mask, 'Season'] = '2020'
    mask = (flu['Date']>='2021-08-01')&(flu['Date']<='2022-06-30')
    flu.loc[mask, 'Season'] = '2021'

    flu = flu[~flu['Season'].isna()]

    return flu

def read_flu_vaccine():

    a = pd.read_csv('Flu_vaccination/flu_vaccine.csv')
    a = a.set_index('Month').stack().reset_index()
    a.columns = ['Month', 'Year', 'Count']
    a['Day'] = 15
    a['Date'] = pd.to_datetime(a[['Year', 'Month', 'Day']])

    return a

def read_covid(county_fips, county_nms):

    ## Step one Covid_PSU (before 2020/12/11 reported from Friday to Friday):
    b1 = pd.read_csv('COVID_PSU/covid_cases_weekly_summary_counties_v20210304.csv')
    b1['Start date'] = pd.to_datetime(b1['date'])
    b1['End date'] = b1['Start date'] + datetime.timedelta(days=6)
    b1 = b1[(b1['Jurisdiction']=='Centre_PSU')&(b1['Start date']<='2020/12/11')]
    b1 = b1[['Start date', 'End date', 'new_cases_adj']]
    b1.rename({'new_cases_adj':'New cases'}, axis=1, inplace=True)
    b1['Step'] = 'Step 1'

    # Step two Covid_PSU (After 2020/12/11 to 4/10/2022 reported from Monday to Monday):
    b2 = pd.read_csv('COVID_PSU/PSU_COVID_Dashboard_Student_20220408.csv')
    b2.rename({'Positive_Tests': 'New cases', 'Week_Start': 'Start date', 'Week_Stop':'End date'}, axis=1, inplace=True)
    b2['Start date']= pd.to_datetime(b2['Start date'])
    b2['End date']= pd.to_datetime(b2['End date'])
    b2 = b2[['Start date', 'End date', 'New cases']]
    b2['Step'] = 'Step 2'

    # Step three Covid_PSU (From 4/10/2022 to the end of June. Daily tests):
    # This only includes UHS tests. Two weeks in March may have tests from the white building
    # Two types of tests. One is rapid test and the other is CEPHEID
    b3 = pd.read_csv('COVID_PSU/CEPHEID_UHS COVID results 4.7.2022 6.30.2022.csv')
    b3['Collection Date'] = pd.to_datetime(b3['Collection Date'])
    b3 = b3[(b3['POSITIVE']=='POSITIVE')|(b3['Positive']=='Positive')]
    mask = b3['POSITIVE']=='-'
    b3.loc[mask, 'POSITIVE'] = 'POSITIVE'
    b3.rename({'POSITIVE':'New cases'}, axis=1, inplace=True)
    b3 = b3[['Collection Date', 'New cases']]

    b4 = pd.read_csv('COVID_PSU/IDNOW_UHS COVID results 4.7.2022 6.30.2022.csv')
    b4 = b4[b4['COVID-19 ID NOW test'] == 'Positive']
    b4['Collection Date'] = pd.to_datetime(b4['Collection Date'])
    b4.rename({'COVID-19 ID NOW test': 'New cases'}, axis=1, inplace=True)
    b4 = b4[['Collection Date', 'New cases']]

    b34 = pd.concat([b3, b4], axis=0)

    b34 = b34.groupby('Collection Date').count().reset_index()
    b34 = b34[b34['Collection Date']>='2022-04-11'].set_index('Collection Date')
    b34.index = pd.DatetimeIndex(b34.index)
    b34_res = b34.resample('W', label='right').sum().reset_index()
    b34_res.rename({'Collection Date':'End date'}, axis=1, inplace=True)
    b34_res['Start date'] = b34_res['End date'] - pd.Timedelta(6, "d")
    b34_res['Step'] = 'Step 3'

    b = pd.concat([b1, b2, b34_res], axis=0)
    b = b.sort_values('Start date')
    b['Cohort'] = 'Centre-Student'
    # b = b[~((b['Date']=='2021-02-01')&(b['New cases']==69))]
    # b = b.drop_duplicates()

    # Data from EPI week 10. Reported sunday to sunday
    a = pd.read_csv('COVID_CentreCounty/aggregate_cases/COVID-19_Aggregate_Cases_Current_Daily_County_Health_v20220708.csv')
    a = a[a['County FIPS Code'].isin([int(i) for i in county_fips])]
    a.loc[a['County FIPS Code']==42027, 'County Name'] = county_nms[0]
    a.loc[a['County FIPS Code']==42033, 'County Name'] = county_nms[1]
    a.loc[a['County FIPS Code']==42035, 'County Name'] = county_nms[2]
    a.loc[a['County FIPS Code']==42119, 'County Name'] = county_nms[3]
    a.loc[a['County FIPS Code']==42087, 'County Name'] = county_nms[4]
    a.loc[a['County FIPS Code']==42061, 'County Name'] = county_nms[5]
    a.loc[a['County FIPS Code']==42013, 'County Name'] = county_nms[6]
    a['Date'] = pd.to_datetime(a['Date'])
    a = a.sort_values('Date')

    # Create weekly aggregates
    bb = b.copy()
    for c in a['County Name'].unique():
        for idx, row in b.iterrows():
            a1 = a[(a['County Name']==c)&(a['Date']>=row['Start date'])&(a['Date']<=row['End date'])]
            a2 = pd.DataFrame.from_dict({'Start date':row['Start date'],
                               'End date':row['End date'],
                               'Cohort':c,
                               'Step':row['Step'],
                               'New cases':a1['New Cases'].sum()}, orient='index').transpose()

            bb = pd.concat([bb, a2], axis=0)

    bb = bb.sort_values(['Cohort', 'Start date'])

    return bb

def read_covid_vaccine(county_fips, county_nms):

    # Community
    a = pd.read_csv('COVID_Vaccinations/COVID-19_Vaccinations_by_Day_by_County_of_Residence_Current_Health_v20220708.csv')
    a['County Name'] = a['County Name'] + ' County'
    a = a[a['County Name'].isin(county_nms)]
    a = a.fillna(0)
    a['Date'] = pd.to_datetime(a['Date'])
    a.set_index('Date', inplace=True)
    a.index = pd.DatetimeIndex(a.index)
    aa = a.groupby('County Name')
    aaa = aa.apply(lambda x: x.resample('W', label='right').sum().reset_index()).reset_index()
    aaa.rename({'County Name': 'Cohort'}, inplace=True, axis=1)
    aaa.drop('level_1', axis=1, inplace=True)

    # Students
    # PSU vaccination history
    # 1) 2020-12-23; COVID-19 vaccines to employees starts
    # 2) 2021-01-11; Penn State Health today began offering COVID-19 vaccinations to community health care providers and
    # emergency medical services providers not affiliated with its medical centers
    # 3) 2021-02-23; Penn State Health provided COVID-19 vaccines to more than 1,400 members of the public who meet current
    # Phase 1A eligibility on its first day operating
    # 4) 2021-07-26; Penn State Health will offer COVID-19 vaccination appointments to people ages 12 and older at nine Medical Group
    # practices throughout central Pennsylvania beginning Monday, July 26
    # 5) 2021-10-07; Penn State Health offers booster shots and third doses for eligible patients
    # 1) https://onwardstate.com/2021/09/15/penn-state-adds-vaccination-rates-to-covid-19-dashboard/
    # 2021-09-15; 84.7% of students
    # 2) https://www.psu.edu/news/campus-life/story/more-87-university-park-students-vaccinated-positivity-rate-declines/
    # 2021-10-01; more than 87%
    # 3) https://app.powerbi.com/view?r=eyJrIjoiNDY3NjhiMDItOWY0Mi00NzBmLWExNTAtZGIzNjdkMGI0OTM0IiwidCI6IjdjZjQ4ZDQ1LTNkZGItNDM4OS1hOWMxLWMxMTU1MjZlYjUyZSIsImMiOjF9
    # 2022-04-07; 88.6% of students

    mask = (aaa['Date'] < '2021-02-21')&(aaa['Cohort']=='Centre County')
    aaa.loc[mask, 'Step'] = '0 student vaccination'
    mask = (aaa['Date'] >= '2021-02-21')&(aaa['Date'] <='2021-09-15')&(aaa['Cohort']=='Centre County')
    aaa.loc[mask, 'Step'] = '85% student vaccination'
    mask = (aaa['Date'] >= '2021-09-16')&(aaa['Date'] <='2021-10-01')&(aaa['Cohort']=='Centre County')
    aaa.loc[mask, 'Step'] = '87% student vaccination'
    mask = (aaa['Date'] >= '2021-10-02')&(aaa['Date'] <='2022-06-30')&(aaa['Cohort']=='Centre County')
    aaa.loc[mask, 'Step'] = '89% student vaccination'

    b = aaa[['Cohort', 'Date', 'Fully Vaccinated', 'Step']]
    mask = b['Cohort'] == 'Centre County'
    b.loc[mask, 'Cohort'] = 'Centre County Community'

    student_population = 40000

    # Step 0
    bb = b[b['Step'] == '0 student vaccination']
    bb['Cohort'] = 'Centre County Student'
    bb['Fully Vaccinated'] = 0
    b = pd.concat([b, bb], axis=0, ignore_index=True)
    # Step 1
    bb0 = b[b['Step'] == '85% student vaccination']
    bb1 = b['Step'] == '85% student vaccination'
    bb0['Fully Vaccinated Perc'] = bb0['Fully Vaccinated'] / bb0['Fully Vaccinated'].sum()
    bb0['Fully Vaccinated'] = bb0['Fully Vaccinated Perc'] * 0.85 * student_population
    bb0['Cohort'] = 'Centre County Student'
    bb0.drop('Fully Vaccinated Perc', inplace=True, axis=1)
    b.loc[bb1, 'Fully Vaccinated'] = b.loc[bb1, 'Fully Vaccinated'] - bb0['Fully Vaccinated']
    b = pd.concat([b, bb0], axis=0, ignore_index=True)
    # Step 2
    bb0 = b[b['Step'] == '87% student vaccination']
    bb1 = b['Step'] == '87% student vaccination'
    bb0['Fully Vaccinated Perc'] = bb0['Fully Vaccinated'] / bb0['Fully Vaccinated'].sum()
    bb0['Fully Vaccinated'] = bb0['Fully Vaccinated Perc'] * 0.02 * student_population
    bb0['Cohort'] = 'Centre County Student'
    bb0.drop('Fully Vaccinated Perc', inplace=True, axis=1)
    b.loc[bb1, 'Fully Vaccinated'] = b.loc[bb1, 'Fully Vaccinated'] - bb0['Fully Vaccinated']
    b = pd.concat([b, bb0], axis=0, ignore_index=True)
    # Step 3
    bb0 = b[b['Step'] == '89% student vaccination']
    bb1 = b['Step'] == '89% student vaccination'
    bb0['Fully Vaccinated Perc'] = bb0['Fully Vaccinated'] / bb0['Fully Vaccinated'].sum()
    bb0['Fully Vaccinated'] = bb0['Fully Vaccinated Perc'] * 0.02 * student_population
    bb0['Cohort'] = 'Centre County Student'
    bb0.drop('Fully Vaccinated Perc', inplace=True, axis=1)
    b.loc[bb1, 'Fully Vaccinated'] = b.loc[bb1, 'Fully Vaccinated'] - bb0['Fully Vaccinated']
    b = pd.concat([b, bb0], axis=0, ignore_index=True)

    return [aaa,b]

# def student_population_dynamics():
#
#     # Total enrollment, university park
#     # Fall 2017; University Park: 46,610
#     # Fall 2018; University Park: 46,270
#     # Fall 2019; University Park: 46,723;
#     # Fall 2020; University Park; 45,901
#     # Fall 2021; University Park; 46,930

def define_cohorts(patterns, student_tracts):
    ## Calculate aggregated patterns
    patterns['GEOID'] = patterns['GEOID'].astype(str)

    patterns['County'] = patterns['GEOID'].astype(str).str[:5]
    mask = patterns['poi_cbg'].astype('int64').astype(str).isin(student_tracts)
    patterns.loc[mask, 'cbg_cohort'] = 'Centre-student'
    mask = (patterns['County'] == '42027') & (~patterns['poi_cbg'].astype('int64').astype(str).isin(student_tracts))
    patterns.loc[mask, 'cbg_cohort'] = 'Centre-community'
    # mask = (patterns['cbg_cohort'].isna())
    # patterns.loc[mask, 'cbg_cohort'] = patterns.loc[mask, 'County']

    return patterns[~patterns['cbg_cohort'].isna()]

def remove_unreasonable_patterns(patterns):

    patterns = patterns[~(patterns['location_name'] == 'Central Pa Institute Of Science & Techno')]
    patterns = patterns[
        ~(patterns['location_name'] == 'Central Pennsylvania Institute Of Science And Technology')]
    patterns = patterns[~(patterns['location_name'] == 'Earth And Mineral Sciences Museum And Gallery')]
    patterns = patterns[~(patterns['location_name'] == 'Pasto Agricultural Museum Rock Springs')]
    patterns = patterns[~(patterns['location_name'] == 'Souled Home Design')]

    return patterns

def calculate_average_patterns(patterns, student_tracts):

    """
        Aggregates patterns into cbg level and averages them over the periods.
        :param patterns_poi: The raw foot traffic data
        :return: Aggreagted, averaged patterns
    """

    for idx, row in patterns.iterrows():
        print(idx)
        a = json.loads(row['visitor_home_cbgs'])
        if len(a) != 0:
            patterns.loc[idx, 'Total visitors'] = np.float(sum(a.values()))
            s = 0
            c = 0
            for key in a:
                if key in student_tracts:
                    s = s + a[key]
                else:
                    c = c + a[key]
            patterns.loc[idx, 'Student proportion'] = np.float(s / sum(a.values()))
            patterns.loc[idx, 'Community proportion'] = np.float(c / sum(a.values()))

            patterns.loc[idx, 'Student visits'] = np.float(s)
            patterns.loc[idx, 'Community visits'] = np.float(c)

        else:
            patterns.loc[idx, 'Total visitors'] = 0
            patterns.loc[idx, 'Student proportion'] = 0
            patterns.loc[idx, 'Community proportion'] = 0
            patterns.loc[idx, 'Student visits'] = 0
            patterns.loc[idx, 'Community visits'] = 0

    patterns['proportion_max'] = patterns[['Student proportion', 'Community proportion']].max(axis=1)
    mask = patterns['proportion_max']==0
    patterns.loc[mask, 'proportion_max'] = 1

    return patterns

def influenza(cohort_average_patterns, flu):

    flu = flu[(flu['Cohort']=='Centre County Community')|(flu['Cohort']=='Centre County Student')]
    cohort_average_patterns['Date_flu'] = pd.to_datetime(cohort_average_patterns['Date'] - datetime.timedelta(days=1),
                                                         utc=True).dt.strftime('%Y-%m-%d')
    cohort_average_patterns.set_index('Date_flu', inplace=True)
    a = cohort_average_patterns[['Student proportion', 'Community proportion']].stack().reset_index()
    a.rename({'level_1':'Cohort', 0:'proportion'}, axis=1, inplace=True)
    mask = a['Cohort'] == 'Student proportion'
    a.loc[mask, 'Cohort'] = 'Centre County Student'
    mask = a['Cohort'] == 'Community proportion'
    a.loc[mask, 'Cohort'] = 'Centre County Community'
    b = cohort_average_patterns[['Student visits', 'Community visits']].stack().reset_index()
    b.rename({0: 'Visits'}, axis=1, inplace=True)
    b.drop(['level_1', 'Date_flu'], inplace=True, axis=1)
    c = pd.concat([a, b], axis=1)
    c.set_index('Date_flu', inplace=True)
    d = c.merge(cohort_average_patterns[['Total visitors', 'proportion_max']], left_on=c.index, right_on=cohort_average_patterns.index, how='left')
    d.rename({'key_0':'Date'}, axis=1, inplace=True)

    flu['Date'] = flu['Date'].dt.strftime('%Y-%m-%d')
    # flu[(flu['Date']>=d['Date_flu'].min())&(flu['Date']<=d['Date_flu'].max())]

    df = flu.merge(d, left_on=['Date', 'Cohort'], right_on=['Date', 'Cohort'],  how='inner')

    return df

def resample_dates(patterns):

    # patterns['date_range_start'] = patterns['date_range_start'].strftime("%Y-%m-%d %H:%M:%S")
    patterns.set_index('date_range_start', inplace=True)
    patterns2 = pd.DataFrame(columns=['date_range_start', 'placekey', 'Student proportion', 'Community proportion','Total visitors'])
    patterns2.set_index('date_range_start', inplace=True)
    for idx, row in patterns.iterrows():
        row['visits_by_day_props'] = str(list(literal_eval(row['visits_by_day']) / np.sum(literal_eval(row['visits_by_day']))))
        row['visits_by_day_resampled'] = str(list(np.array(literal_eval(row['visits_by_day_props'])) * row['Total visitors']))
        a = pd.DataFrame(columns=['placekey', 'Student proportion', 'Community proportion','Total visitors'],
                         index=[row.name] + (pd.date_range(row.name, periods=6) +
                                                          pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S").tolist())
        a.iloc[:, 0] = row['placekey']
        a.iloc[:, 1] = row['Student proportion']
        a.iloc[:, 2] = row['Community proportion']
        a.iloc[:, 3] = literal_eval(row['visits_by_day_resampled'])

        # a['Total visitors'] = a['Total visitors'].round()

        patterns2 = pd.concat([patterns2, a], axis=0)

    patterns2['Student visits'] = patterns2['Total visitors'] * patterns2['Student proportion']
    patterns2['Community visits'] = patterns2['Total visitors'] * patterns2['Community proportion']




















