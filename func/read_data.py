import pandas as pd
import geopandas as gp
import os
import numpy as np
import json
import plotly.express as px
import ast

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