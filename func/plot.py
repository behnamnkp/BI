import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
import sys
import numpy as np

def pattern_by_mixture(cohort_average_patterns):

    a = cohort_average_patterns[['Student visits', 'Community visits']].stack().reset_index()
    a.rename({0:'Average visitors', 'level_1':'Cohort'}, axis=1, inplace=True)

    b = cohort_average_patterns[['proportion_max']].stack().reset_index()
    b.rename({0:'Cohort average mixture', 'level_1':'Cohort'}, axis=1, inplace=True)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=cohort_average_patterns['date_range_start'], y=cohort_average_patterns['Student visits'], name="Student average visits"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=cohort_average_patterns['date_range_start'], y=cohort_average_patterns['Community visits'], name="Community average visits"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=cohort_average_patterns['date_range_start'], y=b['Cohort average mixture'], name='Cohort average mixture'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Intra- and inter-cohort mixture"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Average visits</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Cohort average mixture</b>", secondary_y=True)

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/1-Visits_and_mixture.png',
                    engine='orca', height=700, width=1500, scale=3)

def plot_flu_all(flu):

    fig = px.line(flu, x="Date", y="Weekly New Cases", color = 'Cohort', title='Flu weekly cases')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/2-Flu-all-cohorts.png',
                    engine='orca', height=700, width=1500, scale=3)

def plot_flu_centre(flu):

    fig = px.line(flu[(flu['Cohort']=='Centre County Student')|(flu['Cohort']=='Centre County Community')|(flu['Cohort']=='Centre County')],
                  x="Date", y="Weekly New Cases", color = 'Cohort', title='Flu weekly cases')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/3-Flu-centre.png',
                    engine='orca', height=700, width=1500, scale=3)

def plot_flu_per_capita(flu):

    a = flu[flu['Cohort']=='Centre County Student']
    a['Weekly New Cases'] = a['Weekly New Cases'] / 38000
    b = flu[flu['Cohort'] == 'Centre County Community']
    b['Weekly New Cases'] = b['Weekly New Cases'] / 129294
    c = flu[flu['Cohort'] == 'Centre County']
    c['Weekly New Cases'] = c['Weekly New Cases'] / (129294+38000)

    # Create figure with secondary y-axis
    fig = make_subplots()

    # Add traces
    fig.add_trace(
        go.Scatter(x=a['Date'], y=a['Weekly New Cases'], name="Student Weekly New Cases Per Capita"),
    )
    fig.add_trace(
        go.Scatter(x=b['Date'], y=b['Weekly New Cases'], name="Community Weekly New Cases Per Capita"),
    )
    fig.add_trace(
        go.Scatter(x=c['Date'], y=c['Weekly New Cases'], name="Centre County Weekly New Cases Per Capita"),
    )

    # Add figure title
    fig.update_layout(
        title_text="Influenza cases per capita"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Weekly New Cases</b>")

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/4-Flu-centre-per-capita.png',
                    engine='orca', height=400, width=1500, scale=3)

def plot_flu_vaccine(flu, flu_vaccine):

    a = flu[flu['Cohort'] == 'Centre County Student']
    b = flu[flu['Cohort'] == 'Centre County Community']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=a['Date'], y=a['Weekly New Cases'], name="Student Weekly New Cases"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=b['Date'], y=b['Weekly New Cases'], name="Community Weekly New Cases"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=flu_vaccine['Date'], y=flu_vaccine['Count'], name='PSU student flu vaccination by month'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Flu and vaccination"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Weekly New Cases</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>PSU student flu vaccination by month</b>", secondary_y=True)

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/5-Flu-centre-vaccination.png',
                    engine='orca', height=700, width=1500, scale=3)

def plot_community_flu_pattern(cohort_average_patterns, flu):

    b = flu[flu['Cohort']=='Centre County Community']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=cohort_average_patterns['Date'], y=cohort_average_patterns['Community visits'], name="Centre County Community average visitors"),
        secondary_y=False,)
    fig.add_trace(
        go.Scatter(x=b['Date'], y=b['Weekly New Cases'], name='Centre County Community Flu Cases'),
        secondary_y=True,)
    # Add figure title
    fig.update_layout(
        title_text="Weekly flu and visits (community)")
    # Set x-axis title
    fig.update_xaxes(title_text="Week")
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Average visits by community</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Centre County Community Flu Cases</b>", secondary_y=True)
    fig.update_layout(width=1500, height=500)
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/6-Community-flu-pattern.png',
                    engine='orca', height=400, width=1800, scale=3)

def plot_student_flu_pattern(cohort_average_patterns, flu):

    b = flu[flu['Cohort']=='Centre County Student']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=cohort_average_patterns['Date'], y=cohort_average_patterns['Student visits'], name="Centre County Student average visitors"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=b['Date'], y=b['Weekly New Cases'], name='Centre County Student Flu Cases'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Weekly flu and visits (Student)"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Average visits by Student</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Centre County Student Flu Cases</b>", secondary_y=True)

    fig.update_layout(width=1800, height=750)

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/7-Student-flu-pattern.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_flu_social_distancing_student(social_distancing, flu):

    a = flu[flu['Cohort']=='Centre County Student']

    c = social_distancing[social_distancing['cohort']=='Student cbg']
    c = c.groupby(['cohort', 'date_range_start']).mean()[
        ['median_home_dwell_time', 'device_count', 'median_percentage_time_home']].reset_index()
    c['date_range_start'] = pd.to_datetime(c['date_range_start'], utc=True)
    c.set_index('date_range_start', inplace=True)
    c = c.resample('W').mean()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=a['Date'], y=a['Weekly New Cases'], name="Student weekly new cases"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=c.index, y=c['median_home_dwell_time'], name='Student median home dwell time'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Flu and home dwelling time (students)"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Weekly New Cases</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Median home dwell time</b>", secondary_y=True)

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/8-Student-flu-social-distancing.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_flu_social_distancing_community(social_distancing, flu):

    a = flu[flu['Cohort']=='Centre County Community']

    c = social_distancing[social_distancing['cohort']=='Community cbg']
    c = c.groupby(['cohort', 'date_range_start']).mean()[
        ['median_home_dwell_time', 'device_count', 'median_percentage_time_home']].reset_index()
    c['date_range_start'] = pd.to_datetime(c['date_range_start'], utc=True)
    c.set_index('date_range_start', inplace=True)
    c = c.resample('W').mean()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=a['Date'], y=a['Weekly New Cases'], name="Community weekly new cases"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=c.index, y=c['median_home_dwell_time'], name='Community median home dwell time'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Flu and home dwelling time (Community)"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Weekly New Cases</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Median home dwell time</b>", secondary_y=True)

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/9-Community-flu-social-distancing.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_social_distancing(social_distancing):

    b = pd.DataFrame(columns=['date_range_start', 'median_home_dwell_time', 'device_count', 'median_percentage_time_home', 'cohort'])
    for item in social_distancing['cohort'].unique():
        a = social_distancing.loc[social_distancing['cohort']==item,
                              ['date_range_start', 'median_home_dwell_time', 'device_count', 'median_percentage_time_home']]
        a['date_range_start'] = pd.to_datetime(a['date_range_start'], utc=True)
        a.set_index('date_range_start', inplace=True)
        a = a.resample('W').mean().reset_index()
        a['cohort'] = item
        b = pd.concat([b, a], axis=0)

    fig = px.line(b, x=b['date_range_start'], y="median_home_dwell_time", color ='cohort', title='Weekly median home dwell time')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/10-social-distancing.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_covid_all(covid):

    fig = px.line(covid, x="Start date", y="New cases", color = 'Cohort', title='Covid cases')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/11-Covid.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_covid_centre(covid):

    a = covid[covid['Cohort'] == 'Centre-Student'].reset_index()
    b = covid[covid['Cohort'] == 'Centre County'].drop_duplicates().reset_index()
    b.rename({'New cases':'New cases Centre'}, inplace=True, axis=1)
    a = pd.concat([a, pd.DataFrame(b['New cases Centre'])], axis=1)
    a['New Cases Community'] = a['New cases Centre'] - a['New cases']
    a.drop(['New cases', 'New cases Centre'], axis=1, inplace=True)
    a.rename({'New Cases Community':'New cases'}, inplace=True, axis=1)
    a['Cohort'] = 'Centre-Community'
    # mask = a['New cases']<0
    # a.loc[mask, 'New cases'] = 01
    a.set_index('index', inplace=True)
    a = a.sort_values('Start date')

    covid = pd.concat([covid, a], axis=0)
    covid = covid.drop_duplicates()

    fig = px.line(covid[(covid['Cohort']=='Centre-Student')|(covid['Cohort']=='Centre County')|(covid['Cohort']=='Centre-Community')], x="Start date", y="New cases", color = 'Cohort', title='Covid cases')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/12-Covid-centre.png',
                    engine='orca', height=750, width=1800, scale=3)

def flu_seasonal(flu):

    f=pd.DataFrame()
    f['2015_2016'] = flu[(flu['Date']>='2015-08-01')&(flu['Date']<='2016-06-30')].groupby('Cohort').sum()
    f['2016_2017'] = flu[(flu['Date']>='2016-08-01')&(flu['Date']<='2017-06-30')].groupby('Cohort').sum()
    f['2017_2018'] = flu[(flu['Date']>='2017-08-01')&(flu['Date']<='2018-06-30')].groupby('Cohort').sum()
    f['2018_2019'] = flu[(flu['Date']>='2018-08-01')&(flu['Date']<='2019-06-30')].groupby('Cohort').sum()
    f['2019_2020'] = flu[(flu['Date']>='2019-08-01')&(flu['Date']<='2020-06-30')].groupby('Cohort').sum()
    f['2020_2021'] = flu[(flu['Date']>='2020-08-01')&(flu['Date']<='2021-06-30')].groupby('Cohort').sum()
    f['2021_2022'] = flu[(flu['Date']>='2021-08-01')&(flu['Date']<='2022-06-30')].groupby('Cohort').sum()

    f = f.stack().reset_index()
    f.columns = ['Cohort', 'Flu season', 'New cases']

    fig = px.bar(f, x="Flu season", y="New cases", color = 'Cohort', title='Covid cases', barmode='group')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/13-flu-seasonal.png',
                    engine='orca', height=750, width=1800, scale=3)

def flu_vaccination_seasonal(flu, flu_vaccine):

    f=pd.DataFrame()

    f['2015_2016'] = flu[(flu['Date']>='2015-08-01')&(flu['Date']<='2016-06-30')].groupby('Cohort').sum()
    f['2016_2017'] = flu[(flu['Date']>='2016-08-01')&(flu['Date']<='2017-06-30')].groupby('Cohort').sum()
    f['2017_2018'] = flu[(flu['Date']>='2017-08-01')&(flu['Date']<='2018-06-30')].groupby('Cohort').sum()
    f['2018_2019'] = flu[(flu['Date']>='2018-08-01')&(flu['Date']<='2019-06-30')].groupby('Cohort').sum()
    f['2019_2020'] = flu[(flu['Date']>='2019-08-01')&(flu['Date']<='2020-06-30')].groupby('Cohort').sum()
    f['2020_2021'] = flu[(flu['Date']>='2020-08-01')&(flu['Date']<='2021-06-30')].groupby('Cohort').sum()
    f['2021_2022'] = flu[(flu['Date']>='2021-08-01')&(flu['Date']<='2022-06-30')].groupby('Cohort').sum()

    f = f[(f.index=='Centre County Community')|(f.index=='Centre County Student')]

    f = f.stack().reset_index()
    f.columns = ['Cohort', 'Flu season', 'New cases']

    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    v = []
    for year in years:
        v.append(flu_vaccine[((flu_vaccine['Month']>=8)&(flu_vaccine['Year']==str(year))) |
                                     ((flu_vaccine['Month']<=6)&(flu_vaccine['Year']==str(year+1)))]['Count'].sum())

    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1)
    # Add traces
    fig.add_trace(
        go.Bar(x=years, y=v, name="Student vaccination"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=f[f['Cohort']=='Centre County Community']['Flu season'],
                   y=f[f['Cohort']=='Centre County Community']['New cases'],
                   name='Community flu cases'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=f[f['Cohort']=='Centre County Student']['Flu season'],
                   y=f[f['Cohort']=='Centre County Student']['New cases'],
                   name='Student flu cases'),
        row=2, col=1
    )
    # Add figure title
    fig.update_layout(
        title_text="Flu and vaccination"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Season")
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Vaccination</b>",row=1, col=1)
    fig.update_yaxes(title_text="<b>Flu new cases</b>", row=2, col=1)
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/14-flu-seasonal-vaccination.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_covid_vaccination_step(covid_vaccine):

    fig = px.line(covid_vaccine, x="Date", y="Partially Vaccinated", color = 'Cohort', title='Covid vaccination (Partially Vaccinated)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/15-Covid-Partially Vaccinated-step.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(covid_vaccine, x="Date", y="Fully Vaccinated", color = 'Cohort', title='Covid vaccination (Fully Vaccinated)')
    fig.add_vline(x='2021-02-21')
    fig.add_vline(x='2021-09-15')
    fig.add_vline(x='2021-10-01')
    fig.add_vline(x='2022-04-07')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/15-Covid-Fully Vaccinated-step.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(covid_vaccine, x="Date", y="First Booster Dose", color = 'Cohort', title='Covid vaccination (First Booster Dose)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/15-Covid-First Booster Dose-step.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(covid_vaccine, x="Date", y="Second Booster Dose", color = 'Cohort', title='Covid vaccination (Second Booster Dose)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/15-Covid-Second Booster Dose-step.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_covid_vaccination(covid_vaccine):

    fig = px.line(covid_vaccine, x="Date", y="Fully Vaccinated", color = 'Cohort', title='Covid vaccination (Fully Vaccinated)')
    fig.add_vline(x='2021-02-21')
    fig.add_vline(x='2021-09-15')
    fig.add_vline(x='2021-10-01')
    fig.add_vline(x='2022-04-07')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/15-Covid-Fully Vaccinated.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_home_pattern(home_pattern):


    fig = px.line(home_pattern, x="date_range_start", y="number_devices_residing", color = 'cohort', title='Home patterns (all counties)')
    fig.add_vline(x='2018-01-02', line_dash="dash")#spring
    fig.add_vline(x='2018-05-06', line_dash="dash")#summer
    fig.add_vline(x='2018-08-11', line_dash="dash")#Fall
    fig.add_vline(x='2018-12-15', line_dash="dash")
    fig.add_vline(x='2019-01-02', line_dash="dash")#spring
    fig.add_vline(x='2019-05-05', line_dash="dash")#spring
    fig.add_vline(x='2019-08-10', line_dash="dash")#summer
    fig.add_vline(x='2019-12-21', line_dash="dash")#Fall
    fig.add_vline(x='2020-01-02', line_dash="dash")
    fig.add_vline(x='2020-05-10', line_dash="dash")#spring
    fig.add_vline(x='2020-08-15', line_dash="dash")#summer
    fig.add_vline(x='2020-12-19', line_dash="dash")#Fall
    fig.add_vline(x='2021-01-02', line_dash="dash")
    fig.add_vline(x='2021-05-09', line_dash="dash")#spring
    fig.add_vline(x='2021-08-14', line_dash="dash")#summer
    fig.add_vline(x='2021-12-18', line_dash="dash")#Fall
    fig.add_vline(x='2022-01-02', line_dash="dash")
    fig.add_vline(x='2022-05-08', line_dash="dash")#spring
    fig.add_vline(x='2022-08-13', line_dash="dash")#summer

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/16-Home-patterns.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(home_pattern, x="date_range_start", y="number_devices_residing_norm", color = 'cohort', title='Home patters (normalized)')
    fig.add_vline(x='2018-01-02', line_dash="dash")  # spring
    fig.add_vline(x='2018-05-06', line_dash="dash")  # summer
    fig.add_vline(x='2018-08-11', line_dash="dash")  # Fall
    fig.add_vline(x='2018-12-15', line_dash="dash")
    fig.add_vline(x='2019-01-02', line_dash="dash")  # spring
    fig.add_vline(x='2019-05-05', line_dash="dash")  # spring
    fig.add_vline(x='2019-08-10', line_dash="dash")  # summer
    fig.add_vline(x='2019-12-21', line_dash="dash")  # Fall
    fig.add_vline(x='2020-01-02', line_dash="dash")
    fig.add_vline(x='2020-05-10', line_dash="dash")  # spring
    fig.add_vline(x='2020-08-15', line_dash="dash")  # summer
    fig.add_vline(x='2020-12-19', line_dash="dash")  # Fall
    fig.add_vline(x='2021-01-02', line_dash="dash")
    fig.add_vline(x='2021-05-09', line_dash="dash")  # spring
    fig.add_vline(x='2021-08-14', line_dash="dash")  # summer
    fig.add_vline(x='2021-12-18', line_dash="dash")#Fall
    fig.add_vline(x='2022-01-02', line_dash="dash")
    fig.add_vline(x='2022-05-08', line_dash="dash")#spring
    fig.add_vline(x='2022-08-13', line_dash="dash")#summer
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/16-Home-patterns-normalized.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(home_pattern[home_pattern['cohort']=='Blair County'], x="date_range_start", y="state_total_home_visitors", title='Home patters (state)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/16-Home-patterns-state.png',
                    engine='orca', height=750, width=1800, scale=3)


    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "number_devices_residing"],
                   name='Number devices (student)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "med_number_devices_residing"],
                   name='Median number of devices (student)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "number_devices_residing"],
                   name='Number devices (community)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "med_number_devices_residing"],
                   name='Median number of devices (community)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-01-02', '2018-01-02'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-05-06', '2018-05-06'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-08-11', '2018-08-11'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-12-15', '2018-12-15'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-01-02', '2019-01-02'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-05-05', '2019-05-05'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-08-10', '2019-08-10'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-12-21', '2019-12-21'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-01-02', '2020-01-02'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-05-10', '2020-05-10'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-08-15', '2020-08-15'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-12-19', '2020-12-19'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-01-02', '2021-01-02'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-05-09', '2021-05-09'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-08-14', '2021-08-14'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-12-18', '2021-12-18'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2022-01-02', '2022-01-02'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2022-05-08', '2022-05-08'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2022-08-13', '2022-08-13'], y=[0, 8000], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/17-Home-patterns-med.png',
                    engine='orca', height=750, width=1800, scale=3)


    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "number_devices_residing_norm"],
                   name='Normalized number of devices (student)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Student', "med_number_devices_residing_norm"],
                   name='Normalized median number of devices (student)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "number_devices_residing_norm"],
                   name='Normalized number of devices (community)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "date_range_start"],
                   y=home_pattern.loc[home_pattern['cohort']=='Centre County Community', "med_number_devices_residing_norm"],
                   name='Normalized median number of devices (community)',
                   mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-01-02', '2018-01-02'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-05-06', '2018-05-06'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-08-11', '2018-08-11'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2018-12-15', '2018-12-15'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-01-02', '2019-01-02'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-05-05', '2019-05-05'], y=[0, 0.013], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-08-10', '2019-08-10'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2019-12-21', '2019-12-21'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-01-02', '2020-01-02'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-05-10', '2020-05-10'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-08-15', '2020-08-15'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2020-12-19', '2020-12-19'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-01-02', '2021-01-02'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-05-09', '2021-05-09'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-08-14', '2021-08-14'], y=[0, 0.016], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2021-12-18', '2021-12-18'], y=[0, 0.013], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2022-01-02', '2022-01-02'], y=[0, 0.013], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2022-05-08', '2022-05-08'], y=[0, 0.013], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=['2022-08-13', '2022-08-13'], y=[0, 0.013], mode='lines',showlegend=False, line=dict(color='black',width=1, dash='dash',)),row=1, col=1
    )

    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/17-Home-patterns-med-norm.png',
                    engine='orca', height=750, width=1800, scale=3)

def plot_state_stats(state_stats):

    fig = px.line(state_stats, x="date", y="total_visits", title='total visits (state)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/18-state-stats-total-visits.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(state_stats, x="date", y="total_home_visitors", title='total_home_visitors (state)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/18-state-stats-total_home_visitors.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(state_stats, x="date", y="total_home_visits", title='total_home_visits (state)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/18-state-stats-total_home_visits.png',
                    engine='orca', height=750, width=1800, scale=3)

    fig = px.line(state_stats, x="date", y="total_home_visitors", title='total_home_visitors (state)')
    fig.show()
    fig.write_image('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/output/plots/Influenza/18-state-stats-total_home_visitors.png',
                    engine='orca', height=750, width=1800, scale=3)



