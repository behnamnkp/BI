import pandas as pd
import os
from func import read_data
from func import plot
import datetime

def main():

    pd.set_option('display.max_columns', 10)
    os.chdir('C:/Users/bzn5190/OneDrive - The Pennsylvania State University/Behavioral_interventions/sources/')

    county_fips = ['42027', '42033', '42035', '42119', '42087', '42061', '42013']
    county_nms = ['Centre County', 'Clearfield County', 'Clinton County', 'Union County', 'Mifflin County',
                  'Huntingdon County', 'Blair County']

    centre_student_tractcode = ["420270120001","420270120002","420270120003","420270120004","420270120005","420270121001",
                                "420270121002","420270121003","420270121004","420270122001","420270122002","420270122003",
                                "420270124001","420270124002","420270124003","420270125001","420270125002","420270126001",
                                "420270126002"]

    # County and cohort populations
    population = read_data.population(county_nms, county_fips, centre_student_tractcode)

    # # Here we read all patterns
    # path = "PA_counties_weekly_patterns_20180101-20220124/PA_counties_weekly_patterns/"
    # df = read_data.read_patterns(path)
    # df.to_csv('all.csv', index=False)
    patterns = pd.read_csv('all.csv', low_memory=False)

    # Here we read places of interest data
    pois = read_data.read_pois()

    # Here we process SafeGraph data
    patterns = patterns.merge(pois, on='placekey')
    # separate student and non-student cohorts
    patterns = read_data.define_cohorts(patterns, centre_student_tractcode)
    # Remove strange cases
    patterns = read_data.remove_unreasonable_patterns(patterns)
    # Calculate average patterns
    patterns = read_data.calculate_average_patterns(patterns, centre_student_tractcode)
    # Resample dates
    # patterns = read_data.resample_dates(patterns)

    # # Read social distancing data:
    # social_distancing = read_data.read_social_distancing()
    # social_distancing.to_csv('all_social_distancing_data.csv', index=False)
    social_distancing = pd.read_csv('all_social_distancing_data.csv', low_memory=False)

    # Read home panel data:
    home_pattern = read_data.read_home_pattern(county_fips, county_nms, centre_student_tractcode)

    # Read state stats
    state_stats = read_data.state_home_devices()

    cohort_average_patterns = patterns[['date_range_start','Total visitors', 'Student proportion',
                                                'Community proportion', 'proportion_max', 'Student visits', 'Community visits']].groupby(['date_range_start']).mean().reset_index()
    plot.pattern_by_mixture(cohort_average_patterns)

    cohort_average_patterns['Date'] = pd.to_datetime(cohort_average_patterns['date_range_start'], format='%Y-%m-%d')
    cohort_average_patterns.drop('date_range_start', axis=1, inplace=True)

    # Here we read Flu data
    flu = read_data.read_flu(county_nms)

    # Read flu vaccine
    flu_vaccine = read_data.read_flu_vaccine()

    # read covid cases
    covid = read_data.read_covid(county_fips, county_nms)

    # Read covid vaccine
    [covid_vaccine_step, covid_vaccine] = read_data.read_covid_vaccine(county_fips, county_nms)



    plot.plot_flu_all(flu)
    plot.plot_flu_centre(flu)
    plot.plot_flu_per_capita(flu)
    plot.plot_flu_vaccine(flu, flu_vaccine)
    plot.plot_community_flu_pattern(cohort_average_patterns, flu)
    plot.plot_student_flu_pattern(cohort_average_patterns, flu)
    plot.plot_flu_social_distancing_student(social_distancing, flu)
    plot.plot_flu_social_distancing_community(social_distancing, flu)
    plot.plot_social_distancing(social_distancing)
    plot.plot_covid_vaccination_step(covid_vaccine_step)
    plot.plot_covid_vaccination(covid_vaccine)
    plot.plot_covid_all(covid)
    plot.plot_covid_centre(covid)
    plot.plot_home_pattern(home_pattern)
    plot.plot_state_stats(state_stats)

    influenza = read_data.influenza(cohort_average_patterns, flu)















