import read_date
import pandas as pd
import os

def main():

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
        "420270126002"]

    read_data.population(county_nms, county_fips, centre_student_tractcode)



