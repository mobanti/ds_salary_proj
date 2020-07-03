# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:38:34 2020

@author: X
"""

import pandas as pd
df = pd.read_csv('glassdoor_jobs.csv')

#Salary parsing
df['Hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0) 
df['Employer_Provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salaary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K', '').replace('$','').replace('CA',''))
min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('emploer provided salary:',''))
df['Min_Salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['Max_Salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['Average_Salary'] = (df['Min_Salary'] + df['Max_Salary']) /2


#Company name text only
df['Company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

#Job Location
df['Location'].value_counts()
df['Same_City'] = df.apply(lambda x: 1 if x['Location'] in x['Headquarters'] else 0, axis = 1)


#Age of company
df['Age'] = df['Founded'].apply(lambda x: x if x < 1 else 2020 - x)

# Parsing of job description (python,etc)

#Python
df['Python'] =df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
#r sudio
df['R_Studio'] =df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

#spark
df['Spark'] =df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

#aws
df['AWS'] =df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

#excel
df['Excel'] =df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

df_out = df

df_out.to_csv('Salary_Data_Cleaned.csv',index = False)