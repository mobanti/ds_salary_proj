# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:17:55 2020

@author: X
"""

import glassdoor_scraper as gs
import pandas as pd
path = "C:/Users/X/Dropbox/Python/ds_salary_proj/chromedriver"

df = gs.get_jobs('data scientist', 100, False, path,5 )

df