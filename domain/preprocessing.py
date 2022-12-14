import pandas as pd
import numpy as np 
import matplotlib
def  bewery_analyse(data):
    hightest=pd.DataFrame(data.groupby(['brewery_name']).beer_abv.nunique().sort_values(ascending=False).iloc[0:1])
    return hightest