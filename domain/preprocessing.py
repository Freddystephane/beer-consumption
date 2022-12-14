import pandas as pd


# function to find which brewery produces the strongest beers by ABV%?
def strongest_brewery(df):
    return pd.DataFrame(df[df['beer_abv']==df['beer_abv'].max()].brewery_name).values()

# function to convert review_time to datetime
def convert_time(df):
    df['date'] = pd.to_datetime(df['review_time'], unit='s')
    df['year'] = df['date'].dt.year
    return df

# function to find the top beers which have the highest average review_overall, review_aroma, review_palate, review_taste and review_appearance and return in dataframe
def top_beers(df):
    df = df.groupby(['beer_beerid']).aggregate({'review_overall': 'mean', 'review_aroma': 'mean',
                                              'review_palate': 'mean', 'review_taste': 'mean',
                                              'review_appearance': 'mean'}).sort_values(by=['review_overall', 'review_aroma', 'review_palate', 'review_taste', 'review_appearance'], ascending=False)
    return df

def top_most_beers(df):
    df = top_beers(df)
    mask = (
    (df['review_overall'] == df['review_overall'].max()) &
    ((df['review_aroma'] == df['review_aroma'].max()) |
    (df['review_palate'] == df['review_palate'].max()) |
    (df['review_taste'] == df['review_taste'].max()) |
    (df['review_appearance'] == df['review_appearance'].max()))
    )
    most_beer = list(df[mask].index)
    return most_beer # df[df["beer_beerid"].isin(most_beer)]

def top_most_beers_apparence_aroma(df):
    df = top_beers(df)
    mask = (
    (df['review_aroma'] == df['review_aroma'].max()) |
    (df['review_appearance'] == df['review_appearance'].max())
    )
    most_beer = list(df[mask].index)
    return most_beer, df # df[df["beer_beerid"].isin(most_beer)]

def beer_after_date(df, year):
    mask = df["year"] > year
    df_date = df[mask]
    beer_id_2010 = list(df_date["beer_beerid"].unique())
    mask_beer = df["beer_beerid"].isin(beer_id_2010)
    return  df[mask_beer]