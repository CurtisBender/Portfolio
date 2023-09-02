import pandas
import pandas as pd
pd.options.mode.chained_assignment = None
import scipy.stats as st
import numpy as np

#Create a dataframe by reading csv
df = pd.read_csv(r'C:\Users\bendc\OneDrive\Documents\Free Throws\free_throws.csv')

#Remove unneeded columns
df.drop(['end_result', 'game', 'game_id', 'period', 'playoffs', 'score', 'season', 'time'], axis='columns', inplace=True)

#Remove any free throws that are not free throw 1 of 2 followed by free throw 2 of 2 by the same player
df['check'] = ((df['play'].str.contains("2 of 2") & df.play.shift().str.contains("1 of 2") &
                df.player.eq(df.player.shift())) |
               (df['play'].str.contains("1 of 2") & df.play.shift(-1).str.contains("2 of 2") &
                df.player.eq(df.player.shift(-1))))

df = df[df['check'] == True]

df.drop('check', axis='columns', inplace=True)

#Create separate dataframes for first and second shots
df_first = df[df['play'].str.contains('1 of 2')]
df_second = df[df['play'].str.contains('2 of 2')]

#Reset the dataframe indices
df_first = df_first.reset_index()
df_first.drop('index', axis='columns', inplace=True)
df_second = df_second.reset_index()
df_second.drop('index', axis='columns', inplace=True)

#Remove the play column
df_first = df_first.drop('play', axis='columns')
df_second = df_second.drop('play', axis='columns')

#Rename the shot_made columns to shot1 and shot2
df_first = df_first.rename(columns={'shot_made': 'shot1'})
df_second = df_second.rename(columns={'shot_made': 'shot2'})

#Combine the first and second shot dataframes to create a dataframe of shot pairs
df_pairs = pd.concat([df_first, df_second], axis=1)
df_pairs = df_pairs.loc[:,~df_pairs.columns.duplicated()]

#Create separate dataframes for shot pairs where the first shot was missed and first shot was made
df_missed_first = df_pairs[df_pairs['shot1']==0]
df_made_first = df_pairs[df_pairs['shot1']==1]

#Add a column that has the count of free throw attempts where the first was missed/made
df_missed_first['missed_first']=1
df_made_first['made_first']=1

#Group the individual free throw attempts by player name and sum the columns
df_missed_first = df_missed_first.groupby('player', as_index=False).sum()
df_made_first = df_made_first.groupby('player', as_index=False).sum()

#Rename the shot2 columns to missed_1st_made_2nd and made_1st_made_2nd
df_missed_first = df_missed_first.rename(columns={'shot2': 'missed_1st_made_2nd'})
df_made_first = df_made_first.rename(columns={'shot2': 'made_1st_made_2nd'})

#Remove the shot1 column
df_missed_first.drop('shot1', axis='columns', inplace=True)
df_made_first.drop('shot1', axis='columns', inplace=True)

#Merge the missed first and made first dataframes by player
df_final = pd.merge(df_missed_first, df_made_first, on="player")

#Remove players that had less than 30 free throws where they missed/made the first shot
df_final = df_final[df_final['missed_first'] >= 30]
df_final = df_final[df_final['made_first'] >= 30]

#Calculate the p-value for each player
#Null hypothesis is that the result of the first free throw shot does not affects the result of the second free throw shot
df_final['theta_hat_1'] = df_final['missed_1st_made_2nd'] / df_final['missed_first']

df_final['theta_hat_2'] = df_final['made_1st_made_2nd'] / df_final['made_first']

df_final['theta_tilde'] = ((df_final['missed_first'] * df_final['theta_hat_1'] + df_final['made_first'] * df_final['theta_hat_2']) /
                           (df_final['missed_first'] + df_final['made_first']))

df_final['d'] = np.abs((df_final['theta_hat_1'] - df_final['theta_hat_2']) /
                 np.sqrt(df_final['theta_tilde'] * (1-df_final['theta_tilde']) * (1/df_final['missed_first'] + 1/df_final['made_first'])))

df_final['p-val'] = 2 * (1 - st.norm.cdf(df_final['d']))

#Count the total number of players that had a large enough sample size
total_num_players = df_final['player'].count()

#Count the number of players that had a statistically significant difference between making and missing the 1st free throw
df_stat_sig = df_final[df_final['p-val'] <= 0.05]
num_stat_sig = df_stat_sig['player'].count()

#Count the number of players that are more likely to make the 2nd free throw after making/missing the first
num_hit_1st_better = df_stat_sig[(df_stat_sig['theta_hat_1'] - df_stat_sig['theta_hat_2']) < 0]['player'].count()

num_hit_1st_worse = df_stat_sig[(df_stat_sig['theta_hat_1'] - df_stat_sig['theta_hat_2']) > 0]['player'].count()

#Create a dataframe that contains the results
results = [total_num_players, num_stat_sig, num_hit_1st_better, num_hit_1st_worse]
df_results = pandas.DataFrame([results], columns=['Total Players', 'Statistically Significant', 'Better to Make 1st', 'Better to Miss 1st'])

print(df_results.iloc[0])