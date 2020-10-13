#!/usr/bin/env python
# coding: utf-8

# In[35]:


# build out ownership predictions
# and scores while you're at it

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
import re
import json
from selenium import webdriver
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import xgboost as xgb


# In[36]:


def get_lineup(day = datetime.today().date(), slate_id = '1', slate_type= 'MAIN'):
    
    
    # in format YYYY-MM-DD
    plyr_site = requests.get(f"https://rotogrinders.com/lineups/nba?date={day}&site=draftkings")
    plyr_soup = BeautifulSoup(plyr_site.text, 'html.parser')
    
    p_df = pd.DataFrame(columns = ['date','slate_id', 'type', 'total_OU','team','opponent','team_OU', 'player', 'starting', 'salary','position', 'proj_fpts_roto','proj_ownr_roto'])

    # scrape https://rotogrinders.com/lineups/nba?date={day}&site=draftkings for data 
    for game in plyr_soup.find_all('li', attrs = {'data-role':'lineup-card'}):
        row = {}
        row['date'] = day
        row['slate_id'] = slate_id
        row['type'] = slate_type
        row['total_OU'],favorite,row['fav_line'] = game.find('div', attrs = {'class':'ou'}).find('div', attrs = {'class': 'sum'}).text.strip().split()
        
        
        away_team = game.find('div', attrs = {'class': 'blk away-team'})
        home_team = game.find('div', attrs = {'class': 'blk home-team'})

        away_str_cnt = 1
        for player in away_team.find_all('li' ,attrs=  {'class':'player'}):
            row['team'] = game['data-away']
            row['opponent'] = game['data-home']
            if row['team'] == favorite:
                row['favorite'] = 'YES'
            else:
                row['favorite'] = 'NO'
            row['team_OU'] = game.find('div', attrs = {'class':'ou'}).find_all('div')[0].text.strip().split('\n')[0]
            try:
                row['player'] = player.find('a').text
                row['salary'] = player['data-salary']
                row['position'] = player['data-pos']
                row['proj_fpts_roto'] = player.find('span', attrs = {'class':'fpts'}).text
                row['proj_ownr_roto'] = player.find('span', attrs = {'class':'pown'}).text
            except:
                continue
            if away_str_cnt <= 5:
                row['starting'] = 'YES'
            else: 
                row['starting'] = 'NO'
            

            away_str_cnt = away_str_cnt +  1
            
            p_df = p_df.append(row, ignore_index=True)

        home_str_cnt = 1 
        for player in home_team.find_all('li' ,attrs=  {'class':'player'}):
            row['team'] = game['data-home']
            row['opponent'] = game['data-away']
            if row['team'] == favorite:
                row['favorite'] = 'YES'
            else: 
                row['favorite'] = 'NO'
            row['team_OU'] = game.find('div', attrs = {'class':'ou'}).find_all('div')[2].text.strip().split('\n')[0]
            try:
                row['player'] = player.find('a').text
                row['salary'] = player['data-salary']
                row['position'] = player['data-pos']
                row['proj_fpts_roto'] = player.find('span', attrs = {'class':'fpts'}).text
                row['proj_ownr_roto'] = player.find('span', attrs = {'class':'pown'}).text
            except:
                continue
            if home_str_cnt <= 5:
                row['starting'] = 'YES'
            else: 
                row['starting'] = 'NO'
            

            home_str_cnt = home_str_cnt + 1
            
            p_df = p_df.append(row, ignore_index=True)
            
            
    # CLEAN UP SOME COLUMNS
    p_df['total_OU'] = p_df['total_OU'].astype(float)
    p_df['team_OU'] = p_df['team_OU'].astype(float)
    p_df['salary'] = p_df['salary'].str.strip(('$K')).replace('',0).astype(float)*1000
    p_df['proj_fpts_roto'] = p_df['proj_fpts_roto'].replace('',0).astype(float)
    p_df['proj_ownr_roto'] = pd.to_numeric(p_df['proj_ownr_roto'].str.strip('%'), errors='coerce')/100
            
    return p_df



def supp_stats(lineup):
    
    pass



def get_day_ownership(dt = datetime.today().strftime('%Y-%m-%d')):
    # the date gives ownership for that day (posted on website next day)
    # might have to open up executable driver file once before
    a = datetime.strptime(dt, '%Y-%m-%d').date() - date(2020, 9, 2)
    day = a.days + 1326
    driver = webdriver.Firefox(executable_path=r'C:\Users\Luke\geckodriver.exe')
    driver.get(f"https://www.linestarapp.com/Ownership/Sport/NBA/Site/DraftKings/PID/{day}")
    try:
        projected = driver.execute_script('return actualResultsDict')
    except: 
        return None
    if len(projected) == 0 :
        return None
    
    df = pd.DataFrame(list(projected.values())[0]).drop_duplicates()
    df = df.replace(r"&#39;", "'", regex= True)
    
    return df


def build_train_set(start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    day_count = (end_date - start_date).days + 1

    final_df = pd.DataFrame()

    for single_date in (start_date + timedelta(n) for n in range(day_count)):
        single_date_ = single_date.strftime('%Y-%m-%d')
        ownership = get_day_ownership(dt = single_date_)
        if ownership is None:
            continue
        lineup = get_lineup(day = single_date_)
        merged_data = pd.merge(lineup, ownership, left_on = 'player', right_on = 'name')
        final_df = final_df.append(merged_data)
        
    target = final_df['owned']/100
    train = final_df.drop(['owned'], axis = 1)
        
    return train, target

def train_model(train, target, features = train.columns, model = LinearRegression()):
    pipe = make_pipeline(OneHotEncoder(handle_unknown = 'ignore'), model)
    mod = pipe.fit(train, target)
    return mod

def pred_ownership(model, df):
    df['predict'] = model.predict(df)
    return df


# In[16]:


X, y = build_train_set(start_date = '2020-09-05',end_date  = '2020-09-07')


# In[40]:


model = train_model(X,y)
today_slate, true_vals = build_train_set(start_date = '2020-09-08', end_date = '2020-09-08')


# In[41]:


predictions = pred_ownership(model, today_slate)
predictions['true_vals'] = true_vals
predictions


# In[42]:


get_lineup('2020-08-10')


# In[7]:


get_day_ownership(dt = '2020-08-2')


# # Scratch

# In[34]:


day = '1325'
results_site = requests.get(f"https://www.linestarapp.com/Ownership/Sport/NBA/Site/DraftKings/PID/{day}")
results_soup = BeautifulSoup(results_site.text, 'html.parser')
table = results_soup.find_all('tbody') #, attrs = {'id':'tableTournament'})

results_soup.find_all('script')


# In[98]:


test= plyr_soup.find_all('li' ,attrs=  {'class':'player'})
#json.loads(test.find('input')['value'])

test[1]

slate = '39113'
slate_type = 'MAIN'

player_df = pd.DataFrame(columns = ['date','slate_id', 'type', 'player',  'salary','position', 'proj_fpts_roto','proj_ownr_roto'])

for player in plyr_soup.find_all('li' ,attrs=  {'class':'player'}):
    row = {}
    row['date'] = datetime.today().date()
    row['slate_id'] = slate
    row['type'] = slate_type
    row['player'] = player.find('a').text
    row['salary'] = player['data-salary']
    row['position'] = player['data-pos']
    row['proj_fpts_roto'] = player.find('span', attrs = {'class':'fpts'}).text
    row['proj_ownr_roto'] = player.find('span', attrs = {'class':'pown'}).text
   
    player_df = player_df.append(row, ignore_index=True)
    

    
player_df

