import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import os
import re



class Rumours:
    def __init__(self, player_profil, img_current_club, img_interested_club, player, img_player, market_value,current_team, interested_club, rumourcreated):
        self.PlayerProfil = player_profil
        self.ImgCurrentClub = img_current_club
        self.ImgInterestedClub = img_interested_club
        self.PlayerName = player
        self.ImgPlayer = img_player
        self.MarketValue = market_value
        self.CurrentTeam = current_team
        self.InterestedClub = interested_club
        self.RumourCreated = rumourcreated


def get_rumour_data(page): 
    # n is the number of pages to scrape
    print("Working on page: ", page)
    
    url=f"https://www.transfermarkt.com/international-rumour-mill/detail/forum/343/page/{page}"
    headers= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
                "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1",
                "Connection":"close", "Upgrade-Insecure-Requests":"100"}
    page= requests.get(url, headers= headers, verify=False)
    soup= BeautifulSoup(page.content, 'html.parser')
    rows = soup.find_all('article', class_='thread')
    rumours = []
    for row in rows[2::]:
        
        player=row.find('div', {'class' :'spielername'}).find('a').text
        player_url=row.find('a', {'class' :'link-zur-gk'})['href']
        player_profil = "https://www.transfermarkt.com"+player_url
        img_player=row.find('img', {'class' :'bilderrahmen-fixed'})['src']
        market_value=row.find('div', {'class' : 'marktwertanzeige hide-for-small'}).find('strong').text
        actual_team=row.find('div', {'class' : 'vereinname'}).find('a').text
        fromto=row.find('div', {'class': 'wechsel-verein-name'}).find('a').text
        rumourcreated=row.find('span', {'itemprop':"datePublished"}).text
        about2clubs = row.find_all('div', {'class' :'gk-wappen'})
        img_current_club=about2clubs[0].find('img')['src']
        img_interested_club=about2clubs[1].find('img')['src']
        current_team = about2clubs[0].find('img')['alt']
        interested_club = about2clubs[1].find('img')['alt']
    
        rumour = Rumours(player_profil, img_current_club, img_interested_club, player, img_player, market_value,current_team, interested_club, rumourcreated)
        rumours.append(rumour)
    rumours = [vars(rumour) for rumour in rumours]
    df = pd.DataFrame(rumours)
    return df

def convertir_valeur(valeur):
   if isinstance(valeur, str):
       valeur_clean = valeur.strip().lower()
       
       if valeur_clean in ['-', '', 'nan', 'none', 'null', 'free transfer', 'unknown']:
           return None
       
       if 'loan fee' in valeur_clean:
           if ':' in valeur_clean:
               value_part = valeur_clean.split(":")[1].strip()
           else:
               value_part = valeur_clean.replace('loan fee', '').strip()
       else:
           value_part = valeur_clean
       
       value_part = value_part.replace('€', '').replace('$', '').replace('£', '').replace(',', '').strip()
       
       if not value_part or value_part == '-':
           return None
       
       if 'k' in value_part:
           try:
               numeric_value = float(value_part.replace('k', ''))
               return numeric_value / 1000
           except ValueError:
               return None
               
       elif 'm' in value_part:
           try:
               numeric_value = float(value_part.replace('m', ''))
               return numeric_value
           except ValueError:
               return None
               
       else:
           try:
               return float(value_part)
           except ValueError:
               return None
   
   elif valeur is None:
       return None
   else:
       try:
           return float(valeur)
       except (ValueError, TypeError):
           return None



def get_data():
    df = [get_rumour_data(i) for i in range(1,400)]
    df=pd.concat(df)
    df.index = range(1, len(df) + 1)
    return df


def transform(dataf):
    # dataf['MarketValue'] = pd.to_numeric(dataf['MarketValue'], errors='coerce')
    dataf['MarketValue'] = dataf['MarketValue'].apply(convertir_valeur)
    return dataf


def print_done():
    print('done')

