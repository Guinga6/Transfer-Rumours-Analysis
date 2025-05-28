import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time




class Player:
    def __init__(self, player, post, age, market_value, nat, joined, left, fee):
        self.PlayerName = player
        self.Post = post
        self.Age = age
        self.MarketValue = market_value
        self.Nationality= nat
        self.Left = left
        self.TeamJoined = joined
        self.Fee = fee
       


def get_toptransfer_data(page):
    # season = list(range(2022,2025))
    url=f"https://www.transfermarkt.com/transfers/saisontransfers/statistik/top/plus/1/galerie/0?saison_id=2024&transferfenster=alle&land_id=&ausrichtung=&spielerposition_id=&altersklasse=&leihe=&page={page}"
    headers= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
                  "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1",
                  "Connection":"close", "Upgrade-Insecure-Requests":"100"}
    page= requests.get(url, headers= headers)
    soup= BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table', class_='items')
    rows = table.find_all('tr', class_=['odd', 'even'])
    players = []
    for row in rows:
        
        name=row.find_all('td')[1].find('a').text
        post=row.find_all('tr')[1].find('td').text
        age=row.find_all('td')[5].text
        market_value=row.find_all('td')[6].text.replace('€', '').strip()
        nat=row.find_all('td')[7].find('img')['alt']
        left=row.find_all('td', {'class':'hauptlink'})[1].find('a')['title']
        joined=row.find_all('td', {'class':'hauptlink'})[2].find('a')['title']
        fee=row.find('td', {'class':'rechts hauptlink'})
        if fee is not None:
            fee = str(fee.text.replace('€', '').strip())
        else:
            fee = str(0)
        player = Player(name, post, age, market_value, nat, joined, left, fee)
        players.append(player)
    players = [vars(player) for player in players]
    df = pd.DataFrame(players)
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
    df = [get_toptransfer_data(i) for i in range(1,80)]
    df=pd.concat(df)
    df.index = range(1, len(df) + 1)
    return df


def transform(dataf):
    dataf['MarketValue'] = dataf['MarketValue'].apply(convertir_valeur)
    dataf['Fee'] = dataf['Fee'].apply(convertir_valeur)
    return dataf

def print_done():
    print('done')

