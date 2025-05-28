import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import os
import re
import time

class RumoursDetail:
    def __init__(self, player, current_club, interested_club, rumour_source, rumour_text ):
        self.Player_name = player
        self.Current_club = current_club
        self.Interested_club = interested_club
        self.Rumour_source = rumour_source
        self.Rumour_text = rumour_text        

# def player_profile(page):

#     # page is the number of pages to scrape
#     url=f"https://www.transfermarkt.com/international-rumour-mill/detail/forum/343/page/{page}"
#     headers= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
#                 "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1",
#                 "Connection":"close", "Upgrade-Insecure-Requests":"100"}
#     page= requests.get(url, headers= headers, verify=False)
#     soup= BeautifulSoup(page.content, 'html.parser')
#     hrefs =soup.find_all('a', {'class' :'link-zur-gk'})
#     for href in hrefs:
#         player_url = (href['href'])
        
#         playerurl = ('https://www.transfermarkt.com'+player_url)
#         return playerurl

    

def get_rumour_detail_data(page): 
    # page is the number of pages to scrape

    url=f"https://www.transfermarkt.com/international-rumour-mill/detail/forum/343/page/{page}"
    headers= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
                "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1",
                "Connection":"close", "Upgrade-Insecure-Requests":"100"}
    
    page= requests.get(url, headers= headers, verify=False)
    soup= BeautifulSoup(page.content, 'html.parser')
    hrefs =soup.find_all('a', {'class' :'link-zur-gk'})
    rumours_detail = []
    i = 0
    for href in hrefs[1:8]:
        i+= 1
        if i % 4 == 0:
            time.sleep(100)

        playerurl = 'https://www.transfermarkt.com'+ href['href']
        headers= {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
                    "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1",
                    "Connection":"close", "Upgrade-Insecure-Requests":"100"}
        
        time.sleep(20)
        page= requests.get(playerurl, headers= headers, verify=False)
        soup= BeautifulSoup(page.content, 'html.parser')
        # service = soup.find('h1').text
        if page.status_code != 503:
            print("⚒️ Working...")
            player = soup.find('div',{'class':'spielername-profil'}).find('a').text
            print("📊 Extracted data for:",playerurl)
            print(len(soup.find_all('td')))

            current_club = soup.find_all('td')[7].find('a').text
            interested_club = soup.find_all('td')[8]
            if interested_club is not None:
                interested_club = interested_club.find('a')
                print("Interested club:", interested_club)
                if interested_club is not None:
                    interested_club = interested_club.text.strip()
                else:
                    interested_club = "interested club not found"
            else:
                interested_club = "interested club not found"
            rumour_source = soup.find('div', {'class':'box-header-forum box-border-top box-border-left box-border-right'}).find('a').text.strip()
            rumour_text = str(soup.find('div', {'class':'content box-border-left box-border-right'}).text.replace(',','').replace('’','').replace('\n',' ').replace('\r',' ').replace('\t',' '))
            
            rumour_detail = RumoursDetail(player, current_club, interested_club, rumour_source, rumour_text )
            rumours_detail.append(rumour_detail)
        else:
            print(len(service))
            print("❌ 503 Service Unavailable")
        


    rumours_detail = [vars(rumour) for rumour in rumours_detail]
    df = pd.DataFrame(rumours_detail)
    return df


# def get_player_profile():
#     urls = [player_profile(i) for i in range(1,5)]
#     return urls

def get_data():
    df = [get_rumour_detail_data(i) for i in range(400,410)]
    df=pd.concat(df)
    print("This is the dataframe in get_data",df)
    df.index = range(1, len(df) + 1)
    return df



def print_done():
    print('done')

# AliGuinga@ensa541.onmicrosoft.com