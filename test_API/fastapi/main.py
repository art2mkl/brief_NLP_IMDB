#shell : uvicorn main:app --reload

from fastapi import FastAPI
import pandas as pd
import json

#scrap
import requests
from bs4 import BeautifulSoup

app = FastAPI()

#SCRAP IMDB
def scraping():   
    #-------------------------------------------------------
    """ scrap data on url imdb
    return dataframe """
    #-------------------------------------------------------

    liste_synopsis = []
    titres = []
    summary = []

    for i in range(1, 1001, 50):

        #avancement
        print(f"Scrap de la partie {i} sur {i+49}")

        # connect db
        url = f"https://www.imdb.com/search/title/?title_type=feature&num_votes=5000,&sort=user_rating,desc&start={i}&ref_=adv_nxt"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        soup = soup.find_all(class_='lister-item-content')

        # scrap summary
        summary += [element.find_all("p", class_='text-muted')[-1].text.replace(
            "\n", "").replace("See full summary\xa0»\n')", "") for element in soup]

        # scrap movies href
        liens = [element.find("a")['href'] for element in soup]

        for lien in liens:

            # connect href
            url_2 = f'https://www.imdb.com{lien}?ref_=adv_li_tt'
            response_2 = requests.get(url_2)
            soup_2 = BeautifulSoup(response_2.content, 'html.parser')

            # scrap synopsis
            liste_synopsis.append(soup_2.find(
                class_='inline canwrap').text.replace("\n", "").strip())

            # scrap titles
            titres.append(soup_2.find('h1').text.replace('\xa0', '').split('(')[0])
            
    return pd.DataFrame({'titres': titres, 'synopsis': liste_synopsis, 'resume': summary})
  
@app.get('/data/')

async def data():
    #-------------------------------------------------------
    """ read datafrom .csv
    return json """
    #-------------------------------------------------------
    #Launch scrapping
    #df = scraping()
    
    #save in csv
    #df.to_csv("imdb_1000.csv", index = False)
    
    #read_csv
    df = pd.read_csv('imdb_1000.csv')
    return json.loads(df.to_json())

# reception post method
# @app.post('/add/')
# async def add_line(data2):
#     return data2
