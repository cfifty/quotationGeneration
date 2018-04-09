import requests
import json
import nltk
from bs4 import BeautifulSoup

trump = 'trumpquotes.csv'
cicero = 'ciceroquotes.csv'
def getTrumpQuotes(numpages = 38):
    ''' Uses api requests to find quotes by Donny.'''
    quoteArray = []
    base_url = 'https://www.brainyquote.com/api/inf'
    for x in range(numpages):
        payload = {"typ":"author","langc":"en","v":"7.5.1b:2722490","ab":"a","pg": x + 1,"id":"126655","vid":"054b0c1d882b9119b7e2e8dc95f4be4f","fdd":"d","m":0}
        response = requests.post(url=base_url, json=payload)

        if 'message' in response.json():
            return quoteArray

        data = json.loads(response.content)
        soup = BeautifulSoup(data.get('content'), 'lxml')

        # Populate quoteArray + encode
        for item in soup.find_all("a", class_="b-qt"):
            sent_text = nltk.sent_tokenize(item.get_text().rstrip().encode("ascii"))
            quoteArray.extend(sent_text)

    return quoteArray

def getCiceroQuotes(numpages = 38):
    ''' Uses api requests to find quotes by Cicero.'''
    quoteArray = []
    base_url = 'https://www.brainyquote.com/api/inf'
    for x in range(numpages):
        payload = {"typ":"author","langc":"en","v":"7.5.1b:2722490","ab":"a","pg": x + 1,"id":"129885","vid":"d4477d1832ad735567db8756c5e266f9","fdd":"d","m":0}
        response = requests.post(url=base_url, json=payload)

        if 'message' in response.json():
            return quoteArray

        data = json.loads(response.content)
        soup = BeautifulSoup(data.get('content'), 'lxml')

        # Populate quoteArray + encode
        for item in soup.find_all("a", class_="b-qt"):
            sent_text = nltk.sent_tokenize(item.get_text().rstrip().encode("ascii"))
            quoteArray.extend(sent_text)

    return quoteArray

def makeFile(quotelist, name):
    with open(name, "wb") as f:
        for index, q in enumerate(quotelist[2:]):
            f.write('"' + q + '"')
            if index != len(quotelist[2:]) - 1:
                f.write(',')
            f.write('\n')

if  __name__ =='__main__':
    makeFile(getTrumpQuotes(), trump)
    makeFile(getCiceroQuotes(), cicero)
    print('Done collecting quotes')
