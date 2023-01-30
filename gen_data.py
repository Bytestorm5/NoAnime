### I'm truly sorry this filthy content has to go onto your computer.
import requests
import json
from dotenv import load_dotenv
import os
from re import escape
from download_image import internal_download_image

#threading is weird
#currently set up just to take the first frame, in the future will implement scraping every frame  
def download_image(result):
    folder = 1 if any([tag in search_list["blacklist"] for tag in result[1]]) else 0
    try:
        return internal_download_image(result[0], folder)
    except Exception as e:
        print(e)
    finally:
        return

load_dotenv()

search_list = json.load(open("searchterms.json"))

API_KEY = os.getenv("TENOR_KEY")
ckey = "no_anime_bot"  # client key for tenor; doesn't seem to matter what I put here

def search_gifs(search_term, lmt = 8):      

    # get the top 8 GIFs for the search term
    r = requests.get(
        "https://tenor.googleapis.com/v2/search?q=%s&key=%s&client_key=%s&limit=%s" % (search_term, API_KEY, ckey, lmt))

    print(r.status_code)
    
    if r.status_code == 200:
        # load the GIFs using the urls for the smaller GIF sizes
        top_8gifs = json.loads(r.content)
        results = []
        for gif in top_8gifs['results']:
            url = gif['media_formats']['gif']['url']
            tags = gif['tags']
            tags.append(search_term)            
            results.append((url, tags))
        return results
        print(top_8gifs)
    else:
        top_8gifs = None

def extract_all_for_term(term, limit=8):
    r = search_gifs(term, limit)
    [download_image(result) for result in r]

if __name__ == "__main__":
    GIFS_PER_TERM = 16

    term_skew = len(search_list['blacklist']) / len(search_list['whitelist'])
    vert = True
    if term_skew < 1:
        term_skew = 1 / term_skew
        vert = False

    print(f"Term Skew: {term_skew}")
    print(f"{len(search_list['blacklist'])} Blacklisted Terms: {search_list['blacklist']}")
    print(f"{len(search_list['whitelist'])} Whitelisted Terms: {search_list['whitelist']}")

    for term in search_list['whitelist']:
        extract_all_for_term(term, GIFS_PER_TERM if vert else int(term_skew * GIFS_PER_TERM))
    for term in search_list['blacklist']:
        extract_all_for_term(term, int(term_skew * GIFS_PER_TERM) if vert else GIFS_PER_TERM)