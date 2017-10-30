# Author: Jake Toffler
# Date: 10/30/2017

import pandas
import re
import requests
import os

from bs4 import BeautifulSoup

# Don't display too much data
pandas.options.display.max_rows = 4

"""
Pitchers' Repertoires

For all pitchers in the Brooks Baseball database, find how many pitches they have. This information may be used as
a sort of "response variable" to complement Sig's data.

Once we have a list of pitchers and the number of pitches they throw, we can compare this to the pitchers
in Sig's data and see how similar they are.  Hopefully, they will be similar which means we can use this as a way
to train our model to count the number of clusters.
"""


def playerIDs(filename):
    """
    This function inputs the playerID csv found here: http://legacy.baseballprospectus.com/sortable/playerid_list.php,
    opens it as a Pandas dataframe, and outputs a list of playerIDs used in the Brooks Baseball url.
    The list will inevitably change.  If you need to re-run this program, I recommend keeping the old player ID list,
    downloading the new list, and running it on only the player IDs that are in the new list and not the old list.
    That said, if a player's repertoire changes, this method could miss that change.
    """
    table = pandas.read_csv(filename)
    playerID_list = table['MLBCODE']
    return playerID_list



def dictionaryInitializer(filename):
    """
    This function inputs a file and checks to see whether or not it exists.  If it does, it will take the text of that
    file and turn it into a dictionary.  Otherwise, it will create an empty dictionary.
    This dictionary will store every pitcher the pitches they throw, according to Brooks Baseball.
    """
    if os.path.isfile(filename):
        f = open(filename, "r")
        pitcherDict = eval(f.read())
        f.close
    else:
        pitcherDict = {}
    return pitcherDict


def repertoireScraper(list, dict):
    """
    This function inputs a list of player IDs (each ID is a unique 6-digit MLBCODE) and a dictionary and either adds or
    updates the pitcher name and pitch repertoire to that dictionary.
    Note that if one of the IDs in the list is for a non-pitcher, their name will be 'None' and their repertoire will be
    an empty list.  This program handles that by creating and subsequently overwriting a "None" entry for any non-pitcher.
    """
    for id in list:
        r = requests.get("http://www.brooksbaseball.net/landing.php?player="+str(id))
        soup = BeautifulSoup(r.text, "html.parser")

        # Initialize the pitcher's name and an empty list on every iteration
        name = None
        pitches = []

        # Names are stored in the "h1" tag.  Find the "h1" tag and store the text inside of it
        for pitcher in soup.find_all("h1"):
            name = pitcher.contents[0].strip()

        # Pitches are shown with unique font on the website so they can be found using the "font" tag.  Store the text inside of it
        for pitch in soup.find_all("font"):
            pitches.append(pitch.contents[0].lower())

        # Create a list of unique pitches
        dict[name] = set(pitches)
    return dict


def outputGenerator(filename, dict):
    """
    This function inputs the name of a file and a dictionary.  If the file does not exist, it will create the file and
    write the dictionary as a string in a .txt file (which can be used as an input in the dictionaryInitializer function).
    If the file already exists, it will overwrite the file.
    """
    f = open(filename, "w+")
    f.write(str(dict))
    f.close()


playerID_file = ("playerid_list.csv")
pitcherDict_file = ("pitcherDict.txt")


pitcherDict = dictionaryInitializer(pitcherDict_file)
playerID_list = (playerIDs(playerID_file))

repertoireDict = repertoireScraper(playerID_list, pitcherDict)

outputGenerator(pitcherDict_file, repertoireDict)