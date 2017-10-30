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

For all pitchers in a given pitchFX dataset, find how many pitches they have according to the Brooks Baseball database. 

In my context, this information will be used to compare the number of pitches a player has according to Brooks Baseball
against the number of pitches a player has according to my dataset.  Hopefully, they will be similar which means I will
be able to use this as a response variable to train my clustering model on how to count the number of pitch clusters a 
given pitcher should have.
"""


#def playerIDs(filename):
#    """
#    This function inputs the playerID csv found here: http://legacy.baseballprospectus.com/sortable/playerid_list.php,
#    opens it as a Pandas dataframe, and outputs a list of playerIDs used in the Brooks Baseball url.
#    The list will inevitably change.  If you need to re-run this program, I recommend keeping the old player ID list,
#    downloading the new list, and running it on only the player IDs that are in the new list and not the old list.
#    That said, if a player's repertoire changes, this method could miss that change.
#    """
#    table = pandas.read_csv(filename)
#    player_table = table[['LASTNAME', 'FIRSTNAME', 'MLBCODE']]
#    return player_table

def pitcherIDs(filename):
    """
    This function inputs the pitchFX dataset (which already includes the pitcher's ID) and outputs a list of unique IDs
    """
    table = pandas.read_csv(filename)
    pitchers = table['pitcher_id'].drop_duplicates()
    return pitchers


def dictionaryInitializer(filename):
    """
    This function inputs a text file and checks to see whether or not it exists.  If it does, it will take the text of
    that file and evaluate it.  The text of this fileshould be a dictionary turned into a string.  If you have not
    already created this file, it will initiate an empty dictionary.
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


# Change your filepaths here
pitchers_file = "AL Pitchers.csv"
pitcherDict_file = "pitcherDict.txt"

# These are the steps to run the program.  This should not need to be changed if the filepaths are correct.
pitchers_list = (pitcherIDs(pitchers_file))
pitcherDict = dictionaryInitializer(pitcherDict_file)
repertoireDict = repertoireScraper(pitchers_list, pitcherDict)
outputGenerator(pitcherDict_file, repertoireDict)
