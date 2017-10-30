# Author: Jake Toffler
# Github: https://github.com/jaketoffler
# Date: 10/30/2017
"""
Repertoire Comparison

For each pitcher in a PitchFX dataset, compare the number of pitches he has according to that dataset against the
number of pitches he has according to Brooks Baseball.

To create the text file which stores a dictionary of each pitcher and their repertoire, see repertoires.py.
"""

import pandas

# Don't display too much data
#pandas.options.display.max_rows = 4


# This is the number of pitches according to Brooks Baseball
f = open("pitcherDict.txt")
pitcherDict = eval(f.read())
f.close


BBList = []
for pitcher in pitcherDict:
    if pitcher != None:
        pitches = len(pitcherDict[pitcher])
        names = pitcher.split(' ')
        pitchCount = (names[1] + ', ' + names[0], pitches)
        BBList.append(pitchCount)
BBPitches = pandas.DataFrame.from_records(BBList, columns = ("pitcher_name", "pitches"))


# This is the number of pitches according to PitchFX.  Note that since we are taking the last value of each duplicate
# pitcher name, the dataset must be in ascending order by cluster number.
pitchTable = pandas.read_csv("AL Pitchers.csv")
FXPitches = pitchTable[["pitcher_name", "cluster_num"]].drop_duplicates(subset = "pitcher_name", keep = "last")


combinedPitches = pandas.merge(left = BBPitches, right = FXPitches, how = "left", on = "pitcher_name")
print combinedPitches