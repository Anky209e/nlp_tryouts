import json
import os

with open("wiki_data.json","r+") as f:
    data = json.load(f)

sents = ""
for obj in data:
    sents += "\n"
    sents += obj["text"]

with open("wiki_data.txt",'w+') as g:
    g.write(sents)