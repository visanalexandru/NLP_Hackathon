import json
import math
import random
import numpy as np
import string
from gensim.models import Word2Vec


def build_dictionary():
    obj=json.load(open("train.json","r"))
    d={}

    for element in obj:
        length=len(element["ner_tags"])
        current_word=[];
        for x in range(length):
            tag=element["ner_tags"][x]
            if tag not in d:
                d[tag]=[]
            word=element["tokens"][x]

            if word not in string.punctuation:
                d[tag].append(word)

    return d

def eulerian_distance(a,b):
    return np.linalg.norm(a-b)

def find_closest(model,median_dict,word):
    if word in string.punctuation:
        return "O"
    try:
        vec=model.wv[word]
    except:
        return "O"

    minimum_distance=eulerian_distance(median_dict["O"],vec)
    best="O"

    for category in median_dict:
        distance=eulerian_distance(median_dict[category],vec)
        if distance<minimum_distance:
            minimum_distance=distance
            best=category
    return best 



d=build_dictionary()

model = Word2Vec.load('FT_SG_300_25_20/FT_SG_300_25_20.model')
print("Loaded model")

median_dict={} 

for category in d:
    median=np.zeros(300) 
    count=0

    for word in d[category]:
        try:
            median+=model.wv[word]
            count+=1
        except:
            pass

    median/= count
    median_dict[category]=median


## Now build the test results
category_to_id={
        "O":0,
        "PERSON":1,
        "QUANTITY":12,
        "NUMERIC":13,
        "NAT_REL_POL":5,
        "GPE":3,
        "DATETIME":9,
        "ORG":2,
        "PERIOD":10,
        "EVENT":6,
        "FACILITY":15,
        "ORDINAL":14,
        "LOC":4,
        "MONEY":11,
        "WORK_OF_ART":8,
        "LANGUAGE":7,
        }


test_obj=json.load(open("test.json","r"))
result=open("result.csv","w")
id=0


result.write("Id,ner_label\n")

for element in test_obj:
    for word in element["tokens"]:
        closest=find_closest(model,median_dict,word) 
        result.write(f"{id},{category_to_id[closest]}\n")
        id+=1

result.close()
