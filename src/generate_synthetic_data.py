import os
import sys
import ast
import pandas as pd
import itertools
from nltk.corpus import wordnet
from tqdm import tqdm
from functools import partialmethod
from neuspell.noising import ProbabilisticCharacterReplacementNoiser
from itertools import combinations
from itertools import product
from contextlib import contextmanager


word_repl_noiser = ProbabilisticCharacterReplacementNoiser(language="english")
word_repl_noiser.load_resources()

with open("synthetic_cats.txt") as file:
    cats = ast.literal_eval(file.read()) 

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stderr = old_stderr

def generate_spelling_errors(category):
	unique_errors = set()
	for i in range(5):
	    noisy_text = word_repl_noiser.noise([category])[0]
	    if noisy_text != category:
	        unique_errors.add(noisy_text)
	return list(unique_errors)


def generate_synonyms(category):
	syn_objs = wordnet.synsets(category)
	if len(syn_objs) == 0:
	    return []
	synonyms = set()
	for obj in syn_objs[0].lemmas():
	    syn = obj.name()
	    if syn!=category:
	        synonyms.add(syn)
	return list(synonyms)

def generate_combinations_synonyms(row):
	anchor = [row['Category_Set']]
	target = row['synonyms']
	return list(product(anchor,target))

def generate_combinations_misspelled(row):
	anchor = [row['Category_Set']]
	target = row['misspelled']
	return list(product(anchor,target))

def block_print():
    sys.stdout = open(os.devnull, 'w')
    
def enable_print():
    sys.stdout = sys.__stdout__

path = "No_Dup/no_d/no_d/"
synthetic_data = list()
processed = set()
block_print()
with suppress_stderr():
	for index, file in tqdm(enumerate(cats)): 
	    filename = file[0]
	    df = pd.read_csv(path+filename)
	    df['Category_Set'] = df['Category_Set'].astype(str)
	    df['Category_Set'] = df['Category_Set'].apply(lambda x:x.replace(' ','_'))
	    df['Category_Set'] = df['Category_Set'].apply(lambda x:x.lower())
	    df['misspelled']= df['Category_Set'].apply(generate_spelling_errors)
	    df['synonyms'] = df['Category_Set'].apply(generate_synonyms)
	    df['misspelled'] = df.apply(generate_combinations_misspelled,axis=1)
	    df['synonyms'] = df.apply(generate_combinations_synonyms,axis=1)
	    df['synonyms'] = df['synonyms'].map(lambda x:list(map(list, x)))
	    df['misspelled'] = df['misspelled'].map(lambda x:list(map(list, x)))
	    for syn_set in df['synonyms'].tolist():
	        for pair in syn_set:
	            if len(pair) > 0 and (pair[0],pair[1]) not in processed:
	                synthetic_data.append([pair[0],pair[1],1,filename])
	                processed.add((pair[0],pair[1]))
	    for syn_set in df['misspelled'].tolist():
	        for pair in syn_set:
	            if len(pair) > 0 and (pair[0],pair[1]) not in processed:
	                synthetic_data.append([pair[0],pair[1],1,filename])
	                processed.add((pair[0],pair[1]))

synthetic_df = pd.DataFrame(synthetic_data,columns=['cat0','cat1','label','filename'])
synthetic_df.to_csv("synthetic_positives.csv",index=False)	