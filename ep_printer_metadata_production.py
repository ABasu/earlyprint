#!/usr/bin/env python
# coding: utf-8

# ### Project Outline
# 
# * Create 3 new columns: `namePreprocessed`, `_personID`, `nameRegularized`
#     * The leading underscore indicates `personId` is a temporary column. The ID can just be a sequential integer as long as it's unique - it doesn't need to be a true hash.
#     * Whether we can reliably create a `nameRegularized` - i.e. a modern regular version that should replace all variations in the metadata remains to be seen. It might not be possible without a lot of human intervention. 
# * We need the following functions:
#     * `name_preprocess()`: Returns a cleaned up version of the name string or `None` if it doesn't look like a name.
#     * `substitution_cost()`: Needed for `weighted_levenshteain()`
#     * `substitution_cost_dict_generate()`: generate a cost dict for `weighted_levenshtein()`
#     * `weighted_levenshtein()`
#     * `name_pair()`: take a pair of preprocessed names and return a `true` if they are a close enough match that we should compute `weighted_levenshtein()` or `false` if we should ignore them. 
#     * `name_pair_combinations()`: Calls name pair on all possible name pairs from the df
#     * `ner_pubStmt()`: Takes the `pubStmt()` field and runs NER with `Spacy` on it.
# * Procedure:
#     * Read data and create subset for ones that don't have VIAF ID. 
#     * For each row, run `name_preprocess()`. If we get a name back, we store it in `namePreprocessed`
#         * The other rows get written out to a CSV. We need to clean up the pubStmt for these.
#     * Call the `name_pair_combinations()` function on the dataframe. 
#     * This calls `name_pair()` on all combinations of names. We keep the viable name pairs.
#     * If they pass, we run them through `weighted_levenshtein()` and store the result in a `networkx` graph where the nodes are pd.Dataframe `id` and the edges are 1/weigthed_levenshtein() \[i.e. the more similar the nodes, the higher the weight\]. If the `weighted_levenshtein()` score is above a certain threshold, we don't add it to the graph. 
#     * When we are done, we break down the graph into discrete subgraphs using [this approach](https://stackoverflow.com/questions/61536745/how-to-separate-an-unconnected-networkx-graph-into-multiple-mutually-disjoint-gr). Each subgraph will be one name in all its variant forms. This performs the clustering for us. 
#     * We sort these graphs by number of nodes and assign each of them a unique ID starting at 1 and then write everythin out to a CSV.

# In[33]:


# %load_ext line_profiler

# Set up all imports and logging
import itertools, json, logging, re, string, sys
from collections import defaultdict
from statistics import multimode

import networkx as nx
import unidecode
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from strsimpy.weighted_levenshtein import WeightedLevenshtein

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

sh = logging.StreamHandler(sys.stderr)
sh.setLevel(logging.INFO)
fmt = '%(asctime)s %(message)s'
dfmt = '%y-%m-%d  %H:%M:%S'
logging.basicConfig(handlers=(sh,), format=fmt, datefmt=dfmt, level=logging.INFO)


# ##### All adjustable parameters are set here

# In[31]:


# Data files to read
printers_data_file = 'data/printers_etc.csv'
name_abbreviations_file ='./data/name_abbreviations.json'

# We'll output bad data to these files
printers_data_file_date_notparsed_doubleyears = 'data/printers_etc_date_notparsed_doubleyears.csv'
printers_data_file_date_notparsed = 'data/printers_etc_date_notparsed.csv'
printers_data_file_pubstmt_notparsed = 'data/printers_etc_pubstmt_notparsed.csv'

# Final output file
printers_data_file_withhashes = 'data/printers_etc_withhashes.csv'

n_test = 0    # Subset of data for tests. Set to 0 or more than the total datasize to use everything.
records_with_viaf = False    # If False, we eliminate items that already have a resolved VIAF ID

# Change these values to zoom in on shorter yearspans in the visualization
start_year = 1400
end_year = 1700
records_within_datespan = True # If this is set, only the above datespan is kept

# This needs to have both (c1, c2) and (c2, c1) pairs only if the weights are different. 
# Otherwise (c2, c1) etc is generated automatically below.
substitution_cost_dict = {('i', 'j'): 0.3,
                          ('u', 'v'): 0.3,
                          ('y', 'i'): 0.3,
                          ('e', 'y'):.7}


# ##### Read external files and visualize raw data

# In[3]:


# The name abbreviations dictionary can be passed to the name_preprocess function
with open(name_abbreviations_file) as file:
    name_abbreviations = json.load(file)

printers_df = pd.read_csv(printers_data_file)

# Convert dates to numeric but leave strings and nans untouched
# Extract dates with two years - eg. 1660 1662
printers_df['dateParsed'] = printers_df['dateParsed'].apply(pd.to_numeric, errors='ignore')
printers_df_strings = printers_df[printers_df['dateParsed'].apply(lambda x: isinstance(x, str))]
logging.info(f'Writing records with two date strings to {printers_data_file_date_notparsed_doubleyears}')
printers_df_strings.to_csv(printers_data_file_date_notparsed_doubleyears)

# Extract all rows with badly parsed dates (including above rows)
printers_df['dateParsed'] = printers_df['dateParsed'].apply(pd.to_numeric, errors='coerce')
filter_baddates = printers_df['dateParsed'].isna()
printers_df_baddates = printers_df[filter_baddates]
logging.info(f'Writing records with bad date fields to {printers_data_file_date_notparsed}')
printers_df_baddates.to_csv(printers_data_file_date_notparsed)

# Retain rows with well formed dates
printers_df = printers_df[~filter_baddates]

viaf_exists = printers_df[~printers_df['viafId'].isna()]
viaf_needed = printers_df[printers_df['viafId'].isna()]

total_counts = printers_df.groupby(['dateParsed'])['dateParsed'].count()
viaf_exists_counts = viaf_exists.groupby(['dateParsed'])['dateParsed'].count()
viaf_needed_counts = viaf_needed.groupby(['dateParsed'])['dateParsed'].count()

# Set up the plot
sns.set_theme(style="darkgrid")
fig = plt.figure(figsize=(15,10))
grid = plt.GridSpec(3,1)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[2, 0])]

sns.lineplot(data=total_counts, x=total_counts.index, y=total_counts.values, color='k', ax=axes[0])
sns.lineplot(data=viaf_exists_counts, x=viaf_exists_counts.index, y=viaf_exists_counts.values, color='b', ax=axes[1])
sns.lineplot(data=viaf_needed_counts, x=viaf_needed_counts.index, y=viaf_needed_counts.values, color='r', ax=axes[2])

start_year = start_year if int(total_counts.index[0]) < start_year else int(total_counts.index[0])
end_year = end_year if int(total_counts.index[-1]) > end_year else int(total_counts.index[-1])

axes[0].set(xlim=(start_year, end_year), ylim=(1, max(total_counts[start_year:end_year].values+1)), xlabel=None, title='Total number of texts per year.')
axes[1].set(xlim=(start_year, end_year), ylim=(0, max(total_counts[start_year:end_year].values)+1), xlabel=None, title='Number of texts per year with VIAF IDs.')
axes[2].set(xlim=(start_year, end_year), ylim=(0, max(total_counts[start_year:end_year].values)+1), xlabel=None, title='Number of texts per year without VIAF IDs.')
print('Total Number of texts:         {:,}\nTexts with VIAF IDs:           {:,}\nTexts without VIAF IDs:        {:,}\nTexts with bad dates (ignored): {:,}'          .format(len(printers_df), len(viaf_exists), len(viaf_needed), len(printers_df_baddates)))


# In[4]:


# slice data to produce the final dataframe we'll work on
if records_with_viaf:
    df = printers_df
    logging.info('Including records with assigned VIAF IDs')
else:
    df = viaf_needed
    logging.info('Excluding records with assigned VIAF IDs')

if records_within_datespan:
    df = df[(df['dateParsed']>=start_year) & (df['dateParsed']<=end_year)]
    logging.info(f'Keeping records within {start_year} and {end_year}')
    
if n_test==0:
    df = df
    logging.info(f'Keeping all {len(df)} records')
else:
    df = df[:n_test]
    logging.info(f'Keeping the first {len(df)} records')
    


# In[18]:


# Functions

def name_preprocess(full_name, lower_case=False, title_case=True,                     name_abbreviations=name_abbreviations,                     max_length=30, min_tokens=2, max_tokens=3):
    """
    Takes a single name string and returns a cleaned up version of the name or 
    None if it doesn't look like a name.
    
    We remove any preceding 'me', clean up punctuation, extra spaces. Throw out 
    names that are too long or have too few or too many tokens. Substitute vv -> w.
    Also expand name abbreviations. Lower and Title case options available. Titlecase
    takes precedence.
    """
    # stop_words = ['de', 'the', 'of']
    
    full_name = unidecode.unidecode(full_name)
    # If the string splits on a period, we likely have a lastname, firstname format
    split_name = full_name.split(',')
    if len(split_name)>1:
        full_name = f'{split_name[1]} {split_name[0]}'

    # Clean punctuation
    full_name = re.sub(f'^me\s*|[{string.punctuation}]|\d*', '', full_name)
    # Collapse vv to w
    full_name = re.sub('vv', 'w', full_name)
    full_name = re.sub('VV', 'W', full_name)

    #ignore names that are too long
    if len(full_name) > max_length:
        logging.info(f'Too long: Ignoring: {full_name}')
        return None

    #find all strings separated by whitespace
    words = re.findall(r'\b\w+\b', full_name)
    # If anything other than stopwords aren't capitalized we throw it away
    # words = [w for w in words if (w[0].isupper() or w in stop_words)]
    # check if the number of words is at least 2 or over 4
    if len(words) < min_tokens:
        logging.info(f'Too few tokens: Ignoring: {full_name}')
        return None
    if len(words) > max_tokens:
        logging.info(f'Too many tokens: Ignoring: {full_name}')
        return None
    # return the first word and remaining string as a tuple
    first_name = words[0]
    if first_name in name_abbreviations:
        first_name = name_abbreviations[first_name]
    last_name = ' '.join(words[1:])
    
    if lower_case:
        first_name = first_name.lower()
        last_name = last_name.lower()
    
    if title_case:
        first_name = first_name.title()
        last_name = last_name.title()

    return (first_name, last_name)

def substitution_cost_dict_generate(substitution_cost_dict, swapcase_weight=0.2):
    """
    Generate reverse pairs for the cost dictionary. I.e. is ('u', 'v') is supplied,
    generate ('v', 'u'). ('u', 'V') and ('U', 'v') are also generated. Other letters 
    get the swapcase weight. So the cost of 'A'->'a' is swapcase_weight. This can be 
    set to zero, if you don't care about swapping cases.
    """
    reversed_substitution_cost_dict = {}

    for (c1, c2), w in substitution_cost_dict.items():
        # (i, j) -> (j, i)
        if (c2, c1) not in substitution_cost_dict:
            reversed_substitution_cost_dict[(c2, c1)] = w
        # (i, j) -> (i, J) and (I, j)
        if (c1.swapcase(), c2) not in substitution_cost_dict:
            reversed_substitution_cost_dict[(c1.swapcase(), c2)] = w
        if (c1, c2.swapcase()) not in substitution_cost_dict:
            reversed_substitution_cost_dict[(c1, c2.swapcase())] = w
        if (c1.swapcase(), c2.swapcase()) not in substitution_cost_dict:
            reversed_substitution_cost_dict[(c1.swapcase(), c2.swapcase())] = w
        # (a, A), (A, a) etc
        for l in string.ascii_letters:
            if (l, l.swapcase()) not in substitution_cost_dict:
                reversed_substitution_cost_dict[(l, l.swapcase())] = swapcase_weight

    substitution_cost_dict = {**substitution_cost_dict, **reversed_substitution_cost_dict}

    return substitution_cost_dict

def substitution_cost(x, y, substitution_cost_dict=substitution_cost_dict):
    """
    Takes a pair of letters and returns a substitution cost for them
    """
    sc = substitution_cost_dict[(x,y)] if (x, y) in substitution_cost_dict else 1.0
    return sc

def name_pair(n1, n2, letter_similarity_threshold=.5, last_name_length_similarity_threshold=3):
    """
    Takes two preprocessed names (tuples) and returns true is they are a viable match and 
    False is they are unlikely to be a match.

    If names have same or similar initials. 
    If the last name is approximately same length.
    """
    f1, l1 = n1
    f2, l2 = n2
    
    # Do they have same or similar initials
    if (((f1[0]==f2[0]) or (substitution_cost(f1[0], f2[0])<letter_similarity_threshold)) and                       ((l1[0]==l2[0]) or (substitution_cost(l1[0], l2[0])<letter_similarity_threshold))):
        if abs(len(l1)-len(l2)) <= last_name_length_similarity_threshold:
            return True
    return False

def name_pair_combinations(df, dateRange=40):
    """
    Checks all names in a dataframe against each other and returns vialble name pairs 
    as a list of ((index, name), (index, name)) tuples. Note that each name will show
    up at least once as it is similar to itself. So unique names are accounted for.
    """
    filter = []
    df = tuple(df.itertuples())
    for i, n1 in enumerate(df):
        if i%1000 == 0:
            logging.info(f'Name pairs parsed: {i}')
        for n2 in df[i:]:
            if abs(n1[1] - n2[1]) <= dateRange:
                name1 = n1[2]
                name2 = n2[2]
                index1 = n1[0]
                index2 = n2[0]
                if name_pair(name1, name2):
                    filter.append(((index1, name1), (index2, name2)))
    return filter

def weighted_levenshtein_compute(n1, n2, weighted_levenshtein_function, initial_match_weight=.2):
    f1, l1 = n1
    f2, l2 = n2

    if (f1 == f2) and (l1 == l2):
        d1 = 0
        d2 = 0
    elif (len(f1) == 1 or len(f2) == 1) and (substitution_cost(f1[0], f2[0])<1):
        d1 = initial_match_weight
        d2 = weighted_levenshtein(l1, l2)
    else:
        d1 = weighted_levenshtein(f1, f2)
        d2 = weighted_levenshtein(l1, l2)
    return (d1+d2)
  
def weighted_levenshtein_compute_sortedlist(name_pairs, reverse=False):
    distances = {}
    for ((i1, n1), (i2, n2)) in name_pairs:
        dist = weighted_levenshtein_compute(n1, n2, weighted_levenshtein)
        distances[((i1, n1), (i2, n2))] = dist
    distances = {k:v for k, v in sorted(distances.items(), key=lambda x: x[1], reverse=reverse)}
    return distances
    
def dict_to_graph(distances, cut_off=3.5, min_dist=.1):
    graph = nx.Graph()
    i = 0
    for k in distances: 
        i+=1
        if distances[k] <= cut_off:
            graph.add_edge(k[0], k[1], weight=1/(distances[k]+min_dist))
    return(graph)     
        
# We'll call the above function to expand our cost dictionary
substitution_cost_dict = substitution_cost_dict_generate(substitution_cost_dict)

# Initialize the WL function with custom weights
weighted_levenshtein = WeightedLevenshtein(substitution_cost_fn=substitution_cost).distance


# In[37]:


# Main body
# Preprocess all names and create new column for it
df['namePreprocessed'] = df['name'].map(name_preprocess)
# Separate rows that don't look like names. Write them out to file for cleaning
df_unresolved_name = df[df['namePreprocessed'].isnull()]
logging.info(f'Writing {len(df_unresolved_name)} unresolved names to {printers_data_file_pubstmt_notparsed}')
df_unresolved_name.to_csv(printers_data_file_pubstmt_notparsed)

# Keep only resolved names
df = df[~df['namePreprocessed'].isnull()]
logging.info(f'{len(df)} names resolved')

# Trim the dataframe - keep only nameProcessed and parsedDate
df_trimmed = df[['dateParsed', 'namePreprocessed']]

# Generate list of viable name pairs
name_pairs = name_pair_combinations(df_trimmed, dateRange=30)
# Compute sorted list of levenshtein distances
distances = weighted_levenshtein_compute_sortedlist(name_pairs, reverse=False)

# Convert dictionary to a graph and break into a sorted list of subgraphs
g = dict_to_graph(distances)
sub_g = sorted(nx.connected_components(g), key=len, reverse=True)
indices = [[i for (i, n) in sg]for sg in sub_g]
names = [[n for (i, n) in sg]for sg in sub_g]
for hash_ind, sub_graph in enumerate(indices):
    name_pool = names[hash_ind]
    first_names, last_names = zip(*name_pool)
    name = f'{multimode(first_names)[0]} {multimode(last_names)[0]}'
    for row in sub_graph:
        df.at[row,'indexHash'] = hash_ind


# In[32]:


#df.drop(['namePreprocessed'], axis=1)
df.to_csv(printers_data_file_withhashes)


# In[ ]:


# # Slower implementatitions of above function
# def name_pair_combinations_(df, dateRange=40):
#     """
#     Returns vialble name pairs as a list of indices. Note that each name will show
#     up at least once as it is similar to itself. So unique names are accounted for.
#     """
#     filter = []
#     for i, ni1 in enumerate(df.itertuples()):
#         if i%1000 == 0:
#             logging.info(f'Name pairs parsed: {i}')
#         for ni2 in df[i:].itertuples():
#             if abs(ni1.dateParsed - ni2.dateParsed) <= dateRange:
#                 name1 = ni1.namePreprocessed
#                 name2 = ni2.namePreprocessed
#                 index1 = ni1.Index
#                 index2 = ni2.Index
#                 if name_pair(name1, name2):
#                     filter.append(((index1, name1), (index2, name2)))
#     return filter

# def name_pair_combinations__(df):
#     df_trimmed = df.drop(['tcpid','role','role_edited','name','source','title','author','date', 'parsedDate','place','pubStmt','nameResolved','viafId'], axis=1)
#     df_trimmed['indexCopy'] = df_trimmed.index
#     dtj = df_trimmed.join(df_trimmed, how='cross', lsuffix='_1', rsuffix='_2')
#     return dtj.apply(lambda x: name_pair(x['namePreprocessed_1'], x['namePreprocessed_2']), axis=1)

