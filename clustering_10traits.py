#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:40:48 2021

@author: bartek
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt 
 
filepath = ''
df = pd.read_csv(filepath, delimiter='\t')
columns = df.columns # describes columns
x = df[df.columns[0:50]] # trimming columns
pd.set_option("display.max_columns", None) #changes the number of displayed columns, set to display all columns here
x = x.fillna(0) # fills empty values with 0s

kmeans = MiniBatchKMeans(n_clusters=10, random_state=0,batch_size=100, max_iter=100).fit(x)
len(kmeans.cluster_centers_)

#%% Cluster centroids (50) from kmeans are assigned to 10 abstractual personality traits
#assigned our 10 resulting clusters into 10 variables 

one = kmeans.cluster_centers_[0] #personality type 1
two = kmeans.cluster_centers_[1]
three = kmeans.cluster_centers_[2]
four = kmeans.cluster_centers_[3]
five = kmeans.cluster_centers_[4]
six = kmeans.cluster_centers_[5]
seven = kmeans.cluster_centers_[6]
eight = kmeans.cluster_centers_[7]
nine = kmeans.cluster_centers_[8]
ten = kmeans.cluster_centers_[9]



#example of what a single cluster dict looks like

# one_scores = {}

# one_scores['extroversion_score'] = one[0] - one[1] + one[2] - one[3] + one[4] - one[5] + one[6] - one[7] + one[8] - one[9]
# one_scores['neuroticism_score'] = one[0] - one[1] + one[2] - one[3] + one[4] + one[5] + one[6] + one[7] + one[8] + one[9]
# one_scores['agreeableness_score'] =  -one[0] + one[1] - one[2] + one[3] - one[4] - one[5] + one[6] - one[7] + one[8] + one[9]
# one_scores['conscientiousness_score'] = one[0] - one[1] + one[2] - one[3] + one[4] - one[5] + one[6] - one[7] + one[8] + one[9]
# one_scores['openness_score'] = one[0] - one[1] + one[2] - one[3] + one[4] - one[5] + one[6] + one[7] + one[8] + one[9]

#%% Generating scores for each 'personality trait', using simple addition/subtraction

#generated 10 abstract types (bins)
all_types = {'one': one, 'two': two, 'three': three, 'four': four, 'five': five, 'six': six, 'seven': seven, 'eight': eight,
             'nine': nine, 'ten': ten} 

all_types_scores = {}

#loop over all_types dict items defined above: which are 50 columns of scores each

for name, personality_type in all_types.items(): 
    personality_trait = {}

    personality_trait['extroversion_score'] =  personality_type[0] - personality_type[1] +personality_type[2] - personality_type[3] + personality_type[4] - personality_type[5] +personality_type[6] - personality_type[7] + personality_type[8] -personality_type[9]
    personality_trait['neuroticism_score'] =  personality_type[0] - personality_type[1] + personality_type[2] -personality_type[3] + personality_type[4] + personality_type[5] + personality_type[6] + personality_type[7] + personality_type[8] + personality_type[9]
    personality_trait['agreeableness_score'] =  -personality_type[0] +personality_type[1] - personality_type[2] + personality_type[3] - personality_type[4] - personality_type[5] + personality_type[6] - personality_type[7] + personality_type[8] + personality_type[9]
    personality_trait['conscientiousness_score'] = personality_type[0] - personality_type[1] + personality_type[2] -personality_type[3] +personality_type[4] - personality_type[5] +personality_type[6] -personality_type[7] + personality_type[8] + personality_type[9]
    personality_trait['openness_score'] =  personality_type[0] -personality_type[1] + personality_type[2] - personality_type[3] + personality_type[4] - personality_type[5] +personality_type[6] + personality_type[7] + personality_type[8] + personality_type[9]
    
    all_types_scores[name] = personality_trait
    
#each cluster has been assigned scores for each 5 personality traits

#%%    
# Each personality_type has 5 personality_trait values
# Now we need to normalize scores using scaling formula

all_extroversion = []
all_neuroticism =[]
all_agreeableness =[]
all_conscientiousness =[]
all_openness =[]

# for loop to make lists with a personality_trait score from each type

for personality_type, personality_trait in all_types_scores.items():
    all_extroversion.append(personality_trait['extroversion_score'])
    all_neuroticism.append(personality_trait['neuroticism_score'])
    all_agreeableness.append(personality_trait['agreeableness_score'])
    all_conscientiousness.append(personality_trait['conscientiousness_score'])
    all_openness.append(personality_trait['openness_score'])

# normalization equation for each personality trait
# total scores - min(score) / max(score) - min(score)

all_extroversion_normalized = (all_extroversion-min(all_extroversion))/(max(all_extroversion)-min(all_extroversion))
all_neuroticism_normalized = (all_neuroticism-min(all_neuroticism))/(max(all_neuroticism)-min(all_neuroticism))
all_agreeableness_normalized = (all_agreeableness-min(all_agreeableness))/(max(all_agreeableness)-min(all_agreeableness))
all_conscientiousness_normalized = (all_conscientiousness-min(all_conscientiousness))/(max(all_conscientiousness)-min(all_conscientiousness))
all_openness_normalized = (all_openness-min(all_openness))/(max(all_openness)-min(all_openness))

#%%

#counter variable allows the normalized personality trait scores to be filed off
#they will then end up in all_%trait_normmalized[]%

counter = 0

normalized_all_types_scores ={}

for personality_type, personality_trait in all_types_scores.items():
    normalized_personality_trait = {}
    
    normalized_personality_trait['extroversion_score'] = all_extroversion_normalized[counter]
    normalized_personality_trait['neuroticism_score'] = all_neuroticism_normalized[counter]
    normalized_personality_trait['agreeableness_score'] = all_agreeableness_normalized[counter]
    normalized_personality_trait['conscientiousness_score'] = all_conscientiousness_normalized[counter]
    normalized_personality_trait['openness_score'] = all_openness_normalized[counter]
    
    normalized_all_types_scores[personality_type] = normalized_personality_trait
    
    counter+=1
    
# normalized_personality_trait gets recycled on each iteration
# while normalized_all_types_score (dict) has a key added each loop, becomes a nested dictionary
# ---> i.e counter = 0, so new entry becomes: 'one' : ['EXT': normalized score, 'NRT': normalized scores]
    

#%% Plot normalized values


plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['one'].keys(),normalized_all_types_scores['one'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['two'].keys(),normalized_all_types_scores['two'].values())
plt.show()
    
plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['three'].keys(),normalized_all_types_scores['three'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['four'].keys(),normalized_all_types_scores['four'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['five'].keys(),normalized_all_types_scores['five'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['six'].keys(),normalized_all_types_scores['six'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['seven'].keys(),normalized_all_types_scores['seven'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['eight'].keys(),normalized_all_types_scores['eight'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['nine'].keys(),normalized_all_types_scores['nine'].values())
plt.show()

plt.figure(figsize= (12,5))
plt.bar(normalized_all_types_scores['ten'].keys(),normalized_all_types_scores['ten'].values())
plt.show()

    
    