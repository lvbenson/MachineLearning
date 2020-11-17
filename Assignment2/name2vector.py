

import pandas as pd
import operator
import numpy as np

possible_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
'q','r','s','t','u','v','w','x','y','z']
poss_prefixes = [''.join([x,y]) for x in possible_letters for y in possible_letters]
#print(poss_prefixes)
poss_suffixes = [''.join([x,y]) for x in possible_letters for y in possible_letters]


def features(names):
    classify = pd.DataFrame({
        "name": names['name'],
        'gender': [gender for gender in names['gender']],
        'prefix': [name[:2]for name in names['name']], #take the first two letters of every name in our new dataset
        'suffix': [name[-2:] for name in names['name']],  #take last two letters
    
    })

    #print(classify)
    return(classify)

#create conditional probabilities of each prefix and suffix occurring (Part 1 of homework question 1)
def train(names):
    X = features(names)



    cond_pre_male = {prefix: len(X.loc[X['prefix'] == prefix].loc[X['gender']=='m'])/
    len(X.loc[X['gender']=='m']) for prefix in poss_prefixes}

    cond_suff_male = {suffix: len(X.loc[X['suffix']==suffix].loc[X['gender']=='m'])/
    len(X.loc[X['gender']=='m']) for suffix in poss_suffixes}

    cond_pre_female = {prefix: len(X.loc[X['prefix'] == prefix].loc[X['gender']=='f'])/
    len(X.loc[X['gender']=='f']) for prefix in poss_prefixes}

    cond_suff_female = {suffix: len(X.loc[X['suffix']==suffix].loc[X['gender']=='f'])/
    len(X.loc[X['gender']=='f']) for suffix in poss_suffixes}

    model = {"cond_pre_male":cond_pre_male,
    "cond_suff_male":cond_suff_male,
    "cond_pre_female":cond_pre_female,
    "cond_suff_female":cond_suff_female}


    for item in X['name']:
        pre = item[:2]
        suff = item[-2:]
        for k, v in model['cond_pre_male'].items():
            if pre == k:
                pre_prob_male = v
        for k, v in model['cond_suff_male'].items():
            if suff == k:
                suff_prob_male = v
        for k, v in model['cond_pre_female'].items():
            if pre == k:
                pre_prob_female = v
        for k, v in model['cond_suff_female'].items():
            if suff == k:
                suff_prob_female = v
        
        male_prob = pre_prob_male*suff_prob_male
        female_prob = pre_prob_female*suff_prob_female
        if female_prob == 0.0:
            classification = 'male'
        elif male_prob == 0.0:
            classification = 'female'
        else:
            probability = np.log(male_prob/female_prob)
            if probability > 0:
                classification = 'male'
            else:
                classification = 'female'
        #print(classification)

#%%
def test(train_names, test_names):

    classify_test = pd.DataFrame({
        "name": test_names['name'],
    })


    X = features(train_names)

    cond_pre_male = {prefix: len(X.loc[X['prefix'] == prefix].loc[X['gender']=='m'])/
    len(X.loc[X['gender']=='m']) for prefix in poss_prefixes}

    cond_suff_male = {suffix: len(X.loc[X['suffix']==suffix].loc[X['gender']=='m'])/
    len(X.loc[X['gender']=='m']) for suffix in poss_suffixes}

    cond_pre_female = {prefix: len(X.loc[X['prefix'] == prefix].loc[X['gender']=='f'])/
    len(X.loc[X['gender']=='f']) for prefix in poss_prefixes}

    cond_suff_female = {suffix: len(X.loc[X['suffix']==suffix].loc[X['gender']=='f'])/
    len(X.loc[X['gender']=='f']) for suffix in poss_suffixes}

    test_model = {"cond_pre_male":cond_pre_male,
    "cond_suff_male":cond_suff_male,
    "cond_pre_female":cond_pre_female,
    "cond_suff_female":cond_suff_female}

    classification_test = []
    for item in classify_test['name']:
        pre = item[:2]
        suff = item[-2:]
        for k, v in test_model['cond_pre_male'].items():
            if pre == k:
                pre_prob_male = v
        for k, v in test_model['cond_suff_male'].items():
            if suff == k:
                suff_prob_male = v
        for k, v in test_model['cond_pre_female'].items():
            if pre == k:
                pre_prob_female = v
        for k, v in test_model['cond_suff_female'].items():
            if suff == k:
                suff_prob_female = v
        
        
        male_prob = pre_prob_male*suff_prob_male
        female_prob = pre_prob_female*suff_prob_female
        if female_prob == 0.0:
            classification = -1 #male
            classification_test.append(classification)
        elif male_prob == 0.0:
            classification = 1 #female
            classification_test.append(classification)
        else:
            probability = np.log(male_prob/female_prob)
            #print(probability)
            if probability > 0.0:
                classification = -1 #male
                classification_test.append(classification)
            else:
                classification = 1 #female
                classification_test.append(classification)
    classify_test['classification'] = classification_test
    print(classify_test.head())
    classify_test.to_csv('classified_testing_data.csv')
        




    
        
