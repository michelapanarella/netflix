#have netflix_clean.csv already uploaded
#import pandas as pd; import numpy as np; import nltk

#import gensim;from nltk.tokenize import word_tokenize;nltk.download('punkt')

#from nltk.corpus import stopwords
#from gensim import models
#import re

#create empty folder called similarity on your desktop/working directory. 


def netflix_finder():
    print("Movie or TV series?")
    preference = input()
    
    if preference == "Movie":
        netflix_movie = netflix_clean.loc[(netflix_clean['type']=="Movie")]
        print("The ratings are: NR, G, PG, PG-13, 14A, MA, R. Which ratings would you like?")
        rating=input()
        rating_final = word_tokenize(re.sub(',', '', rating))
        netflix_filtered = netflix_movie[netflix_movie['rating'].isin(rating_final)]
        length_movie = "There are " + str(len(netflix_filtered['title'])) + " Movies that are " + rating + "-rated"
        print(length_movie)                                  

    else:
        netflix_tv = netflix_clean.loc[(netflix_clean['type']!="Movie")]
        print("The ratings are: NR, G, PG, PG-13, 14A, MA, R. Which ratings would you like?")
        rating=input()
        rating_final = word_tokenize(re.sub(',', '', rating))
        netflix_filtered = netflix_tv[netflix_tv['rating'].isin(rating_final)]
        length_tv = "There are " + str(len(netflix_filtered['title'])) + " TV shows that are " + rating + "-rated"
        print(length_tv)

    print("Describe what type of " + str(rating) + "-rated " + str(preference) + " you'd like to watch")

    interest=input()
    interest = word_tokenize(interest.lower())
    stop_words = stopwords.words('english') + ["want", "view", "see", "movie", "film", "series", "show", "TV"] 
    interest_final = [word for word in interest if word not in stop_words]

    #now filter the netflix_filtered()
    description = netflix_filtered['description'].apply(lambda x : [word for word in word_tokenize(x.lower()) if word not in stop_words])

    dictionary = gensim.corpora.Dictionary(description)
    corpus_descriptions = [dictionary.doc2bow(description) for description in description]
    tfidf = models.TfidfModel(corpus_descriptions, smartirs='ntc')
    #create an empty folder called similarity - gensim needs this to generate the similarity matrix
    sims = gensim.similarities.Similarity("/similarity", tfidf[corpus_descriptions],
                                        num_features=len(dictionary))


    
    file2_docs = []
    for line in interest_final:
        file2_docs.append(line)

    for line in file2_docs:
        query_doc_bow = dictionary.doc2bow(interest_final) #update an existing dictionary and create bag of words
        query_doc_tf_idf = tfidf[query_doc_bow]

    netflix_filtered['sims'] = sims[query_doc_tf_idf]
    recommend = netflix_filtered.sort_values(by=['sims'], ascending=False)
    print(recommend[['title', 'sims']].head(10))

    

