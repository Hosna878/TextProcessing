"""This file included all the fucntions for text processing and matching
Developer: Hosna Hamdieh"""

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
# nltk.download("punkt")
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from text_clean import *
from bert_serving.client import BertClient
import pandas as pd
import re,io,os
import numpy as np
import ast
import pickle

# model_name_list = ["BERT-Base, Uncased"]

def BERT_train(corpus_sentences:list,model_name:str):
    corpus_sentences = [text_clean(i) for i in corpus_sentences if type(i)==str]
    model = SentenceTransformer(model_name)
    print("Encoding the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    print("Storing file on disc")
    with open(f"embeding_{model_name}.pkl", "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)

def BERT_text(text1,text2,model_name:str,split_with="\n",score_lim=0.5):
    model = SentenceTransformer(model_name)
    sentences1 = text1.split(split_with)
    sentences2 = text2.split(split_with)
    embedding1 = model.encode(sentences1)
    embedding2 = model.encode(sentences2)
    # print(sentence_embeddings.shape)
    # cosine_similarity([embedding1],embedding2)
    similarity = []
    for i in range(len(sentences1)):
        row = {}
        for j in range(len(sentences2)):
            row = ({"t1":sentences1[i],"t2":sentences2[j],"s":util.pytorch_cos_sim(embedding1[i], embedding2[j]).item()})
            if util.pytorch_cos_sim(embedding1[i], embedding2[j]).item() >= score_lim:
                similarity.append(row)
    print(similarity)
    similarity = pd.DataFrame(similarity)
    similarity = similarity.sort_values(by=['s'],ascending=False)
    sim = similarity.groupby('t1')[['t2','s']].agg(list).reset_index()
    # print(sim)
    return sim

# clnl = {"text_orginal":"","index_orginal":"","text_source":"","index_source":""}
def BERT_df(df1,df2,clnl:dict,model_name:str,num_int:int):
    model = SentenceTransformer(model_name)
    sentences1 = df1[clnl["text_orginal"]].tolist()
    sent1 = [text_clean(i) for i in sentences1 if type(i)==str]
    index_sent1 = df1[clnl["index_orginal"]].tolist()
    sentences2 = df2[clnl["text_source"]].tolist()
    index_sent2 = df2[clnl["index_source"]].tolist()
    embedding1 = model.encode(sent1)
    # if trained_model_name=="":
    sent2 = [text_clean(i) for i in sentences2 if type(i)==str]
    embedding2 = model.encode(sent2)
    # else:
    #     embedding2 = pickle.load(open(trained_model_name, 'rb'))
    # print(sentence_embeddings.shape)
    similarity = []
    for i in range(len(sentences1)):
        row = {}
        for j in range(len(sentences2)):
            row = ({"ID":index_sent1[i],"Text":sentences1[i],"Source_ID":index_sent2[j],"Source_Text":sentences2[j],"Matching_Score":util.pytorch_cos_sim(embedding1[i], embedding2[j]).item()})
            # if util.pytorch_cos_sim(embedding1[i], embedding2[j]).item() >= score_lim:
            similarity.append(row)
            # except Exception as e:
            #     print("Error",e)
    # print(similarity)
    similarity = pd.DataFrame(similarity)
    similarity = similarity.sort_values(by=['Matching_Score'],ascending=False)
    sim = similarity.groupby(['ID',"Text"])[["Source_ID",'Source_Text','Matching_Score']].agg(list).reset_index()
    sim['Matching_Score']=sim['Matching_Score'].map(lambda x: x[:num_int] if len(x)>num_int else x)
    sim["Source_ID"]=sim["Source_ID"].map(lambda x: x[:num_int] if len(x)>num_int else x)
    sim['Source_Text']=sim['Source_Text'].map(lambda x: x[:num_int] if len(x)>num_int else x)
    # print(sim)
    return sim

def get_recommendations(title, cosine_sim, indices,score_lim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores_list = [sim[1] for sim in sim_scores if sim[1] >= score_lim]
    # Get the indices
    indices_list = [i[0] for i in sim_scores if i[1] >= score_lim]
    # Return the top 10 most similar movies
    return {"t1":title,"t2":indices_list,"s":sim_scores_list}
    
def TFIDF(text1,text2,split_with="\n",score_lim=0.5):
    tfidf_vectorizer = TfidfVectorizer()
    sentences1 = text1.split(split_with)
    sentences2 = text2.split(split_with)
    # print("start")
    all_sen = sentences1+sentences2
    indexlist = [i for i in range(len(all_sen))]
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_sen)
    metadata = pd.DataFrame({"title":all_sen,"index":indexlist})
    # print("done")
    similarity = []
    indices = pd.Series(metadata.index, index=metadata['title'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)[:len(sentences1)][len(sentences1):]
    for i in range(len(sentences1)):
        row = get_recommendations(title=sentences1[i], cosine_sim=cosine_sim, indices=indices, score_lim=score_lim)
        newlist = [all_sen[i] for i in row["t2"]]
        row["t2"] = newlist
        similarity.append(row)
    print(similarity)
    similarity = pd.DataFrame(similarity)
    similarity = similarity.sort_values(by=['s'],ascending=False)
    sim = similarity.groupby('t1')[['t2','s']].agg(list).reset_index()
    print(sim)
    return sim

if __name__ == "main":   
    text1 = "I need to test this function now to see how things work\nwhat is this place and why it is dark"  
    text2 = "Testing the function is necessary\n I want to sleep\nthis is fun\nlets go out and see what is happening"  
    # BERT_text(text1,text2,model_name='bert-base-nli-mean-tokens',split_with="\n")
    # TFIDF(text1,text2,split_with="\n")