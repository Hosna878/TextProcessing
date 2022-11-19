# Imports
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
from PIL import Image
img = Image.open("I4Data.png")
import os
from i4data_f import *
from text_processing import *
from io import StringIO 
from text_clean import *
# -----------------------------------------------------------
st.set_page_config(page_title="I4Data", page_icon=img)
# -----------------------------------------------------------
sidebar = st.sidebar
# Create a title for your app
sidebar.title("I4Data")
# Display the dataframe
option = sidebar.selectbox('Select one option:',('table data analysis','text parser','text matching','other'))
# st.write('You selected:', option)
if option =='table data analysis':
    filename = sidebar.file_uploader('Open file:', type=["xlsx","csv"])
    
    try:
        df = pd.read_excel(filename)
    except:
        df = pd.read_csv(filename, encoding_errors="ignore", encoding="utf-8")
        # print(df)
    df_display = sidebar.checkbox("Display Raw Data", value=True)
    if len(df) > 0:
        if df_display:

            st.write("Here is the dataset used in this analysis:")

            # Display the dataframe
            st.write(df)
    else:
        st.write("The file size surpasses the limit")
elif option == 'text matching':
    input_type = sidebar.selectbox('Select input type:',('text','file','other'))
    # st.write('You selected:', input_type)
    if input_type == "file":
        filenames = sidebar.file_uploader('Open file:', type=["txt","docx","pdf","doc"], accept_multiple_files=True)
        texts=[]
        i = 0
        for filename in filenames[:2]:
            if i < 2:
                texts.append((filename.read()).decode(encoding="latin-1"))
                print((filename.read()).decode())
                i+=1
            else:
                break
            # print(text)
            # if text:
            #     st.write(f"{filename.name}: {text}")
        text1 = texts[0]
        text2 = texts[1]
    else:
        text1 = sidebar.text_area("Type in the first text", height=100)
        text2 = sidebar.text_area("Type in the second text", height=100)
        
    display_t1 = sidebar.checkbox("Display text1", value=True)
    display_t2 = sidebar.checkbox("Display text2", value=True)
    
    if display_t1:
        st.write(f"text1: {text1}")
    if display_t2:
        st.write(f"text1: {text2}")
            
    if text1 and text2:
        if len(text1.split()) <= 200 and len(text2.split()) <= 200:
            method = sidebar.selectbox('Select matching method:',('BERT','TFIDF','other'))
            if method == "BERT":
                model = sidebar.selectbox('Select model:',('select from the list','type the model name','other'))
                if model == "select from the list":
                    m = sidebar.selectbox('Select model:',("bert-base-nli-mean-tokens", "bert-large-uncased","multi-qa-mpnet-base-dot-v1","bert-base-multilingual-uncased"))
                elif model == "type the model name":
                    m = sidebar.text_input("Type the BERT model name")
                sim_mat = BERT(text1=text1,text2=text2,model_name='bert-base-nli-mean-tokens',split_with=";",score_lim=0.5)
                st.write(sim_mat)
                
            elif method == "TFIDF":
                sim_mat = TFIDF(text1,text2,split_with=";")
                st.write(sim_mat)
        else:
            st.write("The size of the text is big")
                   

    # n_clusters = sidebar.slider(
    #     "Select Number of Clusters",
    #     min_value=2,
    #     max_value=10,
    # )
    # -----------------------------------------------------------
    # A description

    
# -----------------------------------------------------------