# Imports
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
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
option = sidebar.selectbox('Select one option:',('table data analysis','topic modeling','text parser','text matching','other'))
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
            st.write("Here is the raw dataset:")
            st.write(f"Data size: {len(df)}*{len(df.columns)}")
            # Display the dataframe
            st.write(df)
            cl_tup = tuple(sorted(df))
            com_tup = ("==",">=","<=","!=",">","<","in","not in")
            action = sidebar.selectbox('Select from options:',("filter based on vlaue of a column","clean the database","other"))
            if action == "filter based on vlaue of a column":
                column_selected = st.selectbox('Select the column:',cl_tup)
                print(column_selected,type(column_selected))
                comparison_selected = st.selectbox('Select the comparison method:',com_tup)
                print(comparison_selected,type(comparison_selected))
                if column_selected and comparison_selected:
                    if comparison_selected in ["==",">=","<=","!=",">","<"]:
                        value_wanted = st.selectbox("Select the value", pd.unique(df[column_selected]))
                        try:
                            vlaue_wanted = float(value_wanted)
                        except:
                            print("it is text")
                        if com_tup == "==":
                            df2 = df[df[column_selected] == value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                        elif com_tup == ">=":
                            df2 = df[df[column_selected] >= value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                        elif com_tup == "<=":
                            df2 = df[df[column_selected] <= value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                        elif com_tup == ">":
                            df2 = df[df[column_selected] > value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                        elif com_tup == "<":
                            df2 = df[df[column_selected] < value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                        elif com_tup == "!=":
                            df2 = df[df[column_selected] != value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                    elif comparison_selected in ["in","not in"]:
                        value_wanted = st.text_input("Type the list of values seperated by comma (,)")
                        value_wanted = value_wanted.split(",")
                        try:
                            value_wanted = [float(i) for i in value_wanted]
                        except:
                            print("it is text")
                        if com_tup == "in":
                            df2 = df[df[column_selected] in value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                        elif com_tup == "not in":
                            df2 = df[df[column_selected] not in value_wanted]
                            print(value_wanted,type(value_wanted))
                            st.write("Here is the filtered dataset:")
                            st.write(f"Data size: {len(df2)}*{len(df2.columns)}")
                            st.write(df2)
                    
                
        
    else:
        st.write("The file size surpasses the limit")
        
elif option == 'text matching':
    input_type = sidebar.selectbox('Select input type:',('text','text file','table file'))
    # st.write('You selected:', input_type)
    if input_type == "text file":
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
        text1 = texts[0].split("\n")
        text1 = [[i,text1[i]] for i in range(len(text1))]
        text1 = pd.DataFrame(text1, columns =['id', 'text'], dtype = float) 
        text2 = texts[1].split("\n")
        text2 = [[i,text2[i]] for i in range(len(text2))]
        text2 = pd.DataFrame(text2, columns =['id', 'text'], dtype = float) 
        clnl= {"text_orginal":"text","index_orginal":"id","text_source":"text","index_source":"id"}
    elif input_type == "table file":
        filenames = sidebar.file_uploader('Open file:', type=["xlsx","csv"], accept_multiple_files=True)
        texts=[]
        i = 0
        for filename in filenames[:2]:
            if i < 2:
                try:
                    df = pd.read_excel(filename,encoding="latin-1")
                except:
                    df = pd.read_csv(filename, encoding_errors="ignore", encoding="utf-8")
                cl_tup = tuple(sorted(df))
                fn = (filename.read()).decode()
                if i==0:
                    j = "the first file"
                else:
                    j = "the second file"
                column_selected_text = st.selectbox(f'Select the text column for {j}:',cl_tup)
                column_selected_id = st.selectbox(f'Select the id column for {j}:',cl_tup)
                df = df[[column_selected_text,column_selected_id]]
                df.rename(columns={column_selected_text: 'text', column_selected_id: "id"}, inplace=True)
                print(f"Text column: {column_selected_text},Id column: {column_selected_id}")
                texts.append(df)
                # print((filename.read()).decode())
                i+=1
            else:
                break
        
            
        text1 = texts[0]
        text2 = texts[1]
    else:
        text1 = sidebar.text_area("Type in the first text", height=100).split("\n")
        text1 = [[i,text1[i]] for i in range(len(text1))]
        text1 = pd.DataFrame(text1, columns =['id', 'text'], dtype = float) 
        text2 = sidebar.text_area("Type in the second text", height=100).split("\n")
        text2 = [[i,text2[i]] for i in range(len(text2))]
        text2 = pd.DataFrame(text2, columns =['id', 'text'], dtype = float) 
        
        
    display_t1 = sidebar.checkbox("Display text1", value=True)
    display_t2 = sidebar.checkbox("Display text2", value=True)
    
    if display_t1:
        st.write(f"text1:")
        st.write(text1)
    if display_t2:
        st.write(f"text2:")
        st.write(text2)
            
    if len(text1)!=0 and len(text2)!=0:
        if len(text1) <= 200 and len(text2) <= 200:
            method = sidebar.selectbox('Select matching method:',('BERT','TFIDF','other'))
            if method == "BERT":
                model = sidebar.selectbox('Select method to select a model:',('select from the list','type the model name','other'))
                if model == "select from the list":
                    m = sidebar.selectbox('Select model:',('sentence-transformers/bert-base-nli-mean-tokens','bert-base-nli-mean-tokens','paraphrase-multilingual-mpnet-base-v2','all-mpnet-base-v2','paraphrase-multilingual-MiniLM-L12-v2','multi-qa-mpnet-base-dot-v1','distiluse-base-multilingual-cased-v1','distiluse-base-multilingual-cased-v2'))
                elif model == "type the model name":
                    m = sidebar.text_input("Type the BERT model name")
                clnl= {"text_orginal":"text","index_orginal":"id","text_source":"text","index_source":"id"}
                sim_mat = BERT_df(df1=text1,df2=text2,clnl=clnl,model_name=m,num_int=5)
                st.write(sim_mat)
                csv_out = sim_mat.to_csv(index=False).encode('utf-8')
                st.download_button('Download results', csv_out, "Result.csv","text/csv",key='download-csv')
                
            elif method == "TFIDF":
                sim_mat = TFIDF(text1,text2,split_with=";")
                st.write(sim_mat)
                csv_out = sim_mat.to_csv(index=False).encode('utf-8')
                st.download_button('Download results', csv_out, "Result.csv","text/csv",key='download-csv')
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