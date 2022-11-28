

"""
This script includes all the functions used in the feature extraction from job descriptions
location_finder, experience_finder, job_type_finder, abbreviation_finder, name_finder, job_title_finder_CNPNOC, domain_finder_CNPNOC (Additions to the parser API)
Developer: Hosna Hamdieh
Start date: 2022-03-24
Last Update: 2022-09-12
"""

# Libs:
import json
import csv
from datetime import datetime
from langdetect import detect
import re, os, io
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
# nltk.download("punkt")
import string, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import pandas as pd
import numpy as np
from azure.cosmosdb.table.tableservice import TableService
from azure.storage.blob import ContainerClient
from azure.storage.blob import BlobClient, BlobServiceClient
import pyodbc
import textwrap
import ast
from azure.data.tables import TableClient
from collections import Counter
import pickle
import time
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import docx2txt
import textract

############################################################# Inputs ##########################################
CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=jobdetails;AccountKey=bpoQA96H+xpW/I/VOFH3+4hV2AwmWsratSQAtlrWRTfkaHYCGGaKV52yVKhixuqCzUTIC4lyIZr8EKX+DCVRJA==;EndpointSuffix=core.windows.net'
driver = "{ODBC Driver 17 for SQL Server}"
# 17 is better than 13 as it is faster
# Specify server, database, username, password, and query
server = 'candidatemanagementv2-master.database.windows.net'
# databaseDev = 'candidatemanagementv2-master-airudidev'
database = 'candidatemanagementv2-master-emploicomp'
username = 'myadministrator'
password = 'ThisIsThe3rdRandomPassword!!'
query_in = """SELECT  [Id] 
      ,[Title]
      ,[Description]    
  FROM [dbo].[Jobs]"""
# filename = "Test.csv"

special_chars = {
    "b'\\t'": '\t',
    "b'\\r'": '\n',
    "b'\\x07'": '|',
    "b'\\xc4'": 'Ä',
    "b'\\xe4'": 'ä',
    "b'\\xdc'": 'Ü',
    "b'\\xfc'": 'ü',
    "b'\\xd6'": 'Ö',
    "b'\\xf6'": 'ö',
    "b'\\xdf'": 'ß',
    "b'\\xa7'": '§',
    "b'\\xb0'": '°',
    "b'\\x82'": '‚',
    "b'\\x84'": '„',
    "b'\\x91'": '‘',
    "b'\\x93'": '“',
    "b'\\x96'": '-',
    "b'\\xb4'": '´'
}


############################################################################ Functions used in external functions ################################################################

def set_table_service():
    """ Set the Azure Table Storage service """
    return TableService(connection_string=CONNECTION_STRING)


def get_dataframe_from_table_storage_table(table_service, filter_query):
    """ Create a dataframe from table storage data """
    return pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query))


def get_data_from_table_storage_table(table_service, filter_query, SOURCE_TABLE):
    """ Retrieve data from Table Storage """
    for record in table_service.query_entities(SOURCE_TABLE, filter=filter_query):
        yield record

def data_grabber_SQL(query_in,connection_dic_str):
    """This function needes the folowing data as inputs driver,server,database,username,password,query and fetches the data in a list of tuples, the outcome is a data frame"""
    # creat the full connection string
    connection_string = textwrap.dedent(connection_dic_str)
    # Creating a new PYODBC Connection Object
    cnxn: pyodbc.Connection = pyodbc.connect(connection_string)
    # It is risky to autocommit
    # cnxn.autocommit = True
    # Create a new Cursor Object from the connection string
    crsr: pyodbc.Cursor = cnxn.cursor()
    # Depending on the cursor and the driver it could be used
    # crsr.fast_executmany=True
    # Execute the select query against
    # results=crsr.execute(select_sql)
    # results = crsr.execute(query)
    # # Grab the data
    # output=crsr.fetchall()
    # print(type(output),output)
    records = crsr.execute(query_in).fetchall()
    # Define our Column Names and we need only the first outcome of description as it has more than just the name
    columns = [column[0] for column in crsr.description]
    # Dump the records into a datafram
    ddf = pd.DataFrame.from_records(data=records, columns=columns)
    # print(ddf.head(),type(ddf))
    # Close the connection once we are done
    cnxn.close()
    return ddf

def data_grabber(driver, server, database, username, password, query_in):
    """This function needes the folowing data as inputs driver,server,database,username,password,query and fetches the data in a list of tuples, the outcome is a data frame"""
    # creat the full connection string
    connection_string = textwrap.dedent(f"""
                                    Driver={driver};
                                    Server={server};
                                    Database={database};
                                    Uid={username};
                                    Pwd={password};
                                    Encrypt=yes;
                                    TrustServerCertificate=no;
                                    Connection Timeout=30;
                                    """)
    # Creating a new PYODBC Connection Object
    cnxn: pyodbc.Connection = pyodbc.connect(connection_string)
    # It is risky to autocommit
    # cnxn.autocommit = True
    # Create a new Cursor Object from the connection string
    crsr: pyodbc.Cursor = cnxn.cursor()
    # Depending on the cursor and the driver it could be used
    # crsr.fast_executmany=True
    # Execute the select query against
    # results=crsr.execute(select_sql)
    # results = crsr.execute(query)
    # # Grab the data
    # output=crsr.fetchall()
    # print(type(output),output)
    records = crsr.execute(query_in).fetchall()
    # Define our Column Names and we need only the first outcome of description as it has more than just the name
    columns = [column[0] for column in crsr.description]
    # Dump the records into a datafram
    ddf = pd.DataFrame.from_records(data=records, columns=columns)
    # print(ddf.head(),type(ddf))
    # Close the connection once we are done
    cnxn.close()
    return ddf


def convert_doc_to_txt(path):
    return docx2txt.process(path)
    # return textract.process(path)


def convert_doc_to_txt_o(path):
    string = ''
    with open(path, 'rb') as stream:
        stream.seek(2560)  # Offset - text starts after byte 2560
        current_stream = stream.read(1)
        while not (str(current_stream) == "b'\\xfa'"):
            if str(current_stream) in special_chars.keys():
                string += special_chars[str(current_stream)]
            else:
                try:
                    char = current_stream.decode('UTF-8')
                    if char.isalnum():
                        string += char
                except UnicodeDecodeError:
                    string += ''
            current_stream = stream.read(1)
    return string


def convert_pdf_to_txt(path):
    '''Convert pdf content from a file path to text
    :path the file path
    '''
    rsrcmgr = PDFResourceManager()
    codec = "utf-8-sig"
    laparams = LAParams()

    with io.StringIO() as retstr:
        with TextConverter(rsrcmgr, retstr, codec=codec,
                           laparams=laparams) as device:
            with open(path, 'rb') as fp:
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                password = ""
                maxpages = 0
                caching = True
                pagenos = set()

                for page in PDFPage.get_pages(fp,
                                              pagenos,
                                              maxpages=maxpages,
                                              password=password,
                                              caching=caching,
                                              check_extractable=True):
                    interpreter.process_page(page)

                return retstr.getvalue()


def file_reader(file_name):
    file_type = file_name.split(".")[-1]
    text = ""
    try:
        if file_name.endswith(".docx"):
            text = convert_doc_to_txt(file_name)
        elif file_name.endswith(".doc"):
            text = textract.process(file_name).decode()
        elif file_name.endswith(".pdf"):
            text = convert_pdf_to_txt(file_name)
        elif file_name.endswith(".txt"):
            text = open(file_name, "r").read()
            file_name.close()
        else:
            text = ""
    except Exception as e:
        print("Error in file_reader",e)
        os.remove(file_name)
    return text, file_type


def parseval(text: str) -> str:
    "extracts real value from a string"
    try:
        return ast.literal_eval(text)
    except ValueError:
        return text


def blacklist(text_file_name: str) -> list:
    "Returns a list of blacklist from a text file"
    textfile = open(text_file_name, "r")
    bl = textfile.read()
    blacklist = bl.split("\n")
    blacklist = np.unique(blacklist).tolist()
    textfile.close()
    # print(blacklist)
    return blacklist


def remove_urls(text: str) -> str:
    "This function will erase the punctuations from a text"
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%\@\#\;\!\$\,\:)*\b', '', text, flags=re.MULTILINE)
    text = re.sub("[^a-zA-Z\s]", " ", text)
    return text


# def remove_accents_lib(text:str) -> str:
#     nkfd_form = unicodedata.normalize('NFKD', unicode(text))
#     return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def remove_accents(text: str) -> str:
    """
    Removes common accent characters, lower form.
    Uses: regex.
    """
    new = text.lower()
    new = re.sub(r"[àáâãäå]", 'a', new)
    new = re.sub(r"l'", 'l ', new)
    # new = re.sub(r"a'", 'a ', new)
    new = re.sub(r'[èéêë]', 'e', new)
    new = re.sub(r'[ìíîï]', 'i', new)
    new = re.sub(r'[òóôõö]', 'o', new)
    new = re.sub(r'[ùúûü]', 'u', new)
    # new = unidecode.unidecode(new)
    return new

def sub_str_between_x_and_Y(text: str, x: str, y: str) -> str:
    "This function returns the sub string between two characters"
    result = re.search(f'{x}(.*){y}', text)
    return result.group(1)

def text_clean(text: str) -> str:
    "Cleans the inputted text"
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub("[^a-zA-ZÀ-ú& ]+", "", text)
    text = remove_accents(text)
    text = text.strip()
    return text

def clean_text_stopwords(text: str) -> str:
    "Cleans the inputted text"
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub("[^a-zA-ZÀ-ú& ]+", "", text)
    text = remove_accents(text)
    stopwords_list_nltk = list(stopwords.words('english')) + list(stopwords.words('french'))
    stopwords_list_spacy = list(fr_stop) + list(en_stop)
    BlackList = (stopwords_list_nltk + stopwords_list_spacy)
    c = [word for word in text.split() if word not in BlackList]
    text = " ".join(c)
    text = remove_accents(text)
    return text

def stem_tokens(tokens: list) -> list:
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(item) for item in tokens]

def normalize(text: str) -> list:
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

def tfidf_on_list(language: str, tfidf_name: str):
    "This function creates a tfidf on a list of strings and saves it in the directory under the name given"
    pk = "CNP_NOC"
    SourceName = "CNPNOC"
    table_service, filter_query = set_table_service(), f"PartitionKey eq '{pk}'"
    df = pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query, SOURCE_TABLE=SourceName))
    # print(lan)
    alljobs_col_en = "NOCAllExamples"
    alljobs_col_fr = "CNPAllExamples"
    if language == "fr":
        job_title_list = df[alljobs_col_fr].tolist()
    elif language == "en":
        job_title_list = df[alljobs_col_en].tolist()
    else:
        job_title_list = df[alljobs_col_fr].tolist() + df[alljobs_col_en].tolist()
    jtl = []
    for s in job_title_list:
        jtl += s.split("|")
    job_title_list = [" ".join(normalize(job)).strip(" ") for job in jtl]
    cleaned = [clean_text_stopwords(document) for document in job_title_list]
    corpus = np.array(cleaned)
    vectorizer = CountVectorizer(decode_error="replace")
    vec_train = vectorizer.fit_transform(corpus)
    pickle.dump(vectorizer.vocabulary_, open(f'{tfidf_name}.pk', 'wb'))

def simcheck_upload_tfidf(input_list:list, language:str, score_limit:float):
    "This function uses a tfidf that is saved to transform it based on the new data"
    pk = "CNP_NOC"
    SourceName = "CNPNOC"
    table_service, filter_query = set_table_service(), f"PartitionKey eq '{pk}'"
    df = pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query, SOURCE_TABLE=SourceName))
    # print(lan)
    alljobs_col_en = "NOCAllExamples"
    alljobs_col_fr = "CNPAllExamples"
    if language == "fr":
        tfidfname = "tfidf_fr"
        job_title_list = df[alljobs_col_fr].tolist()
    else:
        tfidfname = "tfidf_en"
        job_title_list = df[alljobs_col_en].tolist()
    jtl = []
    for s in job_title_list:
        jtl += s.split("|")
    # job_title_list = [clean_text_stopwords(job).strip(" ") for job in jtl]
    input_list = [clean_text_stopwords(document) for document in input_list]
    input_list = [item for item in input_list if item != ""]
    # print(input_list)
    dt = input_list + job_title_list
    # print(dt)
    corpus = np.array(dt)
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(f"{tfidfname}.pk", "rb")))
    tfidf_m = transformer.fit_transform(loaded_vec.fit_transform(corpus))
    pairwise_sim = (tfidf_m * tfidf_m.T).A
    df = pd.DataFrame(pairwise_sim)
    df.columns = dt
    df.index = dt
    continents = input_list
    ndf = df[df.index.isin(continents)]
    ndf = ndf[job_title_list]
    cnpjt = ndf.idxmax(axis=1)
    df2 = pd.DataFrame(cnpjt)
    df2.columns = ["MList"]
    score = []
    for i, row in df2.iterrows():
        # score += [df.loc[i, row["MList"]]
        s = str(df.loc[i, row["MList"]])
        s = re.findall(r'-?\d+\.?\d*', s)[0]
        # print(s)
        score += [float(s)]
    df2["MScore"] = score
    df2 = df2.reset_index()
    df2.columns = ["List1", "MList", "MScore"]
    df2 = df2.drop_duplicates(subset=['List1'])
    # df2.reset_index(drop=True,inplace=True)
    # print(df2.info())
    df2 = df2[df2["MScore"] >= score_limit]
    # print(df2.info())
    out_list = df2["List1"].tolist()
    matched_list = df2["MList"].tolist()
    match_score = df2["MScore"].tolist()
    match_location = df2.index
    # print(df2)
    return out_list, matched_list, match_score, match_location

def title_sorting(unit_id:list,matched_job:list, matched_sentence:list, BroadCategory:list, MajorGroup:list, SubMajorGroup:list,MinorGroup:list, UnitGroup:list, alter_jobs:list,title_limit:int):
    """This function aims to sort the titles found by the title finder based on the domain repetitions and return all the inputted values in the new order"""
    if matched_job!=[]:
        matched = {}
        matched["Unit_ID"] = unit_id
        matched["CNPTitle"] = matched_job
        matched["BroadCategory"] = BroadCategory
        matched["MajorGroup"] = MajorGroup
        matched["SubMajorGroup"] = SubMajorGroup
        matched["MinorGroup"] = MinorGroup
        matched["UnitGroup"] = UnitGroup
        matched["AlternativeTitle"] = alter_jobs
        matched["Sentence"] = matched_sentence
        matched = pd.DataFrame(matched)
        Major_count = sorted(dict(Counter(matched["MajorGroup"].tolist())).items(), key=lambda kv: kv[1], reverse=True)
        main_Major = Major_count[0][0]
        main_Major_count = Major_count[0][1]
        Minor_count = sorted(dict(Counter(matched["MinorGroup"].tolist())).items(), key=lambda kv: kv[1], reverse=True)
        main_Minor = Minor_count[0][0]
        main_Minor_count = Minor_count[0][1]
        Unit_count = sorted(dict(Counter(matched["UnitGroup"].tolist())).items(), key=lambda kv: kv[1], reverse=True)
        main_Unit = Unit_count[0][0]
        main_Unit_count = Unit_count[0][1]
        # print(f"MajorGroup: {Major_count}\nMinorGroups:{Minor_count}\nUnitGroup:{Unit_count}")
        if main_Unit_count > 1:
            top_df = matched[matched["UnitGroup"] == main_Unit]
            top_df["Confidence"] = ["Great"] * (len(top_df))
            mid_df = matched[matched["MinorGroup"] == main_Minor]
            mid_df["Confidence"] = ["Good"] * (len(mid_df))
            bot_df = matched[matched["MajorGroup"] == main_Major]
            bot_df["Confidence"] = ["Sufficient"] * (len(bot_df))
            frames = [top_df, mid_df, bot_df]
            result = pd.concat(frames)
            # result = result.drop_duplicates(subset=['CNPTitle'], keep_data=['Sentence'])
        elif main_Minor_count > 1:
            top_df = matched[matched["MinorGroup"] == main_Minor]
            top_df["Confidence"] = ["Good"] * (len(top_df))
            mid_df = matched[matched["MajorGroup"] == main_Major]
            mid_df["Confidence"] = ["Sufficient"] * (len(mid_df))
            frames = [top_df, mid_df]
            result = pd.concat(frames)
            # result = result.drop_duplicates(subset=['CNPTitle'], keep_data=['Sentence'])
        elif main_Major_count > 1:
            top_df = matched[matched["MajorGroup"] == main_Major]
            top_df["Confidence"] = ["Sufficient"] * (len(top_df))
            result = top_df
            # result = result.drop_duplicates(subset=['CNPTitle'], keep_data=['Sentence'])
        else:
            result = matched
            result["Confidence"] = ["Low"] * (len(matched))
            # result = result.drop_duplicates(subset=['CNPTitle'], keep_data=['Sentence'])
        # print(len(result))
        r = result.groupby("CNPTitle",as_index=False)
        result = r.agg("first")
        result.update(r.agg({"Sentence": ' | '.join}))
        result = result.head(title_limit)
        # Define the sorter
        sorter = ["Great","Good","Sufficient","Low"]
        # Sorting the result based on confidence of matching
        result.sort_values(by="Confidence", key=lambda column: column.map(lambda e: sorter.index(e)), inplace=True)
        # print(result)
        matched_job = result["CNPTitle"].tolist()
        BroadCategory = result["BroadCategory"].tolist()
        MajorGroup = result["MajorGroup"].tolist()
        SubMajorGroup = result["SubMajorGroup"].tolist()
        MinorGroup = result["MinorGroup"].tolist()
        UnitGroup = result["UnitGroup"].tolist()
        confidence = result["Confidence"].tolist()
        alter_jobs = result["AlternativeTitle"].tolist()
        sentence = result["Sentence"].tolist()
        id = result["Unit_ID"].tolist()
        return id, matched_job, BroadCategory, MajorGroup, SubMajorGroup, MinorGroup, UnitGroup, alter_jobs, confidence, sentence
    else:
        return [],[],[],[],[],[],[],[],[]


########################################################################################### Parser API External functions ########################################################
# Location finder
def location_finder(Text: str) -> str:
    "This function finds the location (city and province in the text if the location is in Canada. This is an addition to the API Parser"
    provinces = {
        'Alberta': ['AB', 'Alberta'],
        'British Columbia': ['BC', 'British Columbia'],
        'Manitoba': ['MB', 'Manitoba'],
        'New Brunswick': ['NB', 'New Brunswick'],
        'Newfoundland and Labrador': ['NL', 'Newfoundland and Labrador'],
        'Northwest Territories': ['NT', 'Northwest Territories'],
        'Nova Scotia': ['NS', 'Nova Scotia'],
        'Nunavut': ['NU', 'Nunavut'],
        'Ontario': ['ON', 'Ontario'],
        'Prince Edward Island': ['PE', 'Prince Edward Island'],
        'Quebec': ['QC', 'Quebec'],
        'Saskatchewan': ['SK', 'Saskatchewan'],
        'Yukon': ['YT', 'Yukon']
    }
    Cities = {
        'Alberta': ['Airdrie', 'Beaumont', 'Brooks', 'Calgary', 'Camrose', 'Chestermere', 'Cold Lake', 'Edmonton',
                    'Fort Saskatchewan', 'Grande Prairie', 'Lacombe', 'Leduc', 'Lethbridge', 'Lloydminster',
                    'Medicine Hat', 'Red Deer', 'Spruce Grove', 'St. Albert', 'Wetaskiwin'],
        'British Columbia': ['Abbotsford', 'Armstrong', 'Burnaby', 'Campbell River', 'Castlegar', 'Chilliwack',
                             'Colwood', 'Coquitlam', 'Courtenay', 'Cranbrook', 'Dawson Creek', 'Delta', 'Duncan',
                             'Enderby', 'Fernie', 'Fort St. John', 'Grand Forks', 'Greenwood', 'Kamloops', 'Kelowna',
                             'Kimberley', 'Langford', 'Langley', 'Maple Ridge', 'Merritt', 'Mission', 'Nanaimo',
                             'Nelson', 'New Westminster', 'North Vancouver', 'Parksville', 'Penticton', 'Pitt Meadows',
                             'Port Alberni', 'Port Coquitlam', 'Port Moody', 'Powell River', 'Prince George',
                             'Prince Rupert', 'Quesnel', 'Revelstoke', 'Richmond', 'Rossland', 'Salmon Arm', 'Surrey',
                             'Terrace', 'Trail', 'Vancouver', 'Vernon', 'Victoria', 'West Kelowna', 'White Rock',
                             'Williams Lake'],
        'Manitoba': ['Brandon', 'Dauphin', 'Flin Flon', 'Morden', 'Portage la Prairie', 'Selkirk', 'Steinbach',
                     'Thompson', 'Winkler', 'Winnipeg'],
        'New Brunswick': ['Bathurst', 'Campbellton', 'Dieppe', 'Edmundston', 'Fredericton', 'Miramichi', 'Moncton',
                          'Saint John'],
        'Newfoundland and Labrador': ['Corner Brook', 'Mount Pearl', "St. John's"],
        'Northwest Territories': ['Yellowknife'],
        'Nova Scotia': ['Halifax', 'Sydney', 'Dartmouth'],
        'Nunavut': ['Iqaluit'],
        'Ontario': ['Barrie', 'Belleville', 'Brampton', 'Brant', 'Brantford', 'Brockville', 'Burlington', 'Cambridge',
                    'Clarence-Rockland', 'Cornwall', 'Dryden', 'Elliot Lake', 'Greater Sudbury', 'Guelph',
                    'Haldimand County', 'Hamilton', 'Kawartha Lakes', 'Kenora', 'Kingston', 'Kitchener', 'London',
                    'Markham', 'Mississauga', 'Niagara Falls', 'Norfolk County', 'North Bay', 'Orillia', 'Oshawa',
                    'Ottawa', 'Owen Sound', 'Pembroke', 'Peterborough', 'Pickering', 'Port Colborne',
                    'Prince Edward County', 'Quinte West', 'Richmond Hill', 'Sarnia', 'Sault Ste. Marie',
                    'St. Catharines', 'St. Thomas', 'Stratford', 'Temiskaming Shores', 'Thorold', 'Thunder Bay',
                    'Timmins', 'Toronto', 'Vaughan', 'Waterloo', 'Welland', 'Windsor', 'Woodstock'],
        'Prince Edward Island': ['Charlottetown', 'Summerside'],
        'Quebec': ['Acton Vale', 'Alma', 'Amos', 'Amqui', 'Baie-Comeau', "Baie-D'Urfé", 'Baie-Saint-Paul', 'Barkmere',
                   'Beaconsfield', 'Beauceville', 'Beauharnois', 'Beaupré', 'Bécancour', 'Bedford', 'Belleterre',
                   'Beloeil', 'Berthierville', 'Blainville', 'Boisbriand', 'Bois-des-Filion', 'Bonaventure',
                   'Boucherville', 'Lac-Brome', 'Bromont', 'Brossard', 'Brownsburg-Chatham', 'Candiac', 'Cap-Chat',
                   'Cap-Santé', 'Carignan', 'Carleton-sur-Mer', 'Causapscal', 'Chambly', 'Chandler', 'Chapais',
                   'Charlemagne', 'Châteauguay', 'Château-Richer', 'Chibougamau', 'Clermont', 'Coaticook',
                   'Contrecoeur', 'Cookshire-Eaton', 'Côte Saint-Luc', 'Coteau-du-Lac', 'Cowansville', 'Danville',
                   'Daveluyville', 'Dégelis', 'Delson', 'Desbiens', 'Deux-Montagnes', 'Disraeli', 'Dolbeau-Mistassini',
                   'Dollard-des-Ormeaux', 'Donnacona', 'Dorval', 'Drummondville', 'Dunham', 'Duparquet', 'East Angus',
                   'Estérel', 'Farnham', 'Fermont', 'Forestville', 'Fossambault-sur-le-Lac', 'Gaspé', 'Gatineau',
                   'Gracefield', 'Granby', 'Grande-Rivière', 'Hampstead', 'Hudson', 'Huntingdon', 'Joliette',
                   'Kingsey Falls', 'Kirkland', 'La Malbaie', 'La Pocatière', 'La Prairie', 'La Sarre', 'La Tuque',
                   'Lac-Delage', 'Lachute', 'Lac-Mégantic', 'Lac-Saint-Joseph', 'Lac-Sergent', "L'Ancienne-Lorette",
                   "L'Assomption", 'Laval', 'Lavaltrie', 'Lebel-sur-Quévillon', "L'Épiphanie", "Léry", "Lévis",
                   "L'Île-Cadieux", "L'Île-Dorval", "L'Île-Perrot", "Longueuil", "Lorraine", "Louiseville", "Macamic",
                   "Magog", "Malartic", "Maniwaki", "Marieville", "Mascouche", "Matagami", "Matane", "Mercier",
                   "Métabetchouan–Lac-à-la-Croix", "Métis-sur-Mer", "Mirabel", "Mont-Joli", "Mont-Laurier", 'Montmagny',
                   "Montreal", "Montreal West", 'Montréal-Est', 'Mont-Saint-Hilaire', "Mont-Tremblant", "Mount Royal",
                   'Murdochville', 'Neuville', 'New Richmond', 'Nicolet', 'Normandin', "Notre-Dame-de-l'Île-Perrot",
                   'Notre-Dame-des-Prairies', "Otterburn Park", 'Paspébiac', "Percé", 'Pincourt', 'Plessisville',
                   'Pohénégamook', 'Pointe-Claire', 'Pont-Rouge', 'Port-Cartier', 'Portneuf', 'Prévost', 'Princeville',
                   'Québec', 'Repentigny', 'Richelieu', 'Richmond', 'Rigaud', 'Rimouski', 'Rivière-du-Loup',
                   'Rivière-Rouge', 'Roberval', 'Rosemère', 'Rouyn-Noranda', 'Saguenay', "Saint-Amable",
                   'Saint-Augustin-de-Desmaures', 'Saint-Basile', 'Saint-Basile-le-Grand', 'Saint-Bruno-de-Montarville',
                   'Saint-Césaire', 'Saint-Charles-Borromée', "Saint-Colomban", 'Saint-Constant', 'Sainte-Adèle',
                   'Sainte-Agathe-des-Monts', 'Sainte-Anne-de-Beaupré', 'Sainte-Anne-de-Bellevue',
                   'Sainte-Anne-des-Monts', 'Sainte-Anne-des-Plaines', 'Sainte-Catherine',
                   'Sainte-Catherine-de-la-Jacques-Cartier', 'Sainte-Julie', 'Sainte-Marguerite-du-Lac-Masson',
                   'Sainte-Marie', 'Sainte-Marthe-sur-le-Lac', 'Sainte-Thérèse', 'Saint-Eustache', 'Saint-Félicien',
                   'Saint-Gabriel', 'Saint-Georges', 'Saint-Hyacinthe', 'Saint-Jean-sur-Richelieu', 'Saint-Jérôme',
                   'Saint-Joseph-de-Beauce', 'Saint-Joseph-de-Sorel', "Saint-Lambert", "Saint-Lazare",
                   "Saint-Lin–Laurentides", "Saint-Marc-des-Carrières", "Saint-Ours", "Saint-Pamphile", "Saint-Pascal",
                   "Saint-Philippe", "Saint-Pie", "Saint-Raymond", "Saint-Rémi", 'Saint-Sauveur', 'Saint-Tite',
                   'Salaberry-de-Valleyfield', 'Schefferville', 'Scotstown', 'Senneterre', 'Sept-Îles', 'Shannon',
                   'Shawinigan', 'Sherbrooke', "Sorel-Tracy", "Stanstead", 'Sutton', "Témiscaming",
                   'Témiscouata-sur-le-Lac', 'Terrebonne', 'Thetford Mines', 'Thurso', 'Trois-Pistoles',
                   'Trois-Rivières', 'Valcourt', "Val-d'Or", 'Val-des-Sources', 'Varennes', 'Vaudreuil-Dorion',
                   'Victoriaville', 'Ville-Marie', 'Warwick', 'Waterloo', 'Waterville', 'Westmount', 'Windsor'],
        'Saskatchewan': ['Estevan', 'Flin Flon', 'Humboldt', 'Lloydminster', 'Martensville', 'Meadow Lake', 'Melfort',
                         'Melville', 'Moose Jaw', 'North Battleford', 'Prince Albert', 'Regina', 'Saskatoon',
                         'Swift Current', 'Warman', 'Weyburn', 'Yorkton'],
        'Yukon': ['Whitehorse']
    }

    city = " "
    province = " "
    try:
        for i in provinces:
            WL = provinces[i]
            # print(WL)
            if WL[0] in Text or WL[1] in Text:
                province = WL[0]
                for c in Cities[i]:
                    if c in Text:
                        city = c
        LocStr = f"{city},{province},Canada".replace(" ,", "")
        return LocStr
    except Exception as e:
        print("Error in location_finder",e)
        return ""

def experience_finder(Text: str) -> str:
    "This function will extract experience from a text. This is an addition to the API Parser"
    List = ["years'", "years of experience", "années d'expérience", "years of", "ans d’expérience", "years", "year",
            "d'expérience", "an", "années", "années'"]
    # e=[]
    for c in List:
        try:
            e = re.findall(f'([^ \r\n]+) {c}?([\r\n]| |$)', Text, re.IGNORECASE)
        except Exception as e:
            print("Error in experience_finder",e)
            e = []
        if e != []:
            # if ''.join(map(lambda x: str(x[0]), e)).isalpha():
            y = re.sub("[^0-9]", "", ''.join(map(lambda x: str(x[0]), e)))
            if any(c.isalpha() for c in ''.join(map(lambda x: str(x[0]), e))) or len(y) > 4:
                return ""
            else:
                text = ' '.join(map(lambda x: str(x[0]), e))
                exlist = [str(int(s)) for s in re.findall(r'\b\d+\b', text)]
                return " ".join(exlist)
    return ""


def job_type_finder(Text: str) -> str:
    "This function will return the job if it finds indicators in the inputted text. This is an addition to the API Parser"
    FT = ["Full-time", "Full Time", "Permanent contract", "Permanent", "à plein temps", "contrat à durée indéterminée",
          "permanente", "permanent", "full_time"]
    PT = ["Contract", "Apprenticeship", "Part-time", "Part time", "Casual", "Seasonal", "Paid internship",
          "Student employment", "Telecommuting", "Unpaid internship", "Internship", "Temporary", "à temps partiel",
          "Contrat", "Apprentissage", "décontractée", "Saisonnière", "Saisonnier", "Stage rémunéré", "Emploi étudiant",
          "Télétravail", "Stage non rémunéré", "Stage", "Temporaire"]
    JT = ""
    try:
        for i in FT:
            if i.lower() in Text.lower():
                JT = "Full_time"
                return "Full_time"
                # break
        for j in PT:
            if j.lower() in Text.lower():
                JT = "Part_time"
                return "Part_time"
                # break
        if JT == "":
            return ""
    except Exception as e:
        print("Error in job_type_finder",e)
        return ""

def abbreviation_finder(Text: str) -> list:
    "This function will find all the abbreviations. This is an addition to the API Parser"
    # ab = re.findall("([A-Z]\.*){2,}s?", text)
    try:
        ab = re.findall(r'[A-Z]{2,}', Text)
        ab = [i for i in ab if i != "I"]
        ab = list(set(ab))
        return ab
    except Exception as e:
        print("Error in abbreviation_finder",e)
        return []


# print(ab_finder(". I have a UPS in the UNI to save all the docs CHIH"))

def name_finder(Text: str) -> list:
    "This function will return names (words starting with capital letter) from the inputted text. This is an addition to the API Parser"
    try:
        List = ["I", "You", "She", "He", "We", "It", "They"]
        ss = re.findall(r". [A-Z]{1,}[a-z]+", Text)
        nn = [s.replace(". ", "") for s in ss]
        # print(nn)
        names = re.findall(r"[A-Z]{1,}[a-z]+", Text)
        # print(names)
        names = [i.strip() for i in names if i.strip() not in List + nn]
        names = list(set(names))
        return names
    except Exception as e:
        print("Error in name_finder",e)
        return []


def title_domain_finder_CNPNOC(file_text:str, cl_en:list, cl_fr:list, SourceName:str, score_limit:float):
    """This function will give back a dataframe having the CNP/NOC matched, titles, profiles, minor groups and major groups. This is an addition to the API Parser
    inputs are the list of job titles we need to match and the source name and the language"""
    file_text = file_text.replace("Sr.", "Sr")
    file_text = file_text.replace("Jr.", "Jr")
    text_list = file_text.split("\n")
    # text_list = [remove_accents(text) for text in text_list][:list_len]
    text_list = [remove_accents(text) for text in text_list]
    pk = "CNP_NOC"
    table_service, filter_query = set_table_service(), f"PartitionKey eq '{pk}'"
    df = pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query, SOURCE_TABLE=SourceName))
    lan = detect(file_text)
    # print(lan)
    if lan == "fr":
        df = df[cl_fr]
    else:
        df = df[cl_en]
    if len(cl_en) == 5 or len(cl_fr) == 5:
        df.columns = ["id","Jobs", "Unit", "Minor", "Major"]
    else:
        df.columns = ["id","Jobs", "Unit", "Minor", "Major", "SubMajor", "Broad"]
    SJTL = df.Jobs.tolist()
    # All the titles
    tjl = []
    for s in SJTL:
        tjl += s.split("|")
    out_list, matched_list, match_score, match_loc = simcheck_upload_tfidf(text_list, lan, score_limit)
    # print(max(match_score),min(match_score))
    matched_list = [i.strip(" ") for i in matched_list]
    id, Unit, Broad, Major, SubMajor, Minor, alter_jobs, matched_job, score, match_location, sentences = [],[], [], [], [], [], [], [], [], [], []

    # revisions are needed to make it faster
    for i in range(len(matched_list)):
        mj = matched_list[i]
        try:
            for index, row in df.iterrows():
                if mj in row.Jobs:
                    matched_job += [mj]
                    score += [match_score[i]]
                    match_location += [match_loc[i]]
                    sentences += [out_list[i]]
                    Major += [row["Major"]]
                    Minor += [row["Minor"]]
                    Unit += [row["Unit"]]
                    id += [row["id"]]
                    alter_jobs += [row["Jobs"].split("|")]
                    if len(cl_en) != 4:
                        SubMajor += [row["SubMajor"]]
                        Broad += [row["Broad"]]
                    else:
                        SubMajor += ["NA"]
                        Broad += ["NA"]
                    # print(f"Profile matched for '{mj}':", row.Domain)
                    break
        except Exception as e:
            print("Error in title_domain_finder_CNPNOC",e)
            print(f"Removing {mj}")

    Major = [i.strip(" ") for i in Major]
    Unit = [i.strip(" ") for i in Unit]
    Minor = [i.strip(" ") for i in Minor]
    alter_jobs = [[i.strip(" ") for i in j] for j in alter_jobs]
    if "NA" not in SubMajor:
        SubMajor = [i.strip(" ") for i in SubMajor]
    if "NA" not in Broad:
        Broad = [i.strip(" ") for i in Broad]
    # print(len(matched_job), len(Major), len(Unit), len(Minor), len(alter_jobs), len(SubMajor), len(Broad), len(sentences), len(score), len(match_location))
    return id, matched_job, Major, Unit, Minor, alter_jobs, SubMajor, Broad, sentences, score, match_location

def seniority_level_finder(Job_Title:str, Experience, E1=60, E2=84, E3=24):
    """This function has 5 inputs that we get two  (Job_Title,Experience) from the two of the additional functions (title_finder and experience_finder)
    and determining the other 3 (E1=60, E2=84, E3=12).
    E1 determines if the person is a manager or a supervisor,
    E2 determins if a person is a lead or not,
    and E3 determines if a person is a junior or not.
    Note:
    We only need the two first inputs to get the outcomes as the others are predetermined."""

    SL = ["junior", "senior", "lead", "supervisor", "manager", "director", "c_suite", "individual_contributor"]
    # SL = ["junior", "senior", "lead", "supervisor", "manager", "director", "cxo", "ic"]
    try:
        LJT = (remove_urls(Job_Title)).lower().split()  # word list
    except:
        Last_Job_Title = str(Job_Title)
        LJT = Last_Job_Title.lower().split()  # word list
    # print(LJT)
    TWE = float(Experience)
    # print(f"The job title was {LJT},and the experience needed was {TWE}")
    # Seniority level (sl)
    sl = 8
    # Managerial indicators CSI (C-Suit), DI (Director) MI (Manager)
    CSI = ["chief", "officer", "executive", "dirigeant", "cadre dirigeant", "cadres dirigeants", "vice", "president",
           "vp", "ceo", "président", "head", "chef", "managing", "haute direction", "vice-présidente", "vice",
           "co-founder", "founder"]
    MI = "manager"
    DI = "director"
    DIf = "directeur"
    MLI = CSI + [MI] + [DI] + [DIf]
    # Entry level indicators
    ELI = ["entry level", "intern", "apprentice", "aid", "support", "clerk", "new grad", "junior", "trainee",
           "assistant", "associate", "specialist", "temp", "jr", "novice", "stagiaire", "apprenti", "aid?", "support",
           "commis", "diplomé", "junior", "assistant", "associé", "specialiste", "interimaire"]
    if TWE == 0.0 or LJT == "":
        sl = 8
    else:
        if len(LJT) != 1 or len(Job_Title) > 4:
            x = 0
            for w in LJT:
                if w in MLI:
                    x += 1
            if x != 0 and TWE >= E1:
                if MI in LJT:
                    sl = 5
                elif DI in LJT or DIf in LJT:
                    sl = 6
                else:
                    sl = 7
            elif x == 0 and TWE <= E2:
                for w in LJT:
                    if w in ELI and TWE < E3:
                        sl = 1
                        break
                    else:
                        sl = 2
            elif x != 0 and TWE < E1:
                sl = 4
            else:
                sl = 3
        else:
            A = re.search('^C', Job_Title)
            B = re.search('O$', Job_Title)
            if (A != None) and (B != None):
                if TWE > E1:
                    sl = 7
                else:
                    sl = 4
            else:
                sl = 8  # When the title has less than 4 letters and is not C-X-O
    # print(sl,SL[sl-1])
    return sl, SL[sl - 1]

def CNP_NOC_Taxonomy(text:str,SourceName:str,out_type:str):
    """This function will extract CNP/NOC taxonomy from the text if existed in 3 levels + the synonyms found in the text
    The outcome could be a list of dictionaries or a dictionary of lists
    The dictionary of lists is easier to be used we just need to used indexes for filtering if not all taxonomy types are needed, based on higher levels
    (Major groups(Knowledge, Skills, ...), Minor groups(type of knowledge, type of skills, ...)"""
    lan = detect(text)

    dict_out = {"Taxonomy_MajorGroup": [], "Taxonomy_MinorGroup": [], "Taxonomy_Descriptor": [], "Taxonomy_Found": []}
    list_out = []
    table_service, filter_query = set_table_service(), f""
    df = pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query, SOURCE_TABLE=SourceName))
    df.reset_index(inplace=True, drop=True)
    parts = df["PartitionKey"].unique()
    # print(parts)
    # print(df)
    # text_word_list = text.split(" ")
    # for word in text_word_list:
    n, t, d, w = [], [], [], []
    for i, row in df.iterrows():
        item = row["Descriptor"].lower()
        item = re.sub(r"\([^()]*\)", "", item)
        item = item.strip(" ")
        # print(item)
        # tax_lam = WordNetLemmatizer().lemmatize(row["Descriptor"], pos="n")
        synonyms = [item]
        if lan == "fr":
            lan = "fre"
            for syn in wn.synsets(item, pos=wn.NOUN):

                for i in syn.lemmas(lang=lan):
                    synonyms.append(i.name().lower())
        else:
            for syn in wn.synsets(item, pos=wn.NOUN):

                for i in syn.lemmas():
                    synonyms.append(i.name().lower())
        synonyms = list(set(synonyms))
        # print(synonyms)
        x = 0
        wl = []
        out = {"Taxonomy_MajorGroup": "", "Taxonomy_MinorGroup": "", "Taxonomy_Descriptor": "", "Taxonomy_Found": []}
        for synonym in synonyms:
            if synonym in text.lower():
                if x == 0:
                    out["Taxonomy_MajorGroup"] = row["PartitionKey"]
                    n += [row["PartitionKey"]]
                    out["Taxonomy_MinorGroup"] = row["Type"]
                    t += [row["Type"]]
                    out["Taxonomy_Descriptor"] = row["Descriptor"]
                    d += [row["Descriptor"]]
                    x += 1
                wl += [synonym]
        if wl != []:
            out["Taxonomy_Found"] = wl
            w += [wl]
        if out != {"Taxonomy_MajorGroup": "", "Taxonomy_MinorGroup": "", "Taxonomy_Descriptor": "", "Taxonomy_Found": []}:
            list_out.append(out)
    dict_out["Taxonomy_MajorGroup"] = n
    dict_out["Taxonomy_MinorGroup"] = t
    dict_out["Taxonomy_Descriptor"] = d
    dict_out["Taxonomy_Found"] = w
    if out_type == "dict":
        return dict_out
    if out_type == "list":
        return list_out
# nltk.download('omw-1.4')

##################################################################################### Testing the info extraction functions ###################################

def info_extraction(file_text:str, source_version:int, score_limit:float, taxonomy_source:str, title_limit:int) -> dict:
    """"
    This function will test all the additional functions all at once having a file_text as the input
    outputs are return in a dictionary to be saved in the DB:
    # Not time consuming (almost nothing)
    names: A list of top words starting with capital letter found in the job description (we can use this list to find names like company name and such)
    abbreviations: A list of top abbreviations found in the job description (we can use this list as a data feature in the matching later)
    job_type: Part_time or Full_time based on the key words search in the job description
    location: A location in Canada having city province in found in the job description
    experience: A number showing the amount of experience they have specified in the job description
    seniority_level_final: A number between 1 and 8 (1,2,3 for technical path 4,5,6,7 for managerial path 8 for those not having the inputs (experience or job title)
    seniority_level_name_final: A string, name of the level (SL = ["junior", "senior", "lead", "supervisor", "manager", "director", "c_suite", "individual_contributor"])
    # A bit time-consuming (Around 3 sec)
    Matchedjob: List of rearranged job titles that were found and their group was the most repeated one
    BroadCategory: List of broad categories for the matched job titles
    MajorGroup: List of Major groups for the matched job titles
    SubMajorGroup: List of Sub_Major groups for the matched job titles
    MinorGroup: List of Minor groups for the matched job titles
    UnitGroup: List of Unit groups for the matched job titles
    alter_jobs: List of other job titles that could have been a match based on the Unit group the matches were in
    confidence: The level of confidence we have in the rearranged titles based on the level we have found the most repeated group (Great, Good, Sufficient, Low)
    sentence: The sentences we have found the match to double check and be used for later analysis and matching
    """
    # print(f"Your are testing with CNP_NOC_{source_version}")
    if source_version == 16:
        """Test with CNP NOC 2016"""
        SourceName = "CNPNOC16"
        alljobs_col_en = "NOCIndexOfTitles"
        alljobs_col_fr = "CNPIndexOfTitles"
        cl_fr = ["RowKey","CNPIndexOfTitles", "CNPProfileName", "CNPMinorGroup", "CNPMajorGroup"]
        cl_en = ["RowKey","NOCIndexOfTitles", "NOCProfileName", "NOCMinorGroup", "NOCMajorGroup"]
    else:
        """Test with CNP NOC 2021"""
        SourceName = "CNPNOC"
        alljobs_col_en = "NOCAllExamples"
        alljobs_col_fr = "CNPAllExamples"
        cl_fr = ["RowKey","CNPAllExamples", "CNPUnitGroup", "CNPMinorGroup", "CNPMajorGroup", "CNPSubMajorGroup",
                 "CNPBroadCategory"]
        cl_en = ["RowKey","NOCAllExamples", "NOCUnitGroup", "NOCMinorGroup", "NOCMajorGroup", "NOCSubMajorGroup",
                 "NOCBroadCategory"]
    """Testing the title and domain finder"""
    start = datetime.now()

    unit_id, matched_job, MajorGroup, UnitGroup, MinorGroup, alter_jobs, SubMajorGroup, BroadCategory, matched_sentences_list, matched_scores_list, match_location = title_domain_finder_CNPNOC(file_text=file_text, cl_en=cl_en, cl_fr=cl_fr, SourceName=SourceName,score_limit=score_limit)
    # print(f"matched_job:\n{matched_job},\nUnitGroup:\n{UnitGroup},\nMinorGroup:\n{MinorGroup},\nSubMajorGroup:\n{SubMajorGroup},\nMajorGroup:\n{MajorGroup},\nBroadCategory:\n{BroadCategory},\nalternative jobs:\n{alter_jobs}")
    t10 = datetime.now() - start
    """Testing Name finder"""
    start3 = datetime.now()
    names = name_finder(Text=file_text)
    # print(f"Names:\n{names}")
    """Testing Abbreviation finder"""
    abbreviations = abbreviation_finder(Text=file_text)
    # print(f"Abbreviations:\n{abbreviations}")
    """Testing Job type finder"""
    job_type = job_type_finder(Text=file_text)
    # print(f"Job Type:{job_type}")
    """Testing Experience finder"""
    experience = experience_finder(Text=file_text)
    # print("Experience:", experience)
    """Testing Location finder"""
    location = location_finder(Text=file_text)
    # print(f"Location:{location}")
    tr = datetime.now() - start3
    start4 = datetime.now()

    unit_id, Matchedjob, BroadCategory, MajorGroup, SubMajorGroup, MinorGroup, UnitGroup, alter_jobs, confidence, sentence = title_sorting(unit_id=unit_id,title_limit=title_limit,
        matched_job=matched_job, matched_sentence=matched_sentences_list, BroadCategory=BroadCategory,
        MajorGroup=MajorGroup, SubMajorGroup=SubMajorGroup, MinorGroup=MinorGroup, UnitGroup=UnitGroup,
        alter_jobs=alter_jobs)
    # print("OK in title finder")
    # print("###############################################################################")
    #print(f"The jobs are now rearranged based on the domain repetition:\nMatchedJob:\n{Matchedjob},\nAlternativeJobs:\n{alter_jobs},\nUnitGroup:\n{UnitGroup},\nMinorGroup:\n{MinorGroup},\nSubMajorGroup:\n{SubMajorGroup},\nMajorGroup:\n{MajorGroup},\nBroadCategory:\n{BroadCategory},\nConfidence:\n{confidence},\nSentence:\n{sentence}")
    print("###############################################################################")
    t8 = datetime.now() - start4
    start5 = datetime.now()
    seniority_level, seniority_level_name = [], []
    for job_title in Matchedjob:
        experience = re.findall(r'\d+', experience)
        try:
            experience = experience[-1]
            if int(experience) >= 40:
                experience = "0.0"
        except:
            experience = "0.0"
        # print(experience)
        sl, sl_name = seniority_level_finder(Job_Title=job_title, Experience=experience, E1=60, E2=84, E3=24)
        seniority_level += [sl]
        seniority_level_name += [sl_name]
    seniority_level_final = max(set(seniority_level), key=seniority_level.count)
    seniority_level_name_final = max(set(seniority_level_name), key=seniority_level_name.count)
    t9 = datetime.now() - start5
    tr += t9
    taxonomy_dict = CNP_NOC_Taxonomy(text=file_text, SourceName=taxonomy_source, out_type="dict")
    # taxonmy_dict = {"Taxonomy_MajorGroup": 'str', "Taxonomy_MinorGroup": 'str', "Taxonomy_Descriptor": "str","Taxonomy_Found": ["list"]}
    # print("Seniority:", seniority_level_name_final)
    # print(t8)
    t = datetime.now() - start
    # print(f"Total run time is {t}, the run time for title/domain finder is {t10}, and the run time for the other functions is {tr}, it took {t8} to rearrange the titles based on domains")
    # print(f"Total run time is {t}, the run time for title/domain finder is {t10}, the run time for title finder is {t6}, the run time for domain finder is {t7}, and the run time for the other functions is {tr}, it took {t8} to rearrange the titles based on domains")
    dictionary_out = {"CNPTitle": Matchedjob, "CNPBroadCategory": BroadCategory, "CNPMajorGroup": MajorGroup,
                      "CNPSubMajorGroup": SubMajorGroup, "CNPMinorGroup": MinorGroup, "CNPUnitGroup": UnitGroup,
                      "CNPAlternativeJobs": alter_jobs, "MatchingConfidence": confidence, "Sentence": sentence,
                      "SeniorityLevel": seniority_level_name_final, "Location": location, "Experience": experience,
                      "JobType": job_type, "Abbreviations": abbreviations, "Names": names, "RunTime": t,
                      "Taxonomy_MajorGroup": taxonomy_dict["Taxonomy_MajorGroup"], "Taxonomy_MinorGroup": taxonomy_dict["Taxonomy_MinorGroup"],
                      "Taxonomy_Descriptor": taxonomy_dict["Taxonomy_Descriptor"], "Taxonomy_Found": taxonomy_dict["Taxonomy_Found"], "Unit_ID": unit_id}
    # return Matchedjob, BroadCategory, MajorGroup, SubMajorGroup, MinorGroup, UnitGroup, alter_jobs, confidence, sentence, seniority_level_name_final, location, experience, job_type, abbreviations, names
    return dictionary_out

def DataEnrich_blob(partition_key:str,container_name:str, connection_string:str, source_version:int, file_ID_list:list, file_name_list:list, test_size:int, SourceNameOut:str, score_limit:float,text_len:int, outcome_format:str,title_limit:int, taxonomy_source = "CNPNOCTaxonomy"):
    """
    This function aims to access the files in a container having connection_string + container_name and enrich file content with data features extracted from the file text and store the features in a table service having CONNECTION_STRING + SourceNameOut (table name)
    There are 4 ways to limit the files we want to parse:
    1. We can give it a list of id s (file_ID_list) to work on or give it an empty list
    2. We can give it a list of file names s (file_name_list) to work on or give it an empty list
    3. If we give it an empty list for both file name and file id s it will parse all files that have not been parsed before,
    4. we can also test the process, giving it test_size to do the process for limited number of files and if we put 0 for this variable it will do all files
    score_limit limits the titles matched and will only return the ones that had score above the limit
    text_len is a limit for the min words that the file should have and if the file has less words, that file is skipped
    """
    ts = TableClient.from_connection_string(CONNECTION_STRING, SourceNameOut)
    pk1 = "EC_Candidates"
    table_service1, filter_query1 = set_table_service(), f"PartitionKey eq '{pk1}'"
    blob_svc = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_svc.get_container_client(container_name)
    blob_list = container_client.list_blobs()
    files = [blob.name for blob in blob_list]
    # print(files)
    if outcome_format == "TableService":
        try:
            ExistingData = pd.DataFrame(get_data_from_table_storage_table(table_service1, filter_query1, SOURCE_TABLE=SourceNameOut))
            # print("Size of the Existing data:", len(ExistingData))
            existing_files = ExistingData["FileName"].tolist()
            # print(len(existing_files))
            files = [file for file in files if file not in existing_files]
            # print(len(files))
            # print(files)
        except:
            print("No old data was found")
    extention = (".docx", ".pdf", ".txt", ".doc")
    files = [file for file in files if file.endswith(extention)]
    if file_ID_list != []:
        file_name_list = []
        for f in files:
            for item in file_ID_list:
                if str(item) in (f.split("_")[-1]).split(".")[0]:
                    file_name_list.append(f)
                    # print(f)
                    break
        print("Number of files ready to be done:",len(file_name_list))
    elif file_name_list != []:
        file_name_list = file_name_list
    else:
        file_name_list = files
    s = 0
    if test_size == 0:
        test_size = -1
    list_of_JSONs = []
    for file in file_name_list[:test_size]:
        fileName = file
        time_s = time.process_time()
        try:
            s += 1
            "ID at the end"
            id = (file.split("_")[-1]).split(".")[0]
            idb = (file.split("_")[0]).split(".")[0]
            if id.isnumeric():
                id = id
            else:
                id = "0"
            blobClient = container_client.get_blob_client(file)
            with open(file, "wb") as f:
                data = blobClient.download_blob()
                f.write(data.readall())
            m_time = os.path.getmtime(file)
            dt_m = datetime.fromtimestamp(m_time)
            text, file_type = file_reader(file)
            # print(text)
            os.remove(file)
            if text != "" and len(text.split(" ")) > text_len:
                try:
                    # print(fileName)
                    dictionary_out = info_extraction(file_text=text, source_version=source_version, score_limit=score_limit, taxonomy_source=taxonomy_source,title_limit=title_limit)
                    # print("OK")
                    lan = detect(text)
                    length = len(text.split(" "))
                    entity = {u'PartitionKey': u"{}".format(partition_key), u'RowKey': u"{}".format(id),
                              'CNPTitle': "{}".format(dictionary_out["CNPTitle"]),
                              'Sentence': "{}".format(dictionary_out["Sentence"]),
                              "Unit_ID": "{}".format(dictionary_out["Unit_ID"]),
                              'MatchingConfidence': "{}".format(dictionary_out["MatchingConfidence"]),
                              'CNPUnitGroup': "{}".format(dictionary_out["CNPUnitGroup"]),
                              'CNPMinorGroup': "{}".format(dictionary_out["CNPMinorGroup"]),
                              'CNPMajorGroup': "{}".format(dictionary_out["CNPMajorGroup"]),
                              'CNPBroadCategory': "{}".format(dictionary_out["CNPBroadCategory"]),
                              'CNPSubMajorGroup': "{}".format(dictionary_out["CNPSubMajorGroup"]),
                              "CNPAlternativeJobs": "{}".format(dictionary_out["CNPAlternativeJobs"]),
                              "Taxonomy_MajorGroup": "{}".format(dictionary_out["Taxonomy_MajorGroup"]),
                              "Taxonomy_MinorGroup": "{}".format(dictionary_out["Taxonomy_MinorGroup"]),
                              "Taxonomy_Descriptor": "{}".format(dictionary_out["Taxonomy_Descriptor"]),
                              "Taxonomy_Found": "{}".format(dictionary_out["Taxonomy_Found"]),
                              "Location": "{}".format(dictionary_out["Location"]),
                              "Experience": "{}".format(dictionary_out["Experience"]),
                              "JobType": "{}".format(dictionary_out["JobType"]),
                              "Abbreviations": "{}".format(dictionary_out["Abbreviations"]),
                              "Names": "{}".format(dictionary_out["Names"]),
                              "SeniorityLevel": "{}".format(dictionary_out["SeniorityLevel"]),
                              "FileText": "{}".format(text), "FileUpdateDate": "{}".format(dt_m),
                              "FileType": "{}".format(file_type),"FileName": "{}".format(fileName),
                              "Language": "{}".format(lan),"TextLength": "{}".format(length),
                              "RunTime": "{}".format(dictionary_out["RunTime"]),
                              # "SelectionStatus": "{}".format(selection(length, dt_m, Unit, UnitGroupList, duration, text_length))
                              }
                    # print("OK")
                    for key in entity.keys():
                        if len(entity[key].encode("utf-16")) > 32000:
                            entity[key] = "OverSized"
                            # print("large string:",key)
                    if outcome_format == "TableService":
                        try:
                            c = ts.upsert_entity(entity)
                            print(f"Text for file:'{file}' was matched and was added to the table")
                        except Exception as e:
                            print(f"Data_Insert_Error : {e}")
                            print(f"Text for file:'{file}' could not be added to the database")
                        print("++++++++++++++++++++++++")
                    else:
                        json_object = json.dumps(entity, indent=4)
                        list_of_JSONs.append(json_object)
                except Exception as e:
                    print("Process Error",str(e))
            else:
                pass
        except Exception as e:
            print("Total Process Error",str(e))
        print(f"Text {file} process time : {time.process_time() - time_s}")
    return list_of_JSONs

def DataEnrich_SQL(partition_key:str,SQL_access_dict:dict, source_version:int, file_ID_list:list, test_size:int, SourceNameOut:str, score_limit:float,text_len:int,column_name_list:list,outcome_format:str,title_limit:int,taxonomy_source= "CNPNOCTaxonomy"):
    """This function aims to access the text in a SQL server having SQL_access_dict
    (A dict with this format: SQL_access_dict = {"driver" :"{ODBC Driver 17 for SQL Server}",
    "server" :"rp-client-input.database.windows.net",
    "database" :"RP_client_inputs-EC",
    "username" :"myadministrator",
    "password" :"97FZkgkGhXPjSHc",
    "query_in"  :'SELECT * FROM Jobs'})
    and enrich content with data features extracted from the text and store the features in a table service having CONNECTION_STRING + SourceNameOut (table name)
    we can give it a list of id s (FileIDList) to work on or give it an empty list to parse all files, we can also test the process giving it test_size to do the process for limited number of files and if we put 0 for this variable it will do all files
    socore_limit limits the titles matched and will only return the ones that had score above the limit
    text_len is a limit for the min words that the file should have and if the file has less words that file is skiped
    """
    ts = TableClient.from_connection_string(CONNECTION_STRING, SourceNameOut)
    # pk1 = "EC_Jobs"
    table_service1, filter_query1 = set_table_service(), ""
    driver = SQL_access_dict["driver"]
    server = SQL_access_dict["server"]
    database = SQL_access_dict["database"]
    username = SQL_access_dict["username"]
    password = SQL_access_dict["password"]
    query_in = SQL_access_dict["query_in"]
    data = data_grabber(driver, server, database, username, password, query_in)
    if outcome_format == "TableService":
        try:
            ExistingData = pd.DataFrame(get_data_from_table_storage_table(table_service1, filter_query1, SOURCE_TABLE=SourceNameOut))
            # print("Size of the Existing data:", len(ExistingData))
            existing_ids = ExistingData["RowKey"].tolist()
            print(len(existing_ids))
            file_ID_list = [id for id in file_ID_list if str(id) not in existing_ids]
            # print(len(files))
            # print(files)
        except Exception as e:
            print("No old data was found",e)
    # print(len(FileIDList))
    cln_id = column_name_list[0]
    # print(cln_id)
    # print(data.info())
    data = data[data[cln_id].isin(file_ID_list)]
    # print(data.info())
    cln_description = column_name_list[1]
    # text_list = data[cln_description].tolist()
    if test_size != 0:
        data = data.head(test_size)
    list_of_JSONs = []
    for i, row in data.iterrows():
        id = row[cln_id]
        text = row[cln_description]
        # print(text)
        if text != "" and len(text.split(" ")) > text_len:
            time_s = time.process_time()
            try:
                # print(id)
                dictionary_out = info_extraction(file_text=text, source_version=source_version, score_limit=score_limit,taxonomy_source=taxonomy_source,title_limit=title_limit)
                lan = detect(text)
                length = len(text.split(" "))
                entity = {u'PartitionKey': u"{}".format(partition_key), u'RowKey': u"{}".format(id),
                          'CNPTitle': "{}".format(dictionary_out["CNPTitle"]),
                          'Sentence': "{}".format(dictionary_out["Sentence"]),
                          "Unit_ID": "{}".format(dictionary_out["Unit_ID"]),
                          'MatchingConfidence': "{}".format(dictionary_out["MatchingConfidence"]),
                          'CNPUnitGroup': "{}".format(dictionary_out["CNPUnitGroup"]),
                          'CNPMinorGroup': "{}".format(dictionary_out["CNPMinorGroup"]),
                          'CNPMajorGroup': "{}".format(dictionary_out["CNPMajorGroup"]),
                          'CNPBroadCategory': "{}".format(dictionary_out["CNPBroadCategory"]),
                          'CNPSubMajorGroup': "{}".format(dictionary_out["CNPSubMajorGroup"]),
                          "CNPAlternativeJobs": "{}".format(dictionary_out["CNPAlternativeJobs"]),
                          "Taxonomy_MajorGroup": "{}".format(dictionary_out["Taxonomy_MajorGroup"]),
                          "Taxonomy_MinorGroup": "{}".format(dictionary_out["Taxonomy_MinorGroup"]),
                          "Taxonomy_Descriptor": "{}".format(dictionary_out["Taxonomy_Descriptor"]),
                          "Taxonomy_Found": "{}".format(dictionary_out["Taxonomy_Found"]),
                          "Location": "{}".format(dictionary_out["Location"]),
                          "Experience": "{}".format(dictionary_out["Experience"]),
                          "JobType": "{}".format(dictionary_out["JobType"]),
                          "Abbreviations": "{}".format(dictionary_out["Abbreviations"]),
                          "Names": "{}".format(dictionary_out["Names"]),
                          "SeniorityLevel": "{}".format(dictionary_out["SeniorityLevel"]),
                          "FileText": "{}".format(text), #"FileUpdateDate": "{}".format(dt_m),
                          #"FileType": "{}".format(file_type), "FileName": "{}".format(fileName),
                          "Language": "{}".format(lan), "TextLength": "{}".format(length),
                          "RunTime": "{}".format(dictionary_out["RunTime"]),
                          # "SelectionStatus": "{}".format(selection(length, dt_m, Unit, UnitGroupList, duration, text_length))
                          }
                for key in entity.keys():
                    if len(entity[key].encode("utf-16")) > 32000:
                        entity[key] = "OverSized"
                        # print("large string:", key)
                if outcome_format == "TableService":
                    try:
                        c = ts.upsert_entity(entity)
                        print(f"Text for ID:'{id}' was matched and was added to the table")
                    except Exception as e:
                        print(f"Data_Insert_Error : {e}")
                        print(f"Text for ID:'{id}' could not be added to the database")
                    # print("++++++++++++++++++++++++")
                else:
                    json_object = json.dumps(entity, indent=4)
                    # print(json_object)
                    list_of_JSONs.append(json_object)
            except Exception as e:
                print("Total Process Error",e)
            print(f"Text {id} process time : {time.process_time() - time_s}")
    return list_of_JSONs

def filenamelist2parse(filename="EC_CV_JP_CNP.csv",limit=5,cln="JP_MatchedCount",outfilename="parsing_Id_list.csv"):
    df = pd.read_csv(filename,encoding="utf-8-sig")
    df = df[df[cln] >= limit]
    # print(df.CV_FileID)
    cv_id_list = df["CV_FileID"].tolist()
    cv_id_list_n = []
    for j in cv_id_list:
        for i in parseval(j):
            cv_id_list_n.append(i)
    cv_id_list = list(set(cv_id_list_n))
    print("CVID:",len(cv_id_list))
    # print(cv_id_list)
    jp_id_list = df["JP_FileID"].tolist()
    jp_id_list_n = []
    for j in jp_id_list:
        for i in parseval(j):
            jp_id_list_n.append(i)
    jp_id_list = list(set(jp_id_list_n))
    print("JPID:",len(jp_id_list))
    parsing = {"CV_ID": cv_id_list, "JP_ID": jp_id_list}
    df = pd.DataFrame.from_dict(parsing, orient='index')  # Convert dict to df
    # print(df)
    if outfilename != "":
        df.to_csv(outfilename, header=False)  # Convert df to csv
        with open(outfilename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(parsing.keys())
            writer.writerows(zip(*parsing.values()))
    # parsing_df = pd.DataFrame(parsing)
    # parsing_df.to_csv(outfilename,index=False)
    return jp_id_list,cv_id_list


if __name__ == "__main__":
    """Test with CNP NOC 2016"""
    "To test open a text file and the set fileText to the string"
    # extraction_test(fileText=JD,source_version=16, score_limit=0.4)
    """Test with CNP NOC 2021 with one cv text and one jd text"""
    # dict_JD = extraction_test(fileText=JD, source_version=21, score_limit=0.4)
    # print(dict_JD)
    # dict_CV = extraction_test(fileText=CV, source_version=21, score_limit=0.1)
    # print(dict_CV)

    query_in_cv = """SELECT * FROM CVSentenceMatches2"""
    query_in_jp = """SELECT * FROM Jobs"""
    driver = "{ODBC Driver 17 for SQL Server}"
    server = "rp-client-input.database.windows.net"
    database = "RP_client_inputs-EC"
    username = "myadministrator"
    password = "97FZkgkGhXPjSHc"

    """Getting a list of id s for jobs and cvs to parse 
    (Note: You need to have a csv file with unit groups and the jobs and cv count and matches to run it)"""

    # CVfilenamelist2parse(filename="EC_CV_JP_CNP.csv")
    """Test feature functions with CVs stored on the Azure container"""
    FileNameList = ["02 martingarand 2019_129191.docx","-JPhilippeThibeault-CV_129769.pdf","!CV_MJBoucher_121482.pdf", "-CV Marie a.k. Defendini Leccia -_112090.pdf",
                    "01-CV-French-Dilawer Khozem_149639.docx", "02 martingarand 2019_129190.docx"]

    "Test for returning outcomes in a list of JSON s having file id list (We can keep a list of added file lists and then use it to run this function in the parser)"
    # JSON_list = DataEnrich_blob(partition_key:"EC_Candidates",container_name="eccv", connection_string=CONNECTION_STRING, source_version=21, test_size=100, SourceNameOut="ECData", score_limit=0.1, file_ID_list=[], file_name_list=FileNameList, text_len=200, outcome_format="JSON", taxonomy_source="CNPNOCTaxonomy",title_limit=7)
    # print(JSON_list)

    # JSON_list = DataEnrich_blob(partition_key:"EC_Candidates",container_name="eccv", connection_string=CONNECTION_STRING, source_version=21,
    #                             test_size=0, SourceNameOut="ECData", score_limit=0.1, file_ID_list=[],
    #                             file_name_list=FileNameList, text_len=200, outcome_format="TableService",
    #                             taxonomy_source="CNPNOCTaxonomy",title_limit=7)
    # print(JSON_list)
    # Grabing file id list having a csv file with jobs and cv preparsed cnp/noc data

    jp_id_list,cv_id_list = filenamelist2parse(filename="EC_CV_JP_CNP.csv", limit=5, cln="JP_MatchedCount", outfilename="")
    #print(jp_id_list,cv_id_list)

    "Test for storing outcomes in a tale service having file id list"
    DataEnrich_blob(partition_key="EC_Candidates",container_name="eccv", connection_string=CONNECTION_STRING, source_version=21, test_size=0, SourceNameOut="ECCV", score_limit=0.1,file_ID_list=cv_id_list, file_name_list=[], text_len=200, outcome_format="TableService", taxonomy_source="CNPNOCTaxonomy",title_limit=7)
    "Test for returning outcomes in a list of JSON s having file id list"
    # JSON_list = DataEnrich_blob(partition_key="EC_Candidates",container_name="eccv", connection_string=CONNECTION_STRING, source_version=21, test_size=5, SourceNameOut="ECCV", score_limit=0.1, file_ID_list=cv_id_list, file_name_list=[], text_len=200, outcome_format="JSON", taxonomy_source="CNPNOCTaxonomy")
    # print(JSON_list)
    SQL_access_dict = {"driver" :"{ODBC Driver 17 for SQL Server}",
    "server" :"rp-client-input.database.windows.net",
    "database" :"RP_client_inputs-EC",
    "username" :"myadministrator",
    "password" :"97FZkgkGhXPjSHc",
    "query_in"  :"""SELECT * FROM Jobs"""}

    "Testing with SQL inputs"
    #JSON_list = DataEnrich_SQL(partition_key="EC_Jobs",SQL_access_dict=SQL_access_dict, source_version=21, file_ID_list=jp_id_list, test_size=7, SourceNameOut="ECJobsJSON", score_limit=0.1, text_len=200, column_name_list=["Id","Description"],outcome_format="JSON",title_limit=7)
    # print(JSON_list)
    #DataEnrich_SQL(partition_key="EC_Jobs", SQL_access_dict=SQL_access_dict, source_version=21, file_ID_list=jp_id_list,test_size=0, SourceNameOut="ECJobsJSON", score_limit=0.1, text_len=200,column_name_list=["Id", "Description"], outcome_format="TableService", title_limit=7)
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
work/JDExtractor.py at main · Hosna878/workwork/TextParser.py at main · Hosna878/work