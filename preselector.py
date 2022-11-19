"""
This script includes all the functions used in the Preselection process
There are two versions for the preselector one first selecting a sub dataframe using some filters then checking unit groups
and
the other first finding candidates having same Unit groups as the job then checking the other filters.
Pay attention to the format of the inputs specially to the column name list and filter dict
There are also two ways of using it one by 'table services' inputs and the other by 'SQL server' inputs
Developer: Hosna Hamdieh
Start date: 2022-09-21
Updated:
"""
# Libs:

import re
from datetime import *
import string, nltk
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import pandas as pd
from azure.cosmosdb.table.tableservice import TableService
import pyodbc
import textwrap
import ast
import time


CONNECTION_STRING ="DefaultEndpointsProtocol=https;AccountName=jobdetails;AccountKey=3PLXwnKAOAwfDHUtRVTYbUEOvhwPKoZJsyxGVyD/rvVDH8ZCKoXujs+0EMJW2c55hLwUXVapq2W5+AStsLZxAQ==;EndpointSuffix=core.windows.net"

######################################################## Used functions ########################################################
def Data4LastNDays(df,n:int):
    """Selecting a sub dataframe using the tiestamp info"""
    date = df.Timestamp
    date = pd.to_datetime(date.dt.tz_localize(None))
    now = datetime.now()
    df = df[date >= (now - pd.Timedelta(days=n))]
    # print(len(df))
    return df

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

def parseval(s:str):
    """Returning the actual value of a string"""
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s

def remove_urls(text: str) -> str:
    "This function will erase the punctuations from a text"
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%\@\#\;\!\$\,\:)*\b', '', text, flags=re.MULTILINE)
    text = re.sub("[^a-zA-Z\s]", " ", text)
    return text

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

def try_literal_eval(e:str):
    try:
        return ast.literal_eval(e)
    except ValueError:
        return {'average_rating': 0, 'number_of_ratings': 0}

def PreselcetionOffline_version2(candidateEnriched_df,JobProfileEnriched_df,limit_tim:int,limit_tam:int,cl_cnp_id_name:str, cl_id_candidate:str,dict_of_filter:dict, filter_cl_name_dict:dict) -> list:
    """This function is the preselection offline, we need two dataframes as inputs one for candidates and one for the job (pandas series)
    We also need a list of column names related to CNP/NOC enrichment (first item in the list should be the name of the top group, followed by mid group and finally lowest group (unit group),
    we need the level going from 1 to 3 to select the level we need to go to or the CNP/NOC ID (unit group number)
    we need limit to know how many matches we want to see between the job and the CV
    The column names in the database for filtering features is filter_cl_name_dict = {"SeniorityLevel":"SeniorityLevel_column_name","Taxonomy_Descriptor":"Taxonomy_Descriptor_column_name","Location":"Location_column_neme","Language":"Language_column_name","Timestamp":"Timestamp_column_name"}
    dict_of_filter: {"SeniorityLevel":True/False,"Location":True/False,"Language":True/False,"Taxonomy_Descriptor":True/False,"Timestamp":[True/False,Years(int)]} shows the features to use or not to use"""

    # s = time.time()

    cl = cl_cnp_id_name
    selected = []
    JPlist = (JobProfileEnriched_df[cl].apply(try_literal_eval).tolist()[0])
    # print(JPlist,type(JPlist))
    if len(JPlist) > limit_tim:
        limit_tim = len(JPlist)

    jpsl = JobProfileEnriched_df[filter_cl_name_dict["SeniorityLevel"]].tolist()[0]

    jpt = (JobProfileEnriched_df[filter_cl_name_dict["Taxonomy_Descriptor"]].apply(try_literal_eval).tolist()[0])

    jplo = JobProfileEnriched_df[filter_cl_name_dict["Location"]].tolist()[0]

    jpla = JobProfileEnriched_df[filter_cl_name_dict["Language"]].tolist()[0]

    if dict_of_filter["SeniorityLevel"] == True and jpsl != "individual_contributor":
        candidateEnriched_df = candidateEnriched_df[candidateEnriched_df[filter_cl_name_dict["SeniorityLevel"]] == jpsl]

    if dict_of_filter["Location"] == True and jplo != "Canada":
        candidateEnriched_df = candidateEnriched_df[candidateEnriched_df[filter_cl_name_dict["Location"]] == jplo]

    if dict_of_filter["Language"] == True and jpla in ["fr", "en"]:
        candidateEnriched_df = candidateEnriched_df[candidateEnriched_df[filter_cl_name_dict["Language"]] == jpla]

    if dict_of_filter["Timestamp"][0] == True and dict_of_filter["Timestamp"][1]:
        candidateEnriched_df = Data4LastNDays(df=candidateEnriched_df, n=dict_of_filter["Timestamp"][1]*350)

    for i, row in candidateEnriched_df.iterrows():
        try:

            CPlist = ast.literal_eval(row[cl])

            great_match = False
            if JPlist[0] in CPlist:
                great_match = True
            match_list_ti = [item for item in CPlist if item in JPlist]

            if len(match_list_ti) >= limit_tim or great_match == True:
                x = limit_tam

                if dict_of_filter[filter_cl_name_dict["Taxonomy_Descriptor"]] == True and jpt != []:

                    cpt = ast.literal_eval(row[filter_cl_name_dict["Taxonomy_Descriptor"]])
                    match_list_ta = [item for item in cpt if item in jpt]
                    x = len(match_list_ta)
                    # print(x)
                if x >= limit_tam:
                    selected += [row[cl_id_candidate]]
            else:
                continue

        except Exception as e:
            print("Error in preselection:",e)
            pass
    return selected

def PreselcetionOffline(candidateEnriched_df,JobProfileEnriched_df,limit_tim:int,limit_tam:int,cl_cnp_id_name:str, cl_id_candidate:str,dict_of_filter:dict, filter_cl_name_dict:dict) -> list:
    """This function is the preselection offline, we need two dataframes as inputs one for candidates and one for the job (pandas series)
    We also need a list of column names related to CNP/NOC enrichment (first item in the list should be the name of the top group, followed by mid group and finally lowest group (unit group),
    we need the level going from 1 to 3 to select the level we need to go to or the CNP/NOC ID (unit group number)
    we need limit to know how many matches we want to see between the job and the CV
    The column names in the database for filtering features is filter_cl_name_dict = {"SeniorityLevel":"SeniorityLevel_column_name","Taxonomy_Descriptor":"Taxonomy_Descriptor_column_name","Location":"Location_column_neme","Language":"Language_column_name","Timestamp":"Timestamp_column_name"}
    dict_of_filter: {"SeniorityLevel":True/False,"Location":True/False,"Language":True/False,"Taxonomy_Descriptor":True/False,"Timestamp":[True/False,Years(int)]} shows the features to use or not to use"""

    # s = time.time()

    cl = cl_cnp_id_name
    selected = []
    JPlist = (JobProfileEnriched_df[cl].apply(try_literal_eval).tolist()[0])

    if len(JPlist) > limit_tim:
        limit_tim = len(JPlist)

    jpsl = JobProfileEnriched_df[filter_cl_name_dict["SeniorityLevel"]].tolist()[0]

    jpt = (JobProfileEnriched_df[filter_cl_name_dict["Taxonomy_Descriptor"]].apply(try_literal_eval).tolist()[0])

    jplo = JobProfileEnriched_df[filter_cl_name_dict["Location"]].tolist()[0]

    jpla = JobProfileEnriched_df[filter_cl_name_dict["Language"]].tolist()[0]

    if dict_of_filter["Timestamp"][0] == True and dict_of_filter["Timestamp"][1]:
        candidateEnriched_df = Data4LastNDays(df=candidateEnriched_df, n=dict_of_filter["Timestamp"][1]*350)

    for i, row in candidateEnriched_df.iterrows():
        try:

            CPlist = ast.literal_eval(row[cl])

            great_match = False
            if JPlist[0] in CPlist:
                great_match = True
            match_list_ti = [item for item in CPlist if item in JPlist]

            if len(match_list_ti) >= limit_tim or great_match == True:
                sl, lo, la = True, True, True
                x = limit_tam
                if dict_of_filter["SeniorityLevel"] == True and jpsl != "individual_contributor":
                    cpsl = row[filter_cl_name_dict["SeniorityLevel"]]
                    if cpsl != jpsl:
                        sl = False

                if dict_of_filter["Location"] == True and jplo != "Canada":
                    cplo = row[filter_cl_name_dict["Location"]]

                    if cplo != jplo:
                        lo = False

                if dict_of_filter["Language"] == True and jpla in ["fr","en"]:
                    cpla = row[filter_cl_name_dict["Language"]]

                    if cpla != jpla:
                        la = False

                if dict_of_filter["Taxonomy_Descriptor"] == True and jpt != []:

                    cpt = ast.literal_eval(row[filter_cl_name_dict["Taxonomy_Descriptor"]])

                    match_list_ta = [item for item in cpt if item in jpt]
                    x = len(match_list_ta)

                if sl == True and lo == True and la == True and x >= limit_tam:
                    selected += [row[cl_id_candidate]]
            else:
                continue

        except Exception as e:
            print("Error in preselection:",e)
            pass
    return selected

def preselector_with_tableservice_offLine(version:int,job_id_list:list,JobSourceName:str,CandidateSourceName:str,TestSize:int,filter_cl_name_dict:dict,limit_tim:int,limit_tam:int,dict_of_filter:dict, cl_id_job:str, cl_id_candidate:str, cl_cnp_id_name:str) -> dict:
    """This function will test the preselection using the table service with offline enriched data
    A list of job id s can be inputted and it will return a dictionary having job id s as keys and a list of selected CV id as values
    The column names in the database for filtering features is filter_cl_name_dict = {"SeniorityLevel":"SeniorityLevel_column_name","Taxonomy_Descriptor":"Taxonomy_Descriptor_column_name","Location":"Location_column_neme","Language":"Language_column_name","Timestamp":"Timestamp_column_name"}
    we can decide which filters to use and which not to use by inputting dict_of_filter in the format below:
    dict_of_filter: {"SeniorityLevel":True/False,"Location":True/False,"Language":True/False,"Taxonomy_Descriptor":True/False,"Timestamp":[True/False,Years(int)]}
    we can put limits for the number of matches for both unit groups (limit_tim) and taxonomy items (limit_tam)
    """

    selected_dict = {}
    for JobID in job_id_list[:TestSize]:
        s = time.time()
        table_service_job, filter_query_job = set_table_service(), f"{cl_id_job} eq '{JobID}'"
        df_job = pd.DataFrame(get_data_from_table_storage_table(table_service_job, filter_query_job, SOURCE_TABLE=JobSourceName))

        filter = ""
        table_service_candidates, filter_query_candidates = set_table_service(), filter
        df_candidates = pd.DataFrame(get_data_from_table_storage_table(table_service_candidates, filter_query_candidates, SOURCE_TABLE=CandidateSourceName))
        if version == 1:
            SelectionList = PreselcetionOffline(candidateEnriched_df=df_candidates, JobProfileEnriched_df=df_job, filter_cl_name_dict=filter_cl_name_dict, limit_tim=limit_tim, limit_tam=limit_tam, dict_of_filter=dict_of_filter, cl_cnp_id_name=cl_cnp_id_name, cl_id_candidate=cl_id_candidate)
        else:
            SelectionList = PreselcetionOffline_version2(candidateEnriched_df=df_candidates, JobProfileEnriched_df=df_job,filter_cl_name_dict=filter_cl_name_dict, limit_tim=limit_tim,limit_tam=limit_tam, dict_of_filter=dict_of_filter,cl_cnp_id_name=cl_cnp_id_name, cl_id_candidate=cl_id_candidate)
        e = time.time()
        t = e - s
        print(f"{len(SelectionList)} candidates were found for this job and the run time was {t} seconds")
        print(f"Selected list of CV ID for {JobID}: {SelectionList}")
        selected_dict[JobID] = SelectionList
    return selected_dict

def preselector_with_SQL_offLine(JobID:int,SQL_access_dict:dict,name_dict:dict,min_num_candidates:int,limit_tim_percent:int,limit_tam_percent:int,dict_of_filter:dict):
    """This function will test the preselection using the SQL server with offline enriched data
    Pay attention to the format of the inputs specially the dict inputs (SQL_access_dict:dict,name_dict:dict,dict_of_filter:dict)
    A list of job id s can be inputted and it will return a dictionary having job id s as keys and a list of selected CV id as values
    We need sql access information:
    SQL_access_dict  = {"driver" :"{ODBC Driver 17 for SQL Server}",
                        "server" :"rp-client-input.database.windows.net",
                        "database" :"RP_client_inputs-EC",
                        "username" :"myadministrator",
                        "password" :"97FZkgkGhXPjSHc"}
    We can decide which filters to use and which not to use by inputting dict_of_filter in the format below:
    dict_of_filter: {"SeniorityLevel":True/False,"Location":True/False,"Language":True/False,"Taxonomy_Descriptor":True/False,"Timestamp":[True/False,Years(int)]}
    we can put limits for the number of matches for both unit groups (limit_tim) and taxonomy items (limit_tam) by giving the percentage of match we are looking for (limit_tim_percent:int,limit_tam_percent:int, both between 0 to 100)
    The column names in the database for filtering features is given in the dict below along with table names used in queries
    name_dict = {"job_cnp_id_table_name": "jobs_cnp_unit_group",
                    "candidate_cnp_id_table_name": "candidates_cnp_unit_group",
                    "job_parsed_table_name": "parsed_jobs",
                    "candidate_parsed_table_name": "Candidates",
                    "cl_cnp_id_name": "cnp_id",
                    "cl_id_candidate_cnp": "candidate_id",
                    "cl_id_job_cnp": "parsed_job_id",
                    "cl_id_candidate": "cv_id",
                    "cl_id_job": "job_id",
                    "SeniorityLevel": "SeniorityLevel",
                    "Taxonomy_Descriptor": "Taxonomy_Descriptor", "Location": "Location",
                    "Language": "Language", "Timestamp": "Timestamp"}
    """
    start_time = datetime.now()
    selected_dict = {}
    # accessing the sql server and extracting candidateEnriched_df and JobProfileEnriched_df to continue the process
    driver = SQL_access_dict["driver"]
    server = SQL_access_dict["server"]
    database = SQL_access_dict["database"]
    username = SQL_access_dict["username"]
    password = SQL_access_dict["password"]

    # Job cnp list
    job_query_cnp = f"""SELECT *
FROM [dbo].[{name_dict["job_cnp_id_table_name"]}]
WHERE {name_dict["cl_id_job_cnp"]}={JobID}"""
    df_job_cnp = data_grabber(driver=driver, server=server, database=database, username=username, password=password, query_in=job_query_cnp)
    cnp_id_list = df_job_cnp[name_dict["cl_cnp_id_name"]].tolist()
    cnp_id_list = [i for i in cnp_id_list if i!=""]+["ph"]
    filter_cnp = tuple(cnp_id_list)
    # print(filter_cnp)
    # Candidates
    candidate_query_cnp = f"""SELECT {name_dict["cl_id_candidate_cnp"]}   
FROM [dbo].[{name_dict["candidate_cnp_id_table_name"]}]
WHERE {name_dict["cl_cnp_id_name"]} IN {filter_cnp}"""
    df_candidates_cnp = data_grabber(driver=driver, server=server, database=database, username=username, password=password, query_in=candidate_query_cnp)
    # print(df_candidates)
    candidate_id_list = df_candidates_cnp[name_dict["cl_id_candidate_cnp"]].tolist()
    # potential_candidate_number = len(list(set(candidate_id_list)))
    # print(potential_candidate_number)
    # Getting job info
    job_query = f"""SELECT *
                  FROM [dbo].[{name_dict["job_parsed_table_name"]}]
                  WHERE {name_dict["cl_id_job"]}={JobID}"""
    df_job = data_grabber(driver=driver, server=server, database=database, username=username, password=password,
                          query_in=job_query)
    jpsl = df_job[name_dict["SeniorityLevel"]].tolist()[0]
    jplo = df_job[name_dict["Location"]].tolist()[0]
    jpla = df_job[name_dict["Language"]].tolist()[0]
    jpt = ast.literal_eval(df_job[name_dict["Taxonomy_Descriptor"]].tolist()[0])
    # Using the limit percentages to calculate the number of times we want to see a match between candidate and the job in unit groups and taxonomy
    limit_tam = int(len(jpt) * limit_tam_percent / 100)
    limit_tim = int(len(filter_cnp) * limit_tim_percent / 100)
    # using limit of number of times a certain id appeared having matching cnp id as the job
    candidate_id_count_dict = {i: candidate_id_list.count(i) for i in candidate_id_list}
    candidate_id_list = [i for i in candidate_id_count_dict.keys() if candidate_id_count_dict[i] >= limit_tim]
    filter_cv_id = tuple(candidate_id_list)
    candidate_query = f"""SELECT *
            FROM [dbo].[{name_dict["candidate_parsed_table_name"]}]
            WHERE {name_dict["cl_id_candidate"]} IN {filter_cv_id}"""
    potential_candidate_num = len(filter_cv_id)
    # print("Candidates matching cnp filter:",potential_candidate_num)
    # print(df_job)
    if potential_candidate_num != 0 and potential_candidate_num >= min_num_candidates:
        s = time.time()
        if dict_of_filter["Timestamp"][0] == True:
            now = datetime.now()
            acceptable_date = now - timedelta(days=365*dict_of_filter["Timestamp"][1])
            # print(acceptable_date,type(acceptable_date))
            candidate_query += f""" AND {name_dict["Timestamp"]} > CAST('{acceptable_date}' as Date)"""
        if dict_of_filter["SeniorityLevel"] == True and jpsl != "individual_contributor":
            candidate_query += f""" AND {name_dict["SeniorityLevel"]}='{jpsl}'"""

        if dict_of_filter["Location"] == True and jplo != "Canada":
            candidate_query += f""" AND {name_dict["Location"]}='{jplo}'"""

        if dict_of_filter["Language"] == True and jpla in ["FR", "EN"]:
            candidate_query += f""" AND {name_dict["Language"]}='{jpla.lower()}'"""

        df_candidates = data_grabber(driver=driver, server=server, database=database, username=username, password=password, query_in=candidate_query)

        if dict_of_filter["Taxonomy_Descriptor"] == True and jpt != "":

            id_list = []
            for i, row in df_candidates.iterrows():
                # print(row[name_dict["Taxonomy_Descriptor"]])
                cpt = row[name_dict["Taxonomy_Descriptor"]].split("|")
                match_list_ta = [item for item in cpt if item in jpt]
                x = len(match_list_ta)
                if x >= limit_tam:
                    # print("selected")
                    id_list.append(row[name_dict["cl_id_candidate"]])

            df_candidates = df_candidates[df_candidates[name_dict["cl_id_candidate"]].isin(id_list)]
        # print(df_candidates)
        SelectionList = df_candidates[name_dict["cl_id_candidate"]].tolist()
        e = datetime.now()
        t = e - start_time
        # print(f"{len(SelectionList)} candidates were found for this job and the run time was {t} seconds")
        # print(f"Selected list of CV ID for {JobID}: {SelectionList}")
        selected_dict[JobID] = SelectionList
    elif potential_candidate_num < min_num_candidates and potential_candidate_num != 0:
        candidate_query = f"""SELECT *
            FROM [dbo].[{name_dict["candidate_parsed_table_name"]}]
            WHERE {name_dict["cl_id_candidate"]} IN {filter_cv_id}"""
        df_candidates = data_grabber(driver=driver, server=server, database=database, username=username, password=password, query_in=candidate_query)
        # print(df_candidates)
        SelectionList = df_candidates[name_dict["cl_id_candidate"]].tolist()
        e = datetime.now()
        t = e - start_time
        # print(f"{len(SelectionList)} candidates were found for this job and the run time was {t} seconds")
        # print(f"Selected list of CV ID for {JobID}: {SelectionList}")
        selected_dict[JobID] = SelectionList
    else:
        print("No candidate found")
        df_candidates = pd.DataFrame(columns=[], index=[])

    return df_candidates, df_job



if __name__ == "__main__":

    # Inputs (not related to the data source)
    job_id_list = [1, 2, 11]
    dict_of_filter = {"SeniorityLevel": True, "Location": True, "Language": True, "Taxonomy_Descriptor": True, "Timestamp": [True, 2]}

###################################################################
    """Testing with a list of job ID s with table service"""
    # filter_cl_name_dict = {"SeniorityLevel": "SeniorityLevel",
    #                        "Taxonomy_Descriptor": "Taxonomy_Descriptor", "Location": "Location",
    #                        "Language": "Language", "Timestamp": "Timestamp"}
    # selection_dict = preselector_with_tableservice_offLine(version=2, job_id_list=job_id_list, JobSourceName="ECJobsJSON", CandidateSourceName="ECCV", TestSize=3, filter_cl_name_dict=filter_cl_name_dict, limit_tim=1, limit_tam=1, dict_of_filter=dict_of_filter, cl_cnp_id_name="Unit_ID", cl_id_candidate="RowKey", cl_id_job="RowKey")
    # print(selection_dict)
###################################################################
    """Testing with a list of job ID s with SQL servers"""
    # Modify the list below based on the DB also modify the below inputs based on SQL DB

    SQL_access_dict  = {"driver" :"{ODBC Driver 17 for SQL Server}",
    "server" :"rp-client-input.database.windows.net",
    "database" :"RP_client_inputs-EC",
    "username" :"myadministrator",
    "password" :"97FZkgkGhXPjSHc"}

    name_dict = {"job_cnp_id_table_name": "jobs_cnp_unit_group",
                    "candidate_cnp_id_table_name": "candidates_cnp_unit_group",
                    "job_parsed_table_name": "parsed_jobs",
                    "candidate_parsed_table_name": "Candidates",
                    "cl_cnp_id_name": "cnp_id",
                    "cl_id_candidate_cnp": "candidate_id",
                    "cl_id_job_cnp": "parsed_job_id",
                    "cl_id_candidate": "id",
                    "cl_id_job": "id", "SeniorityLevel": "seniority_level","Taxonomy_Descriptor": "taxonomy_descriptor",
                 "Location": "locations","Language": "lang", "Timestamp": "creation_date_2"}
    df_candidates,df_job = preselector_with_SQL_offLine(JobID=7, SQL_access_dict=SQL_access_dict, name_dict=name_dict, min_num_candidates=10, limit_tim_percent=23, limit_tam_percent=50, dict_of_filter=dict_of_filter)
    print(df_candidates,df_job)