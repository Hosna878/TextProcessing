"""
This is the main function run on the VM to extract JD having a list of Job titles in Canada, storing the job post information in a table service on the cloud.
For this script to work we must have some libs installed and some files in directory.
Developer: Hosna Hamdieh
Start date: 2022-01-19
"""
#!/usr/bin/python
#Libs
from azure.cosmosdb.table.tableservice import TableService
from azure.data.tables import TableClient
from azure.core.exceptions import ResourceExistsError, HttpResponseError
from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import TableServiceClient

#Libs
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from langdetect import detect

#Libs
from ExtractionFunctions import LinkExtractor, extract_info_LinkedIn, extract_info_PostJobFree, extract_info_Indeed, extract_info_Jobillico,LocationFinder, DuplicationCheck, LinkedInPage, remove_urls, experienceFinder, jobType, ab_finder, name_finder
# from JobDesEx import Data_insert, data_grabber, DuplicationCheck, LinkedInPage, remove_urls, experienceFinder, jobType, ab_finder, name_finder, DupCheck

# Table service connections
# credential = AzureNamedKeyCredential("jobdetails", "bpoQA96H+xpW/I/VOFH3+4hV2AwmWsratSQAtlrWRTfkaHYCGGaKV52yVKhixuqCzUTIC4lyIZr8EKX+DCVRJA==")
# table_service = TableServiceClient(endpoint="mytableendpoint", credential=credential)
CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=jobdetails;AccountKey=bpoQA96H+xpW/I/VOFH3+4hV2AwmWsratSQAtlrWRTfkaHYCGGaKV52yVKhixuqCzUTIC4lyIZr8EKX+DCVRJA==;EndpointSuffix=core.windows.net'
# SOURCE_TABLE = "jobDetails"
# table_service = TableServiceClient.from_connection_string(conn_str='DefaultEndpointsProtocol=https;AccountName=jobdetails;AccountKey=bpoQA96H+xpW/I/VOFH3+4hV2AwmWsratSQAtlrWRTfkaHYCGGaKV52yVKhixuqCzUTIC4lyIZr8EKX+DCVRJA==;EndpointSuffix=core.windows.net')
# table_client = table_service.get_table_client(table_name="jobdetails")
# print(type(table_client))

# import JobillicoDataGrab as jdg
# import LinkedInDataGrab as ldg
# import IndeedDataGrab as idg
# import PostJobFreeDataGrab as pdg
# The database info should be defined here
# driver, server, database, query, username, password, recordset, tableName = "", "", "", "", "", "", "", ""

# The inputs should be determined here

def openFile(filename):
    # filename = askopenfilename(title = "Open Text File", filetypes=(("Text Files","*.txt"),))
    infile = open(filename, "r")
    TEXT = infile.read()
    infile.close()
    return TEXT

def AllLinks(JobTitle,Location):
    "This function will extract all job post links"
    Links=[]
    sj=[]
    for JT in JobTitle:
        Job_Title = JT
        links = (LinkExtractor(JT,Location))
        Links += links
        sj += [JT for i in links]
    print(f"""There are {len(Links)} Job posts found:""")
    return Links,sj
# Testing the function AllLinks: It works fine
# AllLinks("Art teacher","Montreal, QC")

def set_table_service():
    """ Set the Azure Table Storage service """
    return TableService(connection_string=CONNECTION_STRING)

def get_dataframe_from_table_storage_table(table_service, filter_query):
    """ Create a dataframe from table storage data """
    return pd.DataFrame(get_data_from_table_storage_table(table_service,filter_query))

def get_data_from_table_storage_table(table_service, filter_query,SOURCE_TABLE):
    """ Retrieve data from Table Storage """
    for record in table_service.query_entities(SOURCE_TABLE, filter=filter_query):
        yield record

def NewLinks(JobTitle,Location,SOURCE_TABLE):
    "This function will check the list of job posting pages found against the ones already in the DB to find new posts"
    # the filter_query should be determined or eliminated from the inputs
    # dataInDB = data_grabber(driver,server,database,username,password,query)

    table_service, filter_query = set_table_service(), ""
    dataInDB = pd.DataFrame(get_data_from_table_storage_table(table_service,filter_query,SOURCE_TABLE))
    # rows = len(dataInDB)
    column = dataInDB[u'RowKey']
    values = [int(v) for v in column]
    rows = max(values)
    # print(rows)
    # print(dataInDB.head())
    EntityList = dataInDB.T.to_dict().values()
    # EntityList = dataInDB.to_dict()
    # print(EntityList)
    # dataInDB.to_csv("test1.csv", index=False)
    Links, sj = AllLinks(JobTitle,Location)
    # print(Links,sj)
    el= (dataInDB["ApplyLink"]).tolist()
    L,jt = DuplicationCheck(el, Links, sj)
    # print(L,jt,len(L))
    # textfile = open("Links.txt", "a")
    # for element in L:
    #     textfile.write(element + "\n")
    Note = f"""
---------------------------------------------------------------------------------
There are {len(Links)-len(L)} old job posting links, 
excluding the old job postings that are already extracted in previous extractions
The total new posts are {len(L)}
---------------------------------------------------------------------------------
"""
    print(Note)
    # with open('Logs.txt', 'a') as f:
    #     f.write(Note)
    return L,jt,rows

def dataInsert(entity,SOURCE_TABLE):
    "This function will insert the entity if it does not exist in the database"
    with TableClient.from_connection_string(CONNECTION_STRING, SOURCE_TABLE) as table_service:
        try:
            # [START create_entity]
            # print(" - ")
            created_entity = table_service.create_entity(entity)
            # print(" - ")
            # print("Created entity: {}".format(created_entity))
            print("""\n---------------------\nEntity created\n---------------------\n""")
            # [END create_entity]
            return True
        except:
            print("Error occurred, creating the entity: {}".format(entity))
            return False
def extractor(JobTitle,Domain,Location = "Canada"):
    """This is the main function extracting links from the platforms,
    check if they exist in the database and extract info if it is a new job post,
    finally it will insert the data into the tabel service"""
    start = datetime.now()
    SOURCE_TABLE = "jobDetails"
    # Errors = "Errors:\n"
    # JD = ""
    Jobs = {}
    T, C, D, L, S, CL, sword = [], [], [], [], [], [], []
    titles4, companiesName4, texts4, Job_links4, source4, Company_links4 ,sj4 = [], [], [], [], [], [], []
    titles3, companiesName3, texts3, Job_links3, source3, Company_links3, sj3 = [], [], [], [], [], [], []
    titles1, companiesName1, texts1, Job_links1, source1, Company_links1, sj1 = [], [], [], [], [], [], []
    titles2, companiesName2, texts2, Job_links2, source2, Company_links2, sj2 = [], [], [], [], [], [], []
    Today = f"{(datetime.now()).year}-{(datetime.now()).month}-{(datetime.now()).day}"
    Note = f"""
-----------------------------------------------
Job Detail Extraction from in, i, J!, PJF
Domain = {Domain}
Date: {(datetime.now()).year}-{(datetime.now()).month}-{(datetime.now()).day}
-----------------------------------------------
"""
    print(Note)
    # with open('Logs.txt', 'a') as f:
    #     f.write(Note)
    Links, sj, rows = NewLinks(JobTitle,Location,SOURCE_TABLE)
    # except:
    #     Links = AllLinks(JobTitle,Location)
    #     sj = JobTitle
    # Links,sj = AllLinks(JobTitle,Location)
    inserted=0
    failed=0
    inF=0
    iF=0
    jF=0
    pF=0
    for i in range(len(Links)):
        rows += 1
        l = Links[i]
        sjt = sj[i]
        title, company, text, source = "","","",""
        if "postjobfree" in l:
            source = "postjobfree"
            try:
                title, company, text = extract_info_PostJobFree(l)
                titles4.append(title)
                companiesName4.append(company)
                texts4.append(text)
                Job_links4.append(l)
                source4.append(source)
                sj4.append(sjt)
                cl = ""
                if company != "":
                    cl = LinkedInPage(company)
                loc = LocationFinder(text)
                LocStr = f"{loc[0]},{loc[1]},{loc[2]}".replace(" ,", "")
                entity = {u'PartitionKey': u"{}".format(Domain), u'RowKey': u"{}".format(rows),
                          'SearchedTitle': "{}".format(sjt),
                          'Location': "{}".format(LocStr), 'Date': Today, 'JobTitle': "{}".format(title),
                          'JobType': "{}".format(jobType(text)),
                          'Experience': "{}".format(experienceFinder(text)),
                          'Language': "{}".format(detect(remove_urls((text.lower())))), 'Source': source,
                          'Company': "{}".format(company),
                          'CompanyPage': "{}".format(cl), 'ApplyLink': l,
                          'JobDescription': "{}".format(text),
                          'Abbreviations': "{}".format(ab_finder(text)),
                          'Names': "{}".format(name_finder(text)),
                          "City": "{}".format(loc[0]), "Province": "{}".format(loc[1])}
                c = dataInsert(entity,SOURCE_TABLE)
                if c == True:
                    inserted += 1
                else:
                    failed += 1
            except:
                pF += 1
                pass

        elif "jobillico" in l:
            source = "jobillico"
            try:
                title, company, text = extract_info_Jobillico(l)
                titles2.append(title)
                companiesName2.append(company)
                texts2.append(text)
                Job_links2.append(l)
                source2.append(source)
                sj2.append(sjt)
                cl = ""
                if company != "":
                    cl = LinkedInPage(company)
                loc = LocationFinder(text)
                LocStr = f"{loc[0]},{loc[1]},{loc[2]}".replace(" ,", "")
                entity = {u'PartitionKey': u"{}".format(Domain), u'RowKey': u"{}".format(rows),
                          'SearchedTitle': "{}".format(sjt),
                          'Location': "{}".format(LocStr), 'Date': Today, 'JobTitle': "{}".format(title),
                          'JobType': "{}".format(jobType(text)),
                          'Experience': "{}".format(experienceFinder(text)),
                          'Language': "{}".format(detect(remove_urls((text.lower())))), 'Source': source,
                          'Company': "{}".format(company),
                          'CompanyPage': "{}".format(cl), 'ApplyLink': l,
                          'JobDescription': "{}".format(text),
                          'Abbreviations': "{}".format(ab_finder(text)),
                          'Names': "{}".format(name_finder(text)),
                          "City": "{}".format(loc[0]), "Province": "{}".format(loc[1])}
                c = dataInsert(entity,SOURCE_TABLE)
                if c == True:
                    inserted += 1
                else:
                    failed += 1
            except:
                jF += 1
                pass

        elif "linkedin" in l:
            source = "linkedin"
            try:
                title, company, text = extract_info_LinkedIn(l)
                titles1.append(title)
                companiesName1.append(company)
                texts1.append(text)
                Job_links1.append(l)
                source1.append(source)
                sj1.append(sjt)
                cl = ""
                if company != "":
                    cl = LinkedInPage(company)
                loc = LocationFinder(text)
                LocStr = f"{loc[0]},{loc[1]},{loc[2]}".replace(" ,", "")
                entity = {u'PartitionKey': u"{}".format(Domain), u'RowKey': u"{}".format(rows),
                          'SearchedTitle': "{}".format(sjt),
                          'Location': "{}".format(LocStr), 'Date': Today, 'JobTitle': "{}".format(title),
                          'JobType': "{}".format(jobType(text)),
                          'Experience': "{}".format(experienceFinder(text)),
                          'Language': "{}".format(detect(remove_urls((text.lower())))), 'Source': source,
                          'Company': "{}".format(company),
                          'CompanyPage': "{}".format(cl), 'ApplyLink': l,
                          'JobDescription': "{}".format(text),
                          'Abbreviations': "{}".format(ab_finder(text)),
                          'Names': "{}".format(name_finder(text)),
                          "City": "{}".format(loc[0]), "Province": "{}".format(loc[1])}
                c = dataInsert(entity,SOURCE_TABLE)
                if c == True:
                    inserted += 1
                else:
                    failed += 1
            except:
                inF += 1
                pass

        elif "indeed" in l:
            source= "indeed"
            try:
                title, company, text = extract_info_Indeed(l)
                titles3.append(title)
                companiesName3.append(company)
                texts3.append(text)
                Job_links3.append(l)
                source3.append(source)
                sj3.append(sjt)
                cl = ""
                if company != "":
                    cl = LinkedInPage(company)
                loc = LocationFinder(text)
                LocStr = f"{loc[0]},{loc[1]},{loc[2]}".replace(" ,", "")
                entity = {u'PartitionKey': u"{}".format(Domain), u'RowKey': u"{}".format(rows),
                          'SearchedTitle': "{}".format(sjt),
                          'Location': "{}".format(LocStr), 'Date': Today, 'JobTitle': "{}".format(title),
                          'JobType': "{}".format(jobType(text)),
                          'Experience': "{}".format(experienceFinder(text)),
                          'Language': "{}".format(detect(remove_urls((text.lower())))), 'Source': source,
                          'Company': "{}".format(company),
                          'CompanyPage': "{}".format(cl), 'ApplyLink': l,
                          'JobDescription': "{}".format(text),
                          'Abbreviations': "{}".format(ab_finder(text)),
                          'Names': "{}".format(name_finder(text)),
                          "City": "{}".format(loc[0]), "Province": "{}".format(loc[1])}
                c = dataInsert(entity,SOURCE_TABLE)
                if c == True:
                    inserted += 1
                else:
                    failed += 1
            except:
                iF += 1
                pass
        else:
            print("ERROR")
    T = titles4 + titles3 + titles2 + titles1
    C = companiesName4 + companiesName3 + companiesName2 + companiesName1
    D = texts4 + texts3 + texts2 + texts1
    L = Job_links4 + Job_links3 + Job_links2 + Job_links1
    S = source4 + source3 + source2 + source1
    sword = sj4 + sj3 + sj2 + sj1
    Note = f"""
--------------------------------------------------
The overview of information extraction and storage:
--------------------------------------------------
Data extraction:
The source of the data extracted, {len(L)} job post details, are as followed.
'Post Job Free': {len(Job_links4)},
'Indeed': {len(Job_links3)}
'Jobillico': {len(Job_links2)} 
'LinkedIn': {len(Job_links1)} 
The number of failed attempts to extract information from platforms are as followed.
'Post Job Free': {pF}
'Indeed': {iF}
'Jobillico': {jF}
'LinkedIn': {inF}
--------------------------------------------------
Data storing:
There were {failed} failed data insertion attempts to the database
There were {inserted} job posts, successful added to the database
--------------------------------------------------
"""
    print(Note)
    # with open('Logs.txt', 'a') as f:
    #     f.write(Note)
    t = datetime.now()- start
    SOURCE_TABLE = "ExtractorLogs"
    table_service, filter_query = set_table_service(), ""
    dataInDB = pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query, SOURCE_TABLE))
    # rows = len(dataInDB)
    column = dataInDB[u'RowKey']
    values = [int(v) for v in column]
    rows = max(values)
    entity = {u'PartitionKey': u"{}".format(Domain), u'RowKey': u"{}".format(rows+1),
              'PostJobFreeJP': "{}".format(len(Job_links4)), 'PostJobFreeFailed': "{}".format(pF),
              'IndeedJP': "{}".format(len(Job_links3)), 'IndeedFailed': "{}".format(iF),
              'JobillicoJP': "{}".format(len(Job_links2)), 'JobillicoFailed': "{}".format(jF),
              'LinkedInJP': "{}".format(len(Job_links1)),'LinkedInFailed': "{}".format(inF),
              'JDInserted': "{}".format(inserted), 'JDInsertFailed': "{}".format(failed),'RunTime': "{}".format(t)}
    c = dataInsert(entity,SOURCE_TABLE)
    # This part can be commented if there is no need to make a dataframe or an Excel version of the outcomes.
    CL = []
    for c in C:
        if c != "":
            CL.append(LinkedInPage(c))
        else:
            CL.append("")
    # JD += f"From the links we found, we could extract information from only {len(T)} jobs in total "
    sdate = [str(datetime.now()) for i in range(len(L))]
    slocation = [Location for i in range(len(L))]
    abs = [", ".join(ab_finder(t)) for t in D]
    # print(abs)
    names = [", ".join(name_finder(t)) for t in D]
    # print(names)
    JT = [jobType(text) for text in D]
    e = [experienceFinder(text) for text in D]
    # print(e)
    lang = []
    for d in D:
        if d != "":
            try:
                lang.append(detect(remove_urls((d.lower()))))
            except:
                lang.append("")
        else:
            lang.append("")
    try:
        # Jobs dataframe
        Jobs[u"PartitionKey"] = [u"{}".format(Domain) for i in L]
        Jobs[u"RowKey"] = [u"{}".format(s) for s in sword]
        # Jobs["SearchedTitle"] = sword
        Jobs["Location"] = slocation
        Jobs["Date"] = sdate
        Jobs["JobTitle"] = T
        # Jobs["JobTitleEn"] = ET
        Jobs["JobType"] = JT
        Jobs["Experience"] = e
        if lang != ["" for i in range(len(L))]:
            Jobs["Language"] = lang
        Jobs["Source"] = S
        Jobs["Company"] = C
        Jobs["CompanyPage"] = CL
        Jobs["ApplyLink"] = L
        Jobs["JobDescription"] = D
        # Jobs["JobDescriptionEn"] = ED
        Jobs["Abbreviations"] = abs
        Jobs["Names"] = names

    except:
        print("\nAn error occurred extracting data from platforms\n")
        # Errors += "\nAn error occurred extracting data from platforms\n"

    df_jd = pd.DataFrame(Jobs)
    print("Data extracted:\n", df_jd.head())
    # name = f"{Domain}_{Location}_{datetime.today().year}_{datetime.today().month}_{datetime.today().day}.xlsx"
    # df_jd.to_excel(name)
    # JD += f"\nThe outcomes are saved under the name {name}\n"
    # dataInsert(df_jd)
    # print("Details:\n", JD)

    return df_jd

# def datacorrection(SOURCE_TABLE):
#     table_service, filter_query = set_table_service(), ""
#     df = pd.DataFrame(get_data_from_table_storage_table(table_service,filter_query,SOURCE_TABLE))
#     for row in df:
#         loc = LocationFinder(row['JobDescription'].value)
#         LocStr = f"{loc[0]},{loc[1]},{loc[2]}".replace(" ,", "")
#         entity = {"Date":'{}'.format(((row["Date"].value).split(" "))[0]),'Location': "{}".format(LocStr),"City": "{}".format(loc[0]), "Province": "{}".format(loc[1])}
#         try:
#             created_entity = table_service.upsert_entity(entity)
#         except:
#             print("Error in updating")
# SOURCE_TABLE = "jobDetails"
# datacorrection(SOURCE_TABLE)

#Testing extractor function
# The list of job title related to IT is saved under the name "JT_IT.txt" som make sure it is in the directory or replace the name with the new list
Jt = np.unique((openFile("JT_IT.txt")).split("\n")).tolist()
extractor(JobTitle = Jt ,Domain="IT",Location = "Canada")

# To Test with small job title list
# Jt=["Electronic engineer"]
# Jt = np.unique((openFile("testJL.txt")).split("\n")).tolist()
# extractor(Jt,"Test")

# print(JobTitle)
# extractor(["Data Scientist"],"Test","Montreal, QC")