"""
This script includes all the functions used in the extractor (Job posting info extractor)
Developer: Hosna Hamdieh
Start date: 2022-01-21
"""
# Libs
import requests
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd
import re
import os
from urllib.parse import unquote
import numpy as np
from datetime import datetime, timedelta

############################################################# Basic functions ##########################################

def blacklist(textfilename="BL.txt"):
    textfile=open(textfilename,"r")
    bl=textfile.read()
    blacklist=bl.split("\n")
    blacklist=np.unique(blacklist).tolist()
    textfile.close()
    # print(blacklist)
    return blacklist

############################################################# Link extractors ##########################################
############################################################ LinkedIn ##################################################

# Extracting the initial HTML data
def load_LinkedIn_div(JobTitle,Location):
    JT=JobTitle.replace(" ","%20")
    # print(JT)
    Loc = Location.replace(" ","%20%20")
    # print(Loc)
    # url=f"https://www.linkedin.com/jobs/search/?f_TPR=r86400&geoId=101330853&keywords={JT}&location={Loc}"
    # url = f"https://www.linkedin.com/jobs/search/?keywords={JT}&location={Loc}&locationId=&geoId=101174742&f_TPR=r86400&position=1&pageNum=0"
    url = f"https://ca.linkedin.com/jobs/search/?keywords={JT}&location={Loc}&locationId=&geoId=105829038&f_TPR=r86400&distance=25&position=1&pageNum=0"
    # print("url:",url)
    page = requests.get(url)
    # print(page.status_code)
    soup = BeautifulSoup(page.content,"html.parser")
    job_soup = soup.find("ul",class_='jobs-search__results-list')
    # print(job_soup)
    return job_soup

# Extracting job links from LinkedIn data source
def extract_links_LinkedIn(JobTitle,Location):
    "This function extracts job post links from LinkedIn"
    job_elem=load_LinkedIn_div(JobTitle,Location)
    Job_links=[]
    # Company_links=[]
    for link in job_elem.find_all("a"):
        l=link.get("href")
        if l.endswith("trk=public_jobs_jserp-result_search-card"):
            Job_links.append(l)
        # else:
        #     Company_links.append(l)
    return Job_links
# Testing
# links = extract_links_LinkedIn("Electronic engineer","Ottawa")
# print(links)
# Extracting job links from LinkedIn data source
def extract_links_CompanyLinks_LinkedIn(JobTitle,Location):
    "This function extracts both job post links and company links from LinkedIn"
    job_elem=load_LinkedIn_div(JobTitle,Location)
    Job_links=[]
    Company_links=[]
    for link in job_elem.find_all("a"):
        l=link.get("href")
        if l.endswith("trk=public_jobs_jserp-result_search-card"):
            Job_links.append(l)
        else:
            Company_links.append(l)
    return Job_links,Company_links
############################################################ Jobillico ##################################################

# Extracting the initial HTML data
def load_Jobillico_jobs_div(JobTitle,Location):
    l=Location.replace(", ","%2C%20")
    jt=JobTitle.replace(" ","%20")
    # print(jt)
    url=(f"https://www.jobillico.com/search-jobs?skwd={jt}&scty={l}&icty=0&ipc=0&sil=&sjdpl=&sdl=&imc1=0&imc2=0&flat=0&flng=0&mfil=40&ipg=1&clr=1")
    # print("url:",url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content,"html.parser")
    # print(soup.prettify())
    job_soup = soup.find("div",id="jobOffersList")
    return job_soup

def extract_links_Jobillico(JobTitle,Location):
    "This function extracts job post links from Jobillico"
    job_elem = load_Jobillico_jobs_div(JobTitle,Location)
    Job_links=[]
    # Company_links=[]
    for link in job_elem.find_all("a"):
        l=link.get("href")
        # print(l)
        if l.startswith("/en/job-offer/"):
            l="https://www.jobillico.com"+l
            # print(l)
            Job_links.append(l)
        # elif l.startswith("/see-company/"):
        #     l="https://www.jobillico.com/"+l
        #     # print(l)
        #     Company_links.append(l)
    # print("Job links:",Job_links)
    # print("Companies' LinkedIn page:",Company_links)
    return Job_links

def extract_links_CompanyLinks_Jobillico(JobTitle,Location):
    "This function extracts both job post links and company links from Jobillico"
    job_elem = load_Jobillico_jobs_div(JobTitle,Location)
    Job_links=[]
    Company_links=[]
    for link in job_elem.find_all("a"):
        l=link.get("href")
        # print(l)
        if l.startswith("/en/job-offer/"):
            l="https://www.jobillico.com"+l
            # print(l)
            Job_links.append(l)
        elif l.startswith("/see-company/"):
            l="https://www.jobillico.com/"+l
            # print(l)
            Company_links.append(l)
    # print("Job links:",Job_links)
    # print("Companies' LinkedIn page:",Company_links)
    return Job_links, Company_links

############################################################ PostJobFree ##################################################

# Extracting the initial HTML data
def load_PostJobFree_jobs_div(JobTitle,Location):
    l=Location.replace(", ","%2C+")
    jt=JobTitle.replace(" ","+")
    # print(jt)
    # we can change the number at the end to get more or less outcomes in each run. As this function will be run overtime to reduce redundent job posts I have reduced it from 100 to 50
    url=(f"https://www.postjobfree.com/jobs?q={jt}&l={l}&radius=25&r=50")
    # print("url:",url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content,"html.parser")
    # print(soup.prettify())
    job_soup = soup.find("div",class_="stdContentLayout innercontent")
    return job_soup
# soup=load_PostJobFree_jobs_div("Data Scientist","Ottawa, ON, Canada")

def extract_links_PostJobFree(JobTitle,Location):
    "This function extracts job post links from PostJobFree"
    job_elem = load_PostJobFree_jobs_div(JobTitle,Location)
    Job_links=[]
    # Company_links=[]
    try:
        for link in job_elem.find_all("a"):
            l=link.get("href")
            # print(l)
            if l.startswith("/job/"):
                l="https://www.postjobfree.com"+l
                # print(l)
                Job_links.append(l)
    except:
        pass
    # print("Job links:",Job_links)
    # print("Companies' LinkedIn page:",Company_links)
    # print("Done")
    return Job_links

############################################################ Indeed ##################################################

# SOUP
def load_Indeed_jobs_div(JobTitle,Location):
    getVars = {"q":JobTitle,"l":Location,"formage":"last","sort":"date"}
    url_rest=urllib.parse.urlencode(getVars)
    url=(f"https://ca.indeed.com/jobs?{url_rest}")
    # print("url:",url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content,"html.parser")
    job_soup = soup.find(id="mosaic-provider-jobcards")
    # print("job_soup:    ",job_soup)
    return job_soup

# Extracting the links to the full job postings
def extract_links_indeed(JobTitle,Location):
    job_elem=load_Indeed_jobs_div(JobTitle,Location)
    links=[]
    Clinks=[]
    for link in job_elem.find_all("a"):
        links.append(link.get("href"))
    # print("links:",links)
    link_list=[]
    for l in links:
        if l.startswith("/rc/clk?"):
            l=l.replace("/rc/clk?","")
            L = "https://ca.indeed.com/viewjob?"+l
            link_list.append(L)
    # print("Done")
    return link_list

############################################################# Link extractor ##########################################

def LinkExtractor(JobTitle,Location):
    "This function will extract job posting links from platforms, LinkedIn, PostJobFree, Jobillico, Indeed"
    inL, iL, jL, pjfL = [],[],[],[]

    # extracting job links form LinkedIn
    try:
        inL = extract_links_LinkedIn(JobTitle,Location)
    except:
        pass
    # extracting job links form Indeed
    try:
        iL = extract_links_indeed(JobTitle,Location)
    except:
        pass
    # extracting job links form Jobillico
    try:
        jL = extract_links_Jobillico(JobTitle, Location)
    except:
        pass
    # extracting job links form PostJobFree
    try:
        pjfL = extract_links_PostJobFree(JobTitle,Location)
    except:
        pass
    Links = inL + iL + jL + pjfL
    Note = f"""
-------------------------------------------------------------------------------------------------------
There are {len(Links)} Job posts found on the date {datetime.now()} by searched job title '{JobTitle}':"
LinkedIn: {len(inL)},
Indeed: {len(iL)}, 
Jobillico: {len(jL)}, 
PostJobFree: {len(pjfL)},
-------------------------------------------------------------------------------------------------------
"""
    Note1= f"""
-------------------------------------------------------------------------------------------------------
There are {len(Links)} Job posts found on the date {datetime.now()} by searched job title '{JobTitle}':"
LinkedIn: {len(inL)},
Indeed: {len(iL)}, 
Jobillico: {len(jL)}, 
PostJobFree: {len(pjfL)},
List of links:
{Links}
-------------------------------------------------------------------------------------------------------
"""
    # with open('Logs.txt', 'a') as f:
    #     f.write(Note)
    # print(Note)
#     print(f"""
# -------------------------------------------------------------------------------------------------------
# There are {len(Links)} Job posts found on the date {datetime.now()} by searched job title '{JobTitle}'
# -------------------------------------------------------------------------------------------------------""")
    print(Note)
    return Links
# Testing LinkExtractor: Fully functional
# x = LinkExtractor("Electronic engineer", "Ottawa, ON")
# print(f"""There are {len(x)} Job posts found:
# {x}""")
############################################################# Info extractors ##########################################
# Extracting Job post info, having the job post link
############################################################ LinkedIn ##################################################

def extract_company_LikedIn(link):
    "This function extracts the company's name having LinkedIn job posting link"
    companyName=""
    try:
        company_elem = re.search('-at-(.+?)-[0-9]+', link).group(1).replace("?","")
        companyName = unquote(company_elem)
    except:
        pass
    # print("Company Name:", companyName)
    return companyName
# # Test
# print(extract_company_LikedIn(links[0]))
def extract_title_linkedIn(link):
    "This function extracts the job title having LinkedIn job posting link"
    title = ""
    try:
        title_elem = re.search('view/(.+?)-at', link).group(1).replace("-"," ")
        title_elem = title_elem.replace("%E2%80%93"," - ")
        title_elem = title_elem.replace("d%E2%80%99"," - ")
        title_elem = title_elem.replace("donn%C3%A9"," - ")
        title = unquote(title_elem)
    except:
        pass
    # print("Titles:",titles)
    return title
# Test
# print(extract_title_linkedIn(links[0]))
def extract_text_LinkedIn(link):
    T = ""

    # print("link:",url)
    page = requests.get(link)
    job_elem = BeautifulSoup(page.content,"html.parser")
    # text = job_elem.strings
    BL=["","\n",'','\n','LinkedIn','Expand search', 'Jobs', 'People', 'Learning', 'Dismiss','Join now', 'Sign in','Apply on company website','You can save your resume and apply to jobs in minutes on LinkedIn', 'Sign in', 'Join now', 'You�re signed out', 'Sign in for the full experience', 'Sign in', 'Join now',]+['Expand search', 'Jobs', 'People', 'Learning', 'Dismiss', 'Dismiss', 'Dismiss', 'Dismiss', 'Dismiss', 'Join now', 'Sign in', 'Apply on company website', 'Be among the first 25 applicants', 'Apply on company website', 'Save', 'Save job', 'Save this job with your existing LinkedIn profile, or create a new one.', 'Your job seeking activity is only visible to you.', 'Email', 'Continue', 'Welcome back', 'Sign in to save', 'at', '.', 'Email or phone', 'Password', 'Show', 'Forgot password?', 'Sign in', 'Report this job', 'Sourcing and recruitment strategies', 'WORKPLACE', 'LinkedIn Sponsored', 'Show more', 'Show less', 'Seniority level', 'Employment type', 'Job function', 'Industries', 'Referrals increase your chances of interviewing at Intact by 2x', 'See who you know', 'Turn on job alerts', 'Turn on job alerts', 'On', 'Off', 'LinkedIn', '� 2021', 'About', 'Accessibility', 'User Agreement', 'Privacy Policy', 'Cookie Policy', 'Copyright Policy', 'Brand Policy', 'Guest Controls', 'Community Guidelines', 'Dansk (Danish)', 'Deutsch (German)', 'English (English)', 'Espa�ol (Spanish)', 'Fran�ais (French)', 'Bahasa Indonesia (Bahasa Indonesia)', 'Italiano (Italian)', 'Bahasa Malaysia (Malay)', 'Nederlands (Dutch)', 'Norsk (Norwegian)', 'Polski (Polish)', 'Portugu�s (Portuguese)', 'Svenska (Swedish)', 'Tagalog (Tagalog)', 'T�rk�e (Turkish)', 'Language', 'Create job alert', 'Get email updates for new','jobs in', 'Dismiss', 'By creating this job alert, you agree to the LinkedIn', 'User Agreement', 'and', 'Privacy Policy', '. You can unsubscribe from these emails at any time.', 'Sign in to create more', 'Create job alert', 'Your job alert is set', 'Click the link in the email we sent to', 'to verify your email address and activate your job alert.', 'Done', 'Welcome back', 'Sign in to create your job alert for', 'jobs in', '.', 'Email or phone', 'Password', 'Show', 'Forgot password?', 'Sign in', 'Save time applying to future jobs', 'You can save your resume and apply to jobs in minutes on LinkedIn', 'Sign in', 'Join now', 'You�re signed out', 'Sign in for the full experience', 'Sign in', 'Join now']+["© 2021","Русский (Russian)",
"ภาษาไทย (Thai)",
"Türkçe (Turkish)",
"简体中文 (Chinese (Simplified))",
"正體中文 (Chinese (Traditional))",
"العربية (Arabic)",
"Čeština (Czech)",
"Español (Spanish)",
"Français (French)",
"हिंदी (Hindi)",
"日本語 (Japanese)","한국어 (Korean)",
"Português (Portuguese)",
"Română (Romanian)","🙂"]
    texts = job_elem.stripped_strings
    # print(texts,type(texts))
    for t in texts:
        if t not in BL:
            try:
                te=(unquote(t)).encode('utf-8')
                te=te.decode("utf-8")
                # print(te)
                T+=("\n"+te)
                # print("Total text",T)
            except:
                continue
    # print("details:",details)
    # print(len(urls))
    # print("details1",details1)
    return T
# print(extract_text_LinkedIn(links[0]))
def extract_info_LinkedIn(link):
    "This function extracts Job posting info, having LinkedIn job posting link"
    # Link = link
    company = extract_company_LikedIn(link)
    title = extract_title_linkedIn(link)
    text = extract_text_LinkedIn(link)
    # source = "LinkedIn"
    return title, company, text
# print(extract_info_LinkedIn(links[0]))
############################################################ Jobillico ##################################################

def extract_info_Jobillico(link):
    "This function extracts Job posting info, having Jobillico job posting link"
    # Link = link
    title = ""
    company = ""
    text =""
    # source="Jobillico"
    page = requests.get(link)
    soup = BeautifulSoup(page.content, "html.parser")
    try:
        # company_elem = soup.find("h2",class_="h3 link mt2 mb1").get_text().strip()
        company_elem = soup.find("h2").get_text().strip()
        company = (unquote(company_elem))
        title_elem = soup.find("h1").get_text().strip()
        title=(unquote(title_elem))
    except:
        pass
    # print("Titles:",titles)
    # print("Companies Name:", companiesName)
    BL = blacklist(textfilename="BL.txt")
    texts = soup.stripped_strings
    # print(texts,type(texts))
    for t in texts:
        if t not in BL:
            try:
                te = (unquote(t)).encode('utf-8')
                te = te.decode("utf-8")
                # print(te)
                text += ("\n" + te)
                # print("Total text",T)
            except:
                continue
    return title, company, text
############################################################ PostJobFree ##################################################

def extract_info_PostJobFree(link):
    "This function extracts Job posting info, having PostJobFree job posting link"
    # Link = link
    title = ""
    company = ""
    text = ""
    # source = "PostJobFree"
    page = requests.get(link)
    soup = BeautifulSoup(page.content, "html.parser")
    try:
        # company_elem = soup.find("h2",class_="h3 link mt2 mb1").get_text().strip()
        company_elem = soup.find("span",id="CompanyNameLabel").get_text().strip()
        company = (unquote(company_elem))
        title_elem = soup.find("h1").get_text().strip()
        title = (unquote(title_elem))
    except:
        pass
    # print("link:",url)
    # job_elem = job_elem.find_all("div",class_="card__content__section")
    # text = job_elem.strings
    BL = blacklist(textfilename="BL.txt")
    texts = soup.stripped_strings
    # print(texts,type(texts))
    for t in texts:
        if t not in BL:
            try:
                te = (unquote(t)).encode('utf-8')
                te = te.decode("utf-8")
                # print(te)
                text += ("\n" + te)
                # print("Total text",T)
            except:
                continue
    # print(f"{len(titles)} Titles:",titles)
    # print(f"{len(companiesName)} Companies Name:", companiesName)
    return title, company, text
############################################################ Indeed ##################################################

def extract_info_Indeed(link):
    "This function extracts Job posting info, having Indeed job posting link"
    # Link = link
    title = ""
    company = ""
    text = ""
    # source = "Indeed"
    page = requests.get(link)
    soup = BeautifulSoup(page.content, "html.parser")
    # Job title
    try:
        title_elem = soup.find("h1").get_text()
        title = unquote(title_elem)
    except:
        pass
    try:
        # Company's name
        company_elem = soup.find("div", class_="icl-u-lg-mr--sm icl-u-xs-mr--xs").get_text()
        company = company_elem
    except:
        pass
    try:
        # Job description
        JobInfo = soup.find("div",id="jobDescriptionText").get_text()
        text = JobInfo
    except:
        pass
    return title, company, text

# Location finder
def LocationFinder(text):
    provinces = {
        'Alberta': ['AB','Alberta'],
        'British Columbia': ['BC','British Columbia'],
        'Manitoba': ['MB','Manitoba'],
        'New Brunswick': ['NB','New Brunswick'],
        'Newfoundland and Labrador': ['NL','Newfoundland and Labrador'],
        'Northwest Territories': ['NT','Northwest Territories'],
        'Nova Scotia': ['NS','Nova Scotia'],
        'Nunavut': ['NU','Nunavut'],
        'Ontario': ['ON','Ontario'],
        'Prince Edward Island': ['PE','Prince Edward Island'],
        'Quebec': ['QC','Quebec'],
        'Saskatchewan': ['SK','Saskatchewan'],
        'Yukon': ['YT','Yukon']
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
    for i in provinces:
        WL = provinces[i]
        # print(WL)
        if WL[0] in text or WL[1] in text:
            province = WL[0]
            for c in Cities[i]:
                if c in text:
                    city = c
    return city,province,"Canada"
# text= "Back End Developer / Développeur.euse Back-End job in Warman, SK, Canada - January 2022Sign "
# link=''
# print(LocationFinder(text))
def DuplicationCheck(el,L,jt):
    links = []
    st = []
    # el = (df["ApplyLink"]).tolist()
    # print("existing links in the db", el)
    for i in range(len(L)):

        if "linkedin" in L[i]:
            x = 0
            try:
                c = re.search('https://ca.linkedin.com/jobs/view/(.+?)refId', L[i]).group(1).replace("?","")

                for j in el:
                    if c in j:
                        x = 1
                if x == 1:
                    print("already exists:", L[i])
                else:
                    # print(c)
                    links.append(L[i])
                    st.append(jt[i])
            except:
                pass
        # if L[i] in el:
        #     print(L.pop(i))
        #     print(jt.pop(i))
        elif L[i] not in el:
            links.append(L[i])
            st.append(jt[i])
        else:
            print("already exists:",L[i])
    return links,st

def LinkedInPage(CompanyName):
    cn=(CompanyName.lower()).replace(" ","-")
    cl="https://www.linkedin.com/company/"+cn
    return cl

def remove_urls(TEXT):
    vTEXT0 = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%\@\#\;\!\$\,\:)*\b', '', TEXT, flags=re.MULTILINE)
    vTEXT = re.sub("[^a-zA-Z\s]", " ", vTEXT0)
    return vTEXT

def experienceFinder(text):
    List =["years'","of years experience","années d'expérience", "years of", "ans d’expérience", "years", "year"]
    e=[]
    for c in List:
        e = re.findall(f'([^ \r\n]+) {c}?([\r\n]| |$)', text, re.IGNORECASE)
        if e != []:
            # if ''.join(map(lambda x: str(x[0]), e)).isalpha():
            y = re.sub("[^0-9]", "", ''.join(map(lambda x: str(x[0]), e)))
            if any(c.isalpha() for c in ''.join(map(lambda x: str(x[0]), e))) or len(y) > 4:
                return ""
            else:
                return ' '.join(map(lambda x: str(x[0]), e))
    return ""

def jobType(text):
    FT= ["Full-time","Full Time","Permanent contract", "Permanent"]
    PT= ["Contract","Apprenticeship","Part-time","Part time","Casual","Seasonal","Paid internship","Student employment","Telecommuting","Unpaid internship","Internship","Temporary"]
    JT=""
    for i in FT:
        if i in text:
            JT = "Full_time"
            return "Full_time"
            # break
    for j in PT:
        if j in text:
            JT = "Part_time"
            return "Part_time"
            # break
    if JT == "":
        return ""

def ab_finder(text):
    # ab = re.findall("([A-Z]\.*){2,}s?", text)
    ab = re.findall(r'[A-Z]{2,}', text)
    ab = [i for i in ab if i != "I"]
    ab = list(set(ab))
    return ab
# print(ab_finder(". I have a UPS in the UNI to save all the docs CHIH"))

def name_finder(text):
    List = ["I","You","She","He","We","It","They"]
    ss = re.findall(r". [A-Z]{1,}[a-z]+", text)
    nn = [s.replace(". ","") for s in ss]
    # print(nn)
    names = re.findall(r"[A-Z]{1,}[a-z]+", text)
    # print(names)
    names = [i.strip() for i in names if i.strip() not in List+nn]
    names = list(set(names))
    return names
# print(name_finder("Reza has a UPS in the UNI to save all the docs CHIH. Hosna have Iphone as well."))