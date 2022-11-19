"""This file contains all the text cleaning and preprocessing functions
Developer: Hosna Hamdihe"""
import string, nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import docx2txt
import textract
import re,io,os
import numpy as np
import ast

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