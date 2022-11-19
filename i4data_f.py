
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