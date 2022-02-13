from pdfminer.high_level import extract_text
import docx2txt
import os
import csv
import re
from datetime import datetime

filePath = 'FrontEnd_Data/'

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

header = ['text', 'label']

with open('log.txt', 'a', encoding='UTF8') as log:
    log.write('{0} -- {1}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M"), "Operation started") )
    with open('cv_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        errCount = 0

        for dir in os.scandir(filePath):                     
            for file in os.scandir(dir):
                try:
                    if file.path.endswith('pdf'):
                        text = extract_text_from_pdf(file.path)
                    elif file.path.endswith('docx'):
                        text = extract_text_from_docx(file.path)
                    else:
                        log.write('{0} -- {1}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M"), "\tFile " + file.path + " is not pdf or docx!"))
                        continue

                    writer.writerow([re.compile(r"\s+").sub(" ", text).strip(), dir.name])
                    log.write('{0} -- {1}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M"), "\tProcessing " + file.path + "  OK"))
                except Exception as ex:
                    errCount += 1
                    log.write('{0} -- {1}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M"), "\tError has occured when processing " + file.path + ": " + str(ex)))
                    continue
                finally:
                    log.flush()
                    f.flush()                  
    log.write('{0} -- {1}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M"), "Process finished with : " + str(errCount) + " errors"))

