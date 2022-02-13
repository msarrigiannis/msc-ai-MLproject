from pdfminer.high_level import extract_text
import docx2txt
import sys
import pathlib
import pandas as pd
import re
import nltk
import itertools
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from pickle import load
import numpy as np

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

def processText(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('\S*@\S*\s?', '  ', text)  # remove emails
    text = re.sub('[%s]' % re.escape("""!"$%'()*,/:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text.lower() #lower case

def scoreCandidate(text):
  #extract words
  tokens = nltk.tokenize.word_tokenize(text)

  #generate bigrams and trigrams (e.g. object oriented programming)
  ngrams = list(map(' '.join, nltk.everygrams(tokens, 2, 3)))

  skillScore = 0
  cmpScore = 0
  educationLevel = -1
  educationScore = 0

  #enumerate tokens and look for degrees/skills/companies
  for token in tokens:
    if token not in stopwords.words('english'):
      if token in SKILLS:
        skillScore += 1
      if token in COMPANIES:

        cmpScore += 1
      if token in EDUCATION:
        if EDUCATION[token] > educationLevel:
          educationLevel = EDUCATION[token]     

  #enumerate ngrams and look for companies/skills
  for ngram in ngrams:
    if ngram in SKILLS:
      skillScore += 1
    if ngram in COMPANIES:
      cmpScore += 1

  for univers in UNIVERSITIES:
    if re.search(univers, text):
      educationScore += 1
    

  #return pd.Series({'skills_score': skillScore, 'education_level': educationLevel, 'education_score': educationScore, 'company_score': cmpScore})
  return np.array([skillScore, educationLevel, educationScore, cmpScore])
  #return [skillScore, educationLevel, educationScore, cmpScore]

#list of all degrees encoded 0 = high school, 1 = bachelors, 2 = masters, 3= phd
EDUCATION = {
    'ssc': 0, 
    'hsc': 0, 
    'cbse': 0, 
    'icse': 0, 
    'xii': 0,
    'be' : 1,
    'b.e.': 1,
    'b.e': 1, 
    'bs': 1, 
    'b.s': 1, 
    'btech': 1, 
    'b.tech': 1,
    'me': 2, 
    'm.e': 2, 
    'm.e.': 2, 
    'ms': 2, 
    'm.s': 2, 
    'm.tech': 2, 
    'mtech': 2,
    'phd': 3
}

filepath = sys.argv[1]
skillsPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[2]
companiesPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[3]
universityPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[4]

if filepath.endswith('pdf'):
    text = extract_text_from_pdf(str(pathlib.Path(__file__).parent.resolve()) + '\\' + filepath)
elif filepath.endswith('docx'):
    text = extract_text_from_docx(filepath)
else:
    print("Invalid file format, must be pdf or docx")
    exit()


#load skills and companies list
SKILLS = list(filter(None, open(skillsPath).read().lower().split('\n')))
COMPANIES = list(filter(None, open(companiesPath).read().lower().split('\n')))

#universities might have first the city and then the name or vice versa
unis = filter(None, open(universityPath).read().lower().split('\n'))
UNIVERSITIES = []
for uni in unis:
  for permutation in list(itertools.permutations(uni.split(','))):
    UNIVERSITIES.append(re.sub('\s+', ' ', ' '.join(permutation).strip()))

# load the model
model = load(open(str(pathlib.Path(__file__).parent.resolve()) + '\\' + 'model.pkl', 'rb'))
# load the scaler
scaler = load(open(str(pathlib.Path(__file__).parent.resolve()) + '\\' + 'scaler.pkl', 'rb'))

#process text
processed_text = processText(text)

#extract features
features = scoreCandidate(processed_text)

featuresScaled = scaler.transform(features.reshape(1, -1))

prediction = model.predict(featuresScaled)

pred_text = "None"

if prediction == 1:
    pred_text = "Bad"
elif prediction == 2:
    pred_text = "Good"
elif prediction == 3:
    pred_text = "Maybe"

print("Candidate scores are: " + str({'skills_score': features[0], 'education_level': features[1], 'education_score': features[2], 'company_score': features[3]}))
print("Model prediction is: " + pred_text)

