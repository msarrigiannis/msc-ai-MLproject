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
from pickle import dump

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
    

  return pd.Series({'skills_score': skillScore, 'education_level': educationLevel, 'education_score': educationScore, 'company_score': cmpScore})

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

csvPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[1]
skillsPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[2]
companiesPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[3]
universityPath = str(pathlib.Path(__file__).parent.resolve()) + '\\' + sys.argv[4]

df_val = pd.read_csv(csvPath)

df_val.dropna()
df_val['text'] = df_val.text.apply(lambda x: processText(str(x)))

#load skills and companies list
SKILLS = list(filter(None, open(skillsPath).read().lower().split('\n')))
COMPANIES = list(filter(None, open(companiesPath).read().lower().split('\n')))

#universities might have first the city and then the name or vice versa
unis = filter(None, open(universityPath).read().lower().split('\n'))
UNIVERSITIES = []
for uni in unis:
  for permutation in list(itertools.permutations(uni.split(','))):
    UNIVERSITIES.append(re.sub('\s+', ' ', ' '.join(permutation).strip()))
	
	
df_val = df_val.merge(df_val.text.apply(lambda x: scoreCandidate(x)), left_index=True, right_index=True)

#label encoding
label_dict = {'Bad': 1, 'Good': 2, 'Maybe': 3}

df_val['label'] = df_val.label.map(label_dict)


Y = df_val['label']

#drop raw text column and label
X = df_val.drop('label', 1).drop('text', 1)

# Transform data using minmaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
rescaledX = scaler.transform(X)

model = KNeighborsClassifier(algorithm='brute', leaf_size=25, metric='euclidean',
                     n_neighbors=15, p=10)
model.fit(rescaledX,Y)

# save the model
dump(model, open(str(pathlib.Path(__file__).parent.resolve()) + '\\' + 'model.pkl', 'wb'))
# save the scaler
dump(scaler, open(str(pathlib.Path(__file__).parent.resolve()) + '\\' + 'scaler.pkl', 'wb'))

print("Model and scaler saved successfully!")