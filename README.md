# msc-ai-MLproject
## Marios Sarrigiannis

The aim of this project was to create an automated/cross-domain resume screening system. The system classifies resumes as Good/Maybe/Bad.

The process is explained in the Report notebook (along with code that was run).

* The system accepts files containing role skills, preferable companies, preferable universities (examples are found in the Corpus folder). 
* 2 datasets are provided in the Datasets folder (including the one used for training).
* The system converts CVs from pdf/docx format to csv in order to be used for training.

Code is available as python scripts in the following directory, as well as the requirements.txt in order to setup your environment.

### Convert Resumes to CSV training data format
* CVs need to exist in a directory according to their label (Good/Maybe/Bad) and this path is provided inside the ResFileToCsv.py
* The script produces a log of all CV files processed as well as errors
* The script produces a single csv file containing CV text and label
NOTE: The script accepts only pdf and docx documents.


### Train model based on your own data
1. Generate a training csv file (using the provided utility)
2. Call model_train.py providing the csv file path and the txt paths (e.g. python3 model_train.py train_data.csv skills.txt companies.txt universities.txt)
3. The process will notify upon completion and produce a scaler and a model file (pretrained files are contained for your convenience)

### Classify a Resume
1. Train a model using the process described above
2. Call demo.py providing the CV file path and the txt paths (e.g. python3 demo.py resume.pdf skills.txt companies.txt universities.txt)
3. Obtain candidate scores and predicted label

