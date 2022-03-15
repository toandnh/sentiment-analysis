import re

#a dictionary of contractions.
dict = {'isnt': 'is not',
        'arent': 'are not',
        'wasnt': 'was not',
        'werent': 'were not',
        'cant': 'can not',
        'dont': 'do not',
        'didnt': 'did not',
        'wont': 'will not',
        'wouldnt': 'would not',
        'couldnt': 'could not',
        'havent': 'have not',
        'hasnt': 'has not'}

#process function.
def preprocess_text(s):
    #convert to lower case.
    s = s.casefold()
    #remove punctuations and number.
    s = re.sub('[^a-zA-Z]', ' ', s)
    #remove single char.
    s = re.sub(r"\s+[a-zA-Z]\s+", ' ', s)
    #remove multiple spaces.
    s = re.sub(r'\s+', ' ', s)
    #remove space at the end.
    s = s.strip()
    #remove contraction.
    for key, value in dict.items():
        s = re.sub(key, value, s)
    return s

#convert dataframe to list.
def df_to_list(df):
    X = []
    sentences = list(df)
    for s in sentences:
        X.append(preprocess_text(s))
    return X
