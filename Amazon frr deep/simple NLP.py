import pandas as pd 
import seaborn as sns
from tqdm import tqdm
 
data=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\ML only\amazonfrr\\Reviews.csv")
data.describe(include="all")
corr=data.corr()
corr= sns.heatmap(corr, annot=True)
data.drop("HelpfulnessNumerator",axis=1,inplace=True) 
data.drop("Id",axis=1,inplace=True) 
data.drop("UserId",axis=1,inplace=True) 
data.drop("ProfileName",axis=1,inplace=True) 
data.drop("Summary",axis=1,inplace=True) 
data.dropna(axis=0,inplace=True)
data.isna().sum()
#!pip install neattext
#Data cleaning 
#Data cleaning libs
import nltk 
import neattext as nt
#nltk.download()  # download the necessary dataset , choose : 1.d 2.l 3.all
total_rows = 568454
with tqdm(total=total_rows) as pbar:
    for i in range(len(data["Score"])):
        mytext=data.iloc[i,4]
        docx = nt.TextFrame(text=mytext)
        docx.text 
        data.iloc[i,4]=docx.normalize(level='deep')
        data.iloc[i,4]=docx.remove_puncts()
        data.iloc[i,4]=docx.remove_stopwords()
        data.iloc[i,4]=docx.remove_html_tags()
        data.iloc[i,4]=docx.remove_special_characters()
        data.iloc[i,4]=docx.remove_emojis()
        data.iloc[i,4]=docx.fix_contractions()
        data.iloc[i,4]
        
        pbar.update(1)


print("Data cleaning completed.")


#POS tagging
#it could be :
# [noun
# verb
# adjective
# adverb
# pronoun
# determiner
# conjunction
# preposition
# interjection
# common noun
# proper noun
# mass noun
# count noun]

dataaf=data

nwtxt=[]

import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
newcleantext = []

with tqdm(total=total_rows) as pbar:
    for i in range(len(dataaf["Score"])):
        doc1 = nlp(dataaf.iloc[i, 4])
        postex = []
        
        for token in doc1:
            wordtext = token.text
            poswrd = spacy.explain(token.pos_)

            # Map POS to single characters
            if poswrd == "verb":
                poswrd = "v"
            elif poswrd == "noun":
                poswrd = "n"
            elif poswrd == "adjective":
                poswrd = "a"
            elif poswrd == "adverb":
                poswrd = "r"
            elif poswrd == "pronoun":
                poswrd = "n"
            elif poswrd == "determiner":
                poswrd = "dt"
            elif poswrd == "conjunction":
                poswrd = "cc"
            elif poswrd == "preposition":
                poswrd = "prep"
            elif poswrd == "interjection":
                poswrd = "intj"
            elif poswrd == "common noun":
                poswrd = "n"
            elif poswrd == "proper noun":
                poswrd = "n"
            elif poswrd == "mass noun":
                poswrd = "n"
            elif poswrd == "count noun":
                poswrd = "n"
            else:
                poswrd = "n"

            postex.append(f"({wordtext})({poswrd})")

        newcleantext.append(",".join(postex))
        pbar.update(1)

lemmtext = {"cleantext2": newcleantext}
newtextline = pd.DataFrame(lemmtext)
print("completed..")

dataaf=pd.concat([dataaf,newtextline],axis=1)


####
dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\ML only\amazonfrr\dataafterclean5.csv",index=False)


dataaf=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\ML only\amazonfrr\dataafterclean5.csv")

dataaf=dataaf.drop("Time",axis=1)
dataaf=dataaf.drop("ProductId",axis=1)
dataaf=dataaf.drop("Text",axis=1)


#lemmetizing 
# pos parameter
# "n" for nouns
# "v" for verbs
# "a" for adjectives
# "r" for adverbs
# "s" for satellite adjectives
# Determiner	dt
# Conjunction	cc
# Preposition	prep
# Interjection	intj
# Noun	n
# pos=wordnet.NOUN


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm

newcleantext = []
lemmatizer = WordNetLemmatizer()
dataaf.dropna(axis=0, inplace=True)

with tqdm(total=total_rows-1) as pbar:
    for i in range(len(dataaf["cleantext2"])):
        textsve = ""
        text_in_data = dataaf.iloc[i, 2]
        tokens = [pair.strip("()").split("),") for pair in text_in_data.split("),(")]
        for word_pos in tokens:
            if len(word_pos) == 1:
                # Handle cases where there is only one element in the token
                word, pos = word_pos[0].rstrip(")").rsplit(")(")
            else:
                # Handle cases where both word and POS tag are present
                word, pos = word_pos
                
            if pos == "dt" or pos == "cc" or pos == "prep" or pos == "intj":
                pos = wordnet.NOUN
            
            textsve = lemmatizer.lemmatize(word, pos=pos) +" "+ textsve

        newcleantext.append(textsve)
        pbar.update(1)

lemmtext = {"cleantext": newcleantext}
lemmtext = pd.DataFrame(lemmtext)
print("done")

dataaf=dataaf.drop("cleantext2",axis=1)
dataaf=pd.concat([dataaf,lemmtext],axis=1)



dataaf.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\ML only\amazonfrr\dataaftercleanafterlemm.csv",index=False)



dataaf=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\ML only\amazonfrr\dataaftercleanafterlemm.csv")

dataaf.isna().sum()
dataaf.dropna(axis=0, inplace=True)

