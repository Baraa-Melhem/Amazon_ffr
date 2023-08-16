from keras.models import load_model
import pickle
# recall the model
model2 = load_model(r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Amazon frr deep\deepmodel.h5')

file_path = r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Amazon frr deep/tokenizer.pickle'
with open(file_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

from keras.preprocessing.sequence import pad_sequences 

from text_cleaner import TextCleaner# it's a class that I made so I can make it easier to clean text

mytext = input("Describe your experience with the product: ")
cleaner = TextCleaner(mytext)
cleaned_text = cleaner.clean_text()
pos_tagged_text = cleaner.lemmatize_text(cleaned_text)
lemmatized_text = cleaner.lemmatize_with_wordnet(pos_tagged_text)
print("\n","clean text:",lemmatized_text,"\n")



sequences = loaded_tokenizer.texts_to_sequences([lemmatized_text])
max_sequence_length = 20  
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

prediction = model2.predict(padded_sequences)[0][0]

rounded_prediction = round(prediction)

print("Predicted Score:", prediction)


