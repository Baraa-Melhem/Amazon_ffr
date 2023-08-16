import pandas as pd 

dataaf = pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Amazon frr deep\clean data\dataaftercleanafterlemm.csv")

        
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

# to make sure there is no float in data
for i in range (len(dataaf["cleantext"])):
    dataaf.iloc[i,1]=str(dataaf.iloc[i,1])


x = dataaf["cleantext"]
y = dataaf["label_name"].values  # Convert "Sentiment" to a NumPy array
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
max_sequence_length = 100#to control the length of the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

embedding_dim = 70
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(X_train, y_train, batch_size=50, epochs=10)

mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)



# saving the model
model.save(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Amazon frr deep\deepmodel.h5")

import pickle

file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Amazon frr deep\tokenizer.pickle"
with open(file_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
