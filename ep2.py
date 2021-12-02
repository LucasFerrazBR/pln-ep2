import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer    
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding, Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

#Constroi uma rede neural CNN utilizando a biblioteca Keras, incorpora o embedding criado com Word2Vec na rede.
def constroi_modelo_cnn():
    modelo = Sequential()
    
    modelo.add(embedding_layer)
    modelo.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    modelo.add(MaxPooling1D(pool_size=2))
    #Dropouts reduzem o overfitting da rede ao desativar neurônios durante o treinamento
    modelo.add(Dropout(0.2))
    modelo.add(Flatten())
    modelo.add(Dense(1, activation='sigmoid'))

    modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return modelo

#Lendo os arquivos e retirando as stopwords com pandas.
df_cloroquina = pd.read_excel("ep2-cloroquina-treino.xlsx",sheet_name='train',usecols=['texto','posicao'])
x = df_cloroquina.texto
y = df_cloroquina.posicao

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#Transformando os dados textuais em sequências númericas.
tokenizer = Tokenizer(num_words=11000)
tokenizer.fit_on_texts(x)

x_train_numerico = tokenizer.texts_to_sequences(x_train)
x_test_numerico = tokenizer.texts_to_sequences(x_test)
tamanho_vocabulario = len(tokenizer.word_index) + 1

#Transformando os dados de treino e teste em entradas para a rede neural através de padding.
tamanho_embedding = 300

x_train_padded = pad_sequences(x_train_numerico, padding='post', maxlen=tamanho_embedding)
x_test_padded = pad_sequences(x_test_numerico, padding='post', maxlen=tamanho_embedding)

#Criando a lista de palavras (token) de cada linha de dado do arquivo de dados, o TfidfVectorizer está sendo usado como filtro de palavras.
tfidf = TfidfVectorizer(ngram_range=[1,1], analyzer='word', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')

analyzer = tfidf.build_analyzer()
lista_tokens = []

for item in x:
    tokens = analyzer(item)
    lista_tokens.append(tokens)

#Criando embeddings com vetores de tamanho 300 com a lista de tokens criada anteriormente. Logo após, utilizando a matriz de pesos do modelo Word2Vec para criar uma camada de embeddings para a rede neural.
word2vec = Word2Vec(lista_tokens, vector_size=tamanho_embedding, workers=8, min_count=1)
num_palavras = word2vec.wv.vectors.shape[0]
embedding_layer = Embedding(num_palavras, tamanho_embedding, weights=[word2vec.wv.vectors], input_length=tamanho_embedding, trainable=True)

#Cria uma função de parada antecipada para evitar aumento da função loss da rede.
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')

#Transforma o modelo neural do Keras em um modelo que pode ser entendido pelo Sklearn (wrapper) e inicializa o treinamento e validações.
rede_neural = KerasClassifier(build_fn=constroi_modelo_cnn, epochs=8)
kfolds = StratifiedKFold(n_splits=10, shuffle=True)
resultado = cross_val_score(rede_neural, x_train_padded, y_train, cv=kfolds, scoring='accuracy', fit_params={'callbacks': [early_stopping]}).mean()
print("Acuracia de treino:" , resultado)

#Testa a acuracia da rede com os dados de teste.
rede_neural.fit(x_test_padded, y_test)
predicao = rede_neural.predict(x_test_padded)
acuracia = accuracy_score(predicao, y_test)
print("Acuracia de teste:", acuracia)

