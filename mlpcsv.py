"""
@created_at: 03/06
Esta é a versão que funciona da MLP para o Dataset Sundanese Tweeter Dataset
Utilizando tanto as stopwords quando o CSV fornecido
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv('sundanese_tweets.csv')
# Extrair colunas relevantes
corpus = df['tweet'].tolist()
targets = df['label'].tolist()

with open('stopwords.txt', 'r') as file:
    stopwords = file.read().splitlines()



train_features, test_features, train_targets, test_targets = train_test_split(corpus, targets, test_size=0.1, random_state=123)

vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, norm='l1')

train_features = vectorizer.fit_transform(train_features)
test_features = vectorizer.transform(test_features)

def buildMLPerceptron(train_features, test_features, train_targets, test_targets, num_neurons=5):
    """
    Build de um MLP
    Activation: ReLU
    Optimization Function: SGD, Stochastic Gradient Descent
    Learning rate: Inverse Scalind
    """
    # classifier = MLPClassifier(hidden_layer_sizes=num_neurons, max_iter=100, activation='relu', solver='sgd', verbose=10, random_state=762, learning_rate='invscaling')
    classifier = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.001)
    classifier.fit(train_features, train_targets)

    predictions = classifier.predict(test_features)
    score = np.round(metrics.accuracy_score(test_targets, predictions), 2)
    print("Mean accuracy of predictions: " + str(score))
    print("Salvando o modelo...")

    with open('modelo.pkl', 'wb') as file:
        pickle.dump(classifier, file)


buildMLPerceptron(train_features, test_features, train_targets, test_targets)


print("Carregando o modelo para testar...")

# Carregar o modelo a partir do arquivo
with open('modelo.pkl', 'rb') as file:
    modelo_carregado = pickle.load(file)

novas_frases = [
    "sok geura leungit atuh sia teh corona, matak gelo yeuh aing unggal poe gogoleran",
    "Nu katoel katuhu nu nyerina kenca, goblog wasitna",
    "Bingah pisan patepang sareng pangerasa. Sing katampi kalayan pinuh midulur...",
    "asa hariwang kieu.. lalakon hirup teh asa nyorangan.. asa ieu mah..",
    "Orang mana sih anying, sampis pisan. Bunuh ae lah bunuh"
]


# Realizar a vetorização das novas frases
novas_features = vectorizer.transform(novas_frases)

# Fazer previsões usando o modelo carregado
previsoes = modelo_carregado.predict(novas_features)

# Imprimir as previsões
for frase, previsao in zip(novas_frases, previsoes):
    # print(f"Frase: {frase}\nPrevisão: {previsao}")
    print(f"Previsão: {previsao} | Frase: {frase}")
