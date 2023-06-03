"""
@created_at: 03/06
Esta é a versão que funciona da ELM para o Dataset Sundanese Tweeter Dataset
Utilizando tanto as stopwords quando o CSV fornecido
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

class ELM:
    def __init__(self, num_inputs, num_hidden):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.input_weights = np.random.uniform(-1, 1, (num_hidden, num_inputs))
        self.hidden_biases = np.random.uniform(-1, 1, num_hidden)
        self.output_weights = None
    
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        hidden_activations = self._calculate_hidden_activations(X)
        self.output_weights = np.linalg.pinv(hidden_activations) @ y
        
    def predict(self, X):
        X = np.array(X)
        hidden_activations = self._calculate_hidden_activations(X)
        y_pred = hidden_activations @ self.output_weights
        return y_pred
    
    def _calculate_hidden_activations(self, X):
        total_samples = X.shape[0]
        hidden_activations = np.zeros((total_samples, self.num_hidden))
        for i in range(total_samples):
            hidden_activations[i] = np.tanh(
                np.dot(self.input_weights, X[i]) + self.hidden_biases
            )
        return hidden_activations

def buildELM(train_features, test_features, train_targets, test_targets, label_encoder, num_neurons=200):
    """
    Build de uma ELM
    Activation: tanh
    """
    # Codificar os rótulos usando LabelEncoder
    train_targets_encoded = label_encoder.fit_transform(train_targets)
    
    # Converter os rótulos codificados para one-hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    train_targets_onehot = onehot_encoder.fit_transform(train_targets_encoded.reshape(-1, 1))

    elm = ELM(train_features.shape[1], num_neurons)
    elm.train(train_features, train_targets_onehot)

    # Codificar os rótulos de teste usando LabelEncoder
    test_targets_encoded = label_encoder.transform(test_targets)
    # Converter os rótulos codificados de teste para one-hot encoding
    test_targets_onehot = onehot_encoder.transform(test_targets_encoded.reshape(-1, 1))

    predictions = elm.predict(test_features)
    # Converter as previsões para as classes originais
    predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    score = np.round(metrics.accuracy_score(test_targets, predicted_classes), 2)
    print("Mean accuracy of predictions: " + str(score))
    print("Salvando o modelo...")

    with open('modelo_elm.pkl', 'wb') as file:
        pickle.dump(elm, file)


train_features = train_features.toarray()
test_features = test_features.toarray()

label_encoder = LabelEncoder()
label_encoder.fit(train_targets)

buildELM(train_features, test_features, train_targets, test_targets, label_encoder)


# Testando com novas frases

# Carregue o modelo treinado
with open('modelo_elm.pkl', 'rb') as file:
    elm = pickle.load(file)

new_phrases = [
    "sok geura leungit atuh sia teh corona, matak gelo yeuh aing unggal poe gogoleran",
    "Nu katoel katuhu nu nyerina kenca, goblog wasitna",
    "Bingah pisan patepang sareng pangerasa. Sing katampi kalayan pinuh midulur...",
    "asa hariwang kieu.. lalakon hirup teh asa nyorangan.. asa ieu mah..",
    "Orang mana sih anying, sampis pisan. Bunuh ae lah bunuh"
]
# correct_predictions = ['anger', 'anger', 'joy', 'fear', 'anger']
# predictions = ['anger', 'anger', 'sadness', 'joy', 'anger']

new_features = vectorizer.transform(new_phrases).toarray()


predictions = elm.predict(new_features)

predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

for phrase, predicted_class in zip(new_phrases, predicted_classes):
    print(f"Previsão: {predicted_class} | Frase: {phrase}")
    # print("Frase:", phrase)
    # print("Previsão:", predicted_class)
    # print()

