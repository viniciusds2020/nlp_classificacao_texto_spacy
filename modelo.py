# Bibliotecas
import spacy
import nltk
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Entrada de dados
data = pd.read_csv(r"UnidadeSeguranca20240126_085952.csv", sep = ';', encoding='latin1')

# Baixando recursos adicionais do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Carregando o modelo da língua portuguesa do spacy
nlp = spacy.load("pt_core_news_sm")

# Carregando os stepswords em português do pacote NLTK
stopwords = nltk.corpus.stopwords.words('portuguese')

# Ajustes e definição das regras do negócio
training_data = data[['Descrição do Evento','Tipo do Risco']]
training_data = training_data[pd.notnull(training_data['Descrição do Evento'])]
training_data.columns = ['text','label']
print(training_data.info())

# Treinar com os dados com especificação
train = training_data[training_data.label != 'Não Especificado']
test = training_data[training_data.label == 'Não Especificado']

# Pré-processamento dos dados
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

preprocessed_training_data = train.apply(lambda x: preprocess_text(x['text']), axis=1)
base = pd.concat([pd.DataFrame(preprocessed_training_data),train['label']],axis=1)
base.columns = ['text','label']
print(base.head(3))

# Divisão do dataset
X_train, X_test, y_train, y_test = train_test_split(base.text, base.label, test_size=0.2, random_state=654)

# Vetorizando os textos
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Balanceamento do dataset
rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train_vectorized, y_train)

# Parâmetros do treinamento
param_grid = {
    'n_estimators': [50, 200, 300],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Busca dos melhores parâmetros via pesquisa aleatória
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_grid, n_iter=5, random_state=5654)
random_search.fit(X_train_rus, y_train_rus)
parametros = random_search.best_params_
print("Parâmetros:", parametros)

# Treinamento com os melhores parâmetros
classifier = RandomForestClassifier(**parametros)
classifier.fit(X_train_rus, y_train_rus)

# Avaliando o modelo
y_pred = classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Exemplo de classificação de texto
example_text = "Em abordagem comportamental deparamos com colaborador transitando na passarela da calha da C-3A sendo que a mesma esta com iluminaçao deficiente risco de tropeço e queda."
preprocessed_example_text = preprocess_text(example_text)
vectorized_example_text = vectorizer.transform([preprocessed_example_text])
predicted_class = classifier.predict(vectorized_example_text)
print(f"Texto: {example_text}\nClasse prevista: {predicted_class[0]}")
