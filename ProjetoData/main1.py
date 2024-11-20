#Modelo SVM
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

    #Inicializa o modelo
class SVMModelo:
    def __init__(self):
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def CarregarDataset(self, path):
        try:
            #Carrega o dataset Iris
            names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
            self.df = pd.read_csv(path, names=names, header=None)
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")

    def TratamentoDeDados(self):
        #Verifica valores ausentes
        if self.df.isnull().sum().any():
            self.df = self.df.dropna()

        #Codifica a coluna Species
        species_mapping = {species: idx for idx, species in enumerate(self.df['Species'].unique())}
        self.df['Species'] = self.df['Species'].map(species_mapping)

    def Treinamento(self):
        #Divide os dados em treino e teste
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Treinamento com SVM
        self.model = SVC()

        #Validação cruzada
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"Validação cruzada (Média de acurácia) = {scores.mean():.2f}")

        #Treina o modelo
        self.model.fit(self.X_train, self.y_train)

    def Teste(self):
        #Avalia o modelo nos dados de teste
        y_pred = self.model.predict(self.X_test)

        #Cálculo da acurácia
        correct_predictions = sum(y_pred == self.y_test)
        total_predictions = len(self.y_test)
        acc = correct_predictions / total_predictions
        print(f"Desempenho do modelo SVM (acurácia) = {acc:.2f}")

    #Chama as funções
    def Train(self, path):
        self.CarregarDataset(path)
        if self.df is not None:
            self.TratamentoDeDados()
            self.Treinamento()
            self.Teste()

#Chamar a função Train
if __name__ == "__main__":
    caminho_dataset = "dados/iris.data" 
    modelo = SVMModelo()
    modelo.Train(caminho_dataset)
