#Modelo Regressão Linear
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

    #Inicializa o modelo
class RegressaoLinearModelo:
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
            print("Dataset carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")

    def TratamentoDeDados(self):
        #Verifica valores ausentes
        if self.df.isnull().sum().any():
            print("Dados com valores ausentes encontrados. Removendo linhas com valores ausentes...")
            self.df = self.df.dropna()

        #Codifica a coluna Species
        species_mapping = {species: idx for idx, species in enumerate(self.df['Species'].unique())}
        self.df['Species'] = self.df['Species'].map(species_mapping)
        print("Dados tratados com sucesso!")

    def Treinamento(self):
        #Divide os dados em treino e teste
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Treinamento com Regressão Linear
        self.model = LinearRegression()

        #Validação cruzada
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='r2')
        print(f"Validação cruzada (média de acurácia) = {scores.mean():.2f}")

        #Treina o modelo
        self.model.fit(self.X_train, self.y_train)
        print("Modelo treinado com sucesso!")

    def Teste(self):
        #Avalia o modelo nos dados de teste
        r2_score = self.model.score(self.X_test, self.y_test)
        print(f"Desempenho da regressão linear (acurácia) = {r2_score:.2f}")

    #Chama as funções
    def Train(self, path):
        self.CarregarDataset(path)
        if self.df is not None:
            self.TratamentoDeDados()
            self.Treinamento()
            self.Teste()

    #Chama a função Train
if __name__ == "__main__":
    caminho_dataset = "dados/iris.data" 
    modelo = RegressaoLinearModelo()
    modelo.Train(caminho_dataset)
