#Modelo SVM
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

#Inicializa o modelo
class ModeloSVM:
    def __init__(self):
        self.dados = None
        self.modelo = None
        self.dados_treino = None
        self.dados_teste = None
        self.respostas_treino = None
        self.respostas_teste = None

    def CarregarDataset(self, path):
        try:
            #Carrega o dataset Iris
            colunas = ['ComprimentoSepala', 'LarguraSepala', 'ComprimentoPetala', 'LarguraPetala', 'Especie']
            self.dados = pd.read_csv(path, names=colunas, header=None)
        except Exception as erro:
            print(f"Erro ao carregar os dados: {erro}")

    def TratamentoDeDados(self):
        #Verifica valores ausentes
        if self.dados.isnull().sum().any():
            self.dados = self.dados.dropna()

        #Codifica a coluna especie
        mapeamento_especies = {especie: indice for indice, especie in enumerate(self.dados['Especie'].unique())}
        self.dados['Especie'] = self.dados['Especie'].map(mapeamento_especies)

    def Treinamento(self):
        #Divide os dados em treino e teste
        atributos = self.dados[['ComprimentoSepala', 'LarguraSepala', 'ComprimentoPetala', 'LarguraPetala']]
        respostas = self.dados['Especie']
        self.dados_treino, self.dados_teste, self.respostas_treino, self.respostas_teste = train_test_split(
            atributos, respostas, test_size=0.2, random_state=42
        )

        #Treinamento com SVM
        self.modelo = SVC()

        #Validação cruzada
        validacao = cross_val_score(self.modelo, self.dados_treino, self.respostas_treino, cv=5)
        print(f"Validação cruzada (média de acurácia) = {validacao.mean():.2f}")

        #Treina o modelo
        self.modelo.fit(self.dados_treino, self.respostas_treino)

    def Teste(self):
        #Avalia o modelo nos dados de teste
        respostas_previstas = self.modelo.predict(self.dados_teste)

        #Cálculo da acurácia
        predicoes_corretas = sum(respostas_previstas == self.respostas_teste)
        total_predicoes = len(self.respostas_teste)
        acuracia = predicoes_corretas / total_predicoes
        print(f"Desempenho do modelo SVM (acurácia) = {acuracia:.2f}")

    #Chamada das funções
    def Train(self, path):
        self.CarregarDataset(path)
        if self.dados is not None:
            self.TratamentoDeDados()
            self.Treinamento()
            self.Teste()

#Chama a função Train
if __name__ == "__main__":
    caminho_dataset = "dados/iris.data"
    modelo = ModeloSVM()
    modelo.Train(caminho_dataset)
