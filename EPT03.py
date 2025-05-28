#======== informacoes =========#
# Versao 03
# Dataset externo
# Acuracia média: 45%
# python


from dataset import emotional_dataset  # Importa o dataset
# torch para fazer operações com matrizes complexas para AI
import torch
# biblioteca que ajuda a tokenizar as frases, tirando espaços etc
import re
# funcao utilitaria da lib torchtext que transforma tokens em vocabularioé esse dicionario que a AI usa, e não as palavras
from torchtext.vocab import build_vocab_from_iterator
# funcao pra uniformizar o tamanho dos tensores
from torch.nn.utils.rnn import pad_sequence
# funcao que separa os dados antes de serem digeridos pela AI
from sklearn.model_selection import train_test_split
# funcao para separar os dados para funcionar melhor em gpu
from torch.utils.data import DataLoader, TensorDataset
# redes neurais do torch
import torch.nn as nn

#################  tokenizacao  ########################

frases = [texto for texto, _ in emotional_dataset] # cria uma lista com os varios textos
labels = [label for _, label in emotional_dataset] # cria uma lista com as varias labels

# uma funcao para quebrar lista em tokens
def tokenizador(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    return text.split()

# depois só preciso botar meus dados dentro do tokenizador, FRASE PRO FRASE
tokens_frases = [tokenizador(frase) for frase in frases]

# yield function (funcao iteradora) que percorre a lista labels, pega cada uma das palavras, e cria uma lista de listas, onde cada palavra vai ser uma sub-lista
def iteradora_labels(labels_list):
    for label in labels_list:
        yield [label]  # coloca cada label como uma lista com um único elemento


# o vocab é uma ferramenta que pega uma lista de lista(matriz) e quebra em apenas listas
# depois vai pegar isso e tornar em um vocabulario,
# que tem os tokens de um lado e um indice do outro
vocabulario_frases = build_vocab_from_iterator(tokens_frases)
vocabulario_labels = build_vocab_from_iterator(iteradora_labels(labels))

# mas ainda não é o suficiente, precisamos tornar os vocabulários em indíces
indice_frases = [vocabulario_frases(tokens) for tokens in tokens_frases]
indice_labels = [vocabulario_labels([label]) for label in labels]

# Converte listas de índices do python para tensores do pytorch
# são exatamente iguais as listas de indices do python só que rodam em gpu
tensor_frases = [torch.tensor(seq, dtype=torch.long) for seq in indice_frases]
tensor_labels = torch.tensor(indice_labels, dtype=torch.long).squeeze()  # Remove dimensão extra

# adicionamos padding para deixar os tensores todos com o mesmo tamanho
frases_padded = pad_sequence(
    tensor_frases, 
    batch_first=True,  # Formato [batch, seq_len]
    padding_value=0    # Preenche com 0 onde falta
)

##################### separacao dos dados ########################

# para fazer um bom treino, precisamos separar os dados
# 80% de dados que a AI sabe as labels e usa para estudar
# 20% que não sabe e vai ser usado de "prova" pra ver se aprendeu mesmo
X_train, X_test, y_train, y_test = train_test_split(
    frases_padded,
    tensor_labels,
    test_size=0.2,      # 20% para teste
    random_state=42     # Semente para reprodutibilidade
)

# empacotamos esses dados em batches, que tornar mais eficiente para relizar os testes em gpu
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# usamos dataloaders, empacotadores de dados inteligentes
batch_size = 2  # Tamanho do lote (ajuste conforme sua GPU)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Embaralha os dados de treino
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # Teste não precisa ser embaralhado


###################### execucao ################################

# seta a arquitetura do modelo
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]  # Último vetor da sequência
        return self.fc(last_hidden)


# Hiperparâmetros atualizados
vocab_size = len(vocabulario_frases)
embedding_dim = 260       # Menor embedding
hidden_dim = 200          # LSTM menor
output_dim = len(vocabulario_labels)
learning_rate = 0.001    # Um pouco mais alto agora
batch_size = 4
num_epochs = 10          # Menos épocas para evitar overfitting

model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento
best_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    # Validação
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Época {epoch+1}: Loss: {avg_loss:.4f} | Acurácia: {accuracy:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pt')

# Avaliação final
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=vocabulario_labels.get_itos()))


#============================ interacao =============================#

print("\nDigite frases para testar (ou 'sair' para encerrar):")

while True:
    frase = input("\n> ")
    if frase.lower() == 'sair':
        break
    
    # Tokeniza e converte para índices
    tokens = tokenizador(frase)
    indices = [vocabulario_frases[token] for token in tokens if token in vocabulario_frases]
    
    if not indices:
        print("Não entendi a frase.")
        continue
    
    # Converte para tensor e faz previsão
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
    
    # Mostra resultado simples
    print("→", vocabulario_labels.lookup_token(predicted.item()))