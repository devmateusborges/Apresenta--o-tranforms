# GPT2Tokenizer: Essa classe é responsável pela tokenização dos textos para serem compatíveis com o modelo GPT-2.
# A tokenização é o processo de dividir o texto em unidades menores chamadas de "tokens". O GPT2Tokenizer é
# projetado especificamente para o modelo GPT-2 e converte o texto em uma sequência de tokens numéricos que o
# modelo pode entender. Ele também lida com detalhes específicos do modelo, como lidar com caracteres especiais,
# símbolos e subpalavras.

# GPT2LMHeadModel: Essa classe implementa a arquitetura do modelo GPT-2 para tarefas de geração de linguagem.
# O GPT2LMHeadModel é pré-treinado em um grande conjunto de dados não rotulados para aprender a estrutura e o
# significado das sentenças em um determinado idioma. Ele é capaz de gerar texto condicionalmente com base em um
# texto de entrada fornecido. O modelo é chamado de "LM" (Language Model) "Head" porque possui uma camada de cabeça
# (head) especializada para tarefas de modelagem de linguagem.

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carregar o modelo pré-treinado e o tokenizador
model_name = 'gpt2'  # Nome do modelo pré-treinado
tokenizer = GPT2Tokenizer.from_pretrained(
    model_name)  # Inicializar o tokenizador do GPT2
model = GPT2LMHeadModel.from_pretrained(
    model_name)  # Inicializar o modelo GPT2

print("==================================")
print("===     Iniciando Treinamento  ===")
print("==================================")
# Dados de treinamento
text = "Tente mover o mundo - o primeiro passo será mover a si mesmo."  # Platão
print("Texto original:", text)
# Codificar o texto como tokens numéricos usando o tokenizador
inputs = tokenizer.encode(text, return_tensors='pt')
print("==================================")
print("===             Entrada        ===")
print("==================================")
print("Tokens de entrada:", inputs)

# Treinamento
# Inicializar o otimizador Adam para atualizar os parâmetros do modelo
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Definir os rótulos como as entradas (gerar a próxima palavra no texto)
labels = inputs.clone()

model.train()  # Definir o modo de treinamento para o modelo
print("==================================")
print("===             èpocas         ===")
print("==================================")
for epoch in range(10):  # Número de épocas de treinamento
    optimizer.zero_grad()  # Zerar os gradientes acumulados do otimizador
    # Passar as entradas e rótulos para o modelo
    outputs = model(inputs, labels=labels)
    loss = outputs.loss  # Obter a perda do modelo
    print("ÈPoca:", epoch+1, "PERDA:", loss.item())
    loss.backward()  # Realizar a retropropagação para calcular os gradientes
    optimizer.step()  # Atualizar os parâmetros do modelo com base nos gradientes calculados


print("==================================")
print("===     USO DO TEINAMENTO      ===")
print("==================================")

# Uso do modelo treinado

text = "Tente mover o mundo -"
print("==================================")
print("===          ENTRADA           ===")
print("==================================")
print("Frase de entrada:", text)
# Codificar a frase de entrada como tokens numéricos
input_ids = tokenizer.encode(text, return_tensors='pt')
print("==================================")
print("===     TOKENS DE ENTRADA      ===")
print("==================================")
print("Tokens de entrada:", input_ids)
# Criar a máscara de atenção
attention_mask = torch.ones_like(input_ids)
# Gerar texto com base na frase de entrada e na máscara de atenção
output = model.generate(input_ids, max_length=50,
                        attention_mask=attention_mask)
# Decodificar os tokens gerados em texto
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("==================================")
print("===            saida           ===")
print("==================================")
print("Texto gerado:", generated_text)
