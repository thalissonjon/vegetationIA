# Modelo de rede neural artificial 
Este projeto foi realizado para segmentar imagens para detecção de vegetação.
Foi implementado funções de aumento de dados, divisão de imagens, treinamento de rede, binarização de imagens e inferência do modelo.

### Pré-requisitos

```
pip install -r requirements.txt
```

### augmentation.py
Essa função realiza o aumento de dados em imagens e máscaras para tarefas de segmentação de imagem. Ele aplica transformações de aumento, como rotação aleatória, espelhamento horizontal e vertical, e transposição, para gerar novas variações das imagens e suas respectivas máscaras.

```
python augmentation.py --rgb caminho/para/imagens_rgb --groundtruth caminho/para/mascaras
```
### binarize_images.py
Script para binarizar imagens RGB, convertendo-as em imagens binárias em preto e branco.

```
python binarize_images.py --input caminho/para/imagens --output caminho/para/saida_binarizada
```

### divide_orthomosaic.py
Script que divide uma imagem em chunks menores e os salva como arquivos separados.

```
python divide_orthomosaic.py --input caminho/para/imagens --output caminho/para/saida_chunks
```
### train_model.py
Realiza o treinamento de um modelo de segmentação usando a rede UNet com backbone ResNet34.

```
python train_model.py --rgb caminho/das/imagens --groundtruth caminho/das/mascaras --modelpath caminho/do/modelo_gerado
```

### inference.py
Realiza a inferência de um modelo de segmentação em um conjunto de imagens RGB, utilizando um modelo previamente treinado (arquivo .h5), e salva as previsões resultantes.

```
python inference.py --rgb caminho/das/imagens --modelpath caminho/do/modelo --output caminho/de/saida
```

