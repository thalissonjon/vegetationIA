import argparse
import os
import cv2
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

def inference(img, model_path, output_path):
    model = keras.models.load_model(model_path)
    predict = model.predict(img)

    predict = np.squeeze(predict, axis=0) # remover a primeira dimensão

    predict = cv2.normalize(predict, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # normaliza a predição para 0-255
    predict = predict.astype(np.uint8) # converter para imagem
    
    # plt.imshow(predict, cmap='gray')
    # plt.title('Previsão')
    # plt.show()

    cv2.imwrite(output_path, predict)
    

def main(rgb_path, model_path, output_path):
    for filename in os.listdir(rgb_path):
        print(filename) 
        imginput_path = os.path.join(rgb_path, filename)
        imgoutput_path = os.path.join(output_path, filename)

        img = cv2.imread(imginput_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0)

        inference(img, model_path, imgoutput_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inferência do modelo.")
    parser.add_argument('--rgb', type=str, required=True, help='Corresponde ao caminho do diretório que contém as imagens RGB em blocos')
    parser.add_argument('--modelpath', type=str, required=True, help='Modelo salvo.') 
    parser.add_argument('--output', type=str, required=True, help='Path para a pasta de output.') 
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f'Criado diretório de output: {args.output}')
        
    main(args.rgb, args.modelpath, args.output)