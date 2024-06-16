import cv2
import numpy as np
import argparse
import os


def binarize_image(img, output_path):
    gr_image = img[:,:,1]  
    _, binary_image = cv2.threshold(gr_image, 80, 255, cv2.THRESH_BINARY)
    # binary_image = cv2.adaptiveThreshold(gr_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary_image = cv2.medianBlur(binary_image, 5)

    # binary_image = cv2.bitwise_not(binary_image) # invertendo
    cv2.imwrite(output_path, binary_image)

def main(input_path, output_path):
    for filename in os.listdir(input_path): 
        # print(filename)
        imginput_path = os.path.join(input_path, filename)
        imgoutput_path = os.path.join(output_path, filename)
        img = cv2.imread(imginput_path, cv2.IMREAD_COLOR)
        binarize_image(img, imgoutput_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dividir imagem em chunks menores.")
    parser.add_argument('--input', type=str, required=True, help='Path para a pasta de input.')
    parser.add_argument('--output', type=str, required=True, help='Path para a pasta de output.')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f'Criado diretório de output: {args.output}')
        
    main(args.input, args.output)
    print('As imagens estão binarizadas.')
