from PIL import Image
import argparse
import os

def divide_image(img, output_path):
    tileW, tileH = 256, 256
    imgW, imgH = img.size

    for top in range(0, imgH, tileH):
        for left in range(0, imgW, tileW):
            right = min(left + tileW, imgW)
            bottom = min(top + tileH, imgH)

            if abs(bottom-top) == tileH and abs(left-right) == tileW:
                box = (left, top, right, bottom)
                tile = img.crop(box)
                tile.save(f"{output_path}/tile_{top}_{left}.png")
        
        else:
            print(f'Imagem não salva. {top}.{left}')


def main(input_path, output_path):
    for filename in os.listdir(input_path): 
        imginput_path = os.path.join(input_path, filename)
        print(imginput_path)
        img = Image.open(imginput_path)
        divide_image(img, output_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dividir imagem em chunks menores.")
    parser.add_argument('--input', type=str, required=True, help='Path para a pasta de input.')
    parser.add_argument('--output', type=str, required=True, help='Path para a pasta de output.') 
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f'Criado diretório de output: {args.output}')

    Image.MAX_IMAGE_PIXELS = None
    main(args.input, args.output)


       
        