from skimage import io
import os
import argparse
import random
import albumentations as A

images_to_generate=500


def augmentation(rgb_path, groundtruth):
    rgb_augmentedPath = 'rgb_augmented'
    masks_augmentedPath = 'masks_augmented'
    # rgb_augmentedPath = os.path.join(rgb_path, rgb_augmented)
    # masks_augmentedPath = os.path.join(groundtruth, masks_augmented)

    if not os.path.exists(rgb_augmentedPath):
        os.makedirs(rgb_augmentedPath)
        print(f'Criada a pasta: {rgb_augmentedPath}')

    if not os.path.exists(masks_augmentedPath):
        os.makedirs(masks_augmentedPath)
        print(f'Criada a pasta: {masks_augmentedPath}')

    images = []
    masks = []

    for img in os.listdir(rgb_path):
        images.append(os.path.join(rgb_path, img))
    
    for mask in os.listdir(groundtruth):
        masks.append(os.path.join(groundtruth, mask))
    
    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1)
    ])

    i = 1

    while i<=images_to_generate:
        number = random.randint(0, len(images)-1)
        image = images[number]
        mask = masks[number]
        print(f'Imagem selecionada: {image}')

        npImg = io.imread(image) # img > numpy array
        npMask = io.imread(mask)

        augmented = aug(image=npImg, mask=npMask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        io.imsave(os.path.join(rgb_augmentedPath, f'{os.path.basename(image)}_{i}_augmented.png'), augmented_image)
        io.imsave(os.path.join(masks_augmentedPath, f'{os.path.basename(mask)}_{i}_augmented.png'), augmented_mask)
        print(f'Imagem e máscara de número {i} sendo gerada.')
        i+=1


def main(rgb_path, groundtruth):
    augmentation(rgb_path, groundtruth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aumento de dados de imagens e máscaras.")
    parser.add_argument('--rgb', type=str, required=True, help='Corresponde ao caminho do diretório que contém as imagens RGB em blocos')
    parser.add_argument('--groundtruth', type=str, required=True, help='Corresponde ao caminho do diretório que contém as imagens RGB em blocos')
    args = parser.parse_args()

    main(args.rgb, args.groundtruth)