import argparse
import os
import tensorflow
import segmentation_models as sm
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

# keras.utils.get_custom_objects().update(custom_objects) > generic.utils error

class DiceCallback(Callback):
    def __init__(self, x_val, y_val):
        super(DiceCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.dice_values = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.x_val)
        dice = dice_coef(self.y_val, preds)
        self.dice_values.append(dice)
        print(f'Dice: {dice}')

def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def train(rgb_path, groundtruth, model_path):
    backbone = 'resnet34'
    preprocess_input = sm.get_preprocessing(backbone) # A biblioteca já tem pré-processamento para a rede escolhida

    train_images = []
    for img in os.listdir(rgb_path):
        imgPath = os.path.join(rgb_path, img)
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        train_images.append(img)
    train_images = np.array(train_images)

    train_masks = []
    for mask in os.listdir(groundtruth):
        maskPath = os.path.join(groundtruth, mask)
        mask = cv2.imread(maskPath, 0)
        mask = mask / 255.0
        train_masks.append(mask)
    train_masks = np.array(train_masks)

    X = train_images
    Y = train_masks
    Y = np.expand_dims(Y, axis=3) # Talvez sejá necessário para utilizar como entrada da rede
    # print(X.shape)
    # print(Y.shape)

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Pré-processamento
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # Modelo
    model = sm.Unet(backbone, encoder_weights='imagenet')
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

    print(model.summary())

    dice_callback = DiceCallback(x_val, y_val)
    history = model.fit(
        x_train,
        y_train,
        batch_size = 8,
        epochs = 10,
        verbose = 1,
        validation_data=(x_val, y_val),
        callbacks = [dice_callback]
    )
  
    model_path = os.path.join(model_path, 'modelVegetationV2.h5')
    model.save(model_path)

    if not os.path.exists('figs'):
        os.makedirs('figs')
        print(f'Criado diretório de output: figs')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training/Validation loss')
    plt.xlabel('Epocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figs/loss_plot.png')
    plt.close()

    dice_values = dice_callback.dice_values
    epochs = range(1, len(dice_values) + 1)
    plt.plot(epochs, dice_values, 'b', label='Dice')
    plt.title('Dice Score por época')
    plt.xlabel('Épocas')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.savefig('figs/dice_plot.png')

def main(rgb_path, groundtruth, model_path):
    train(rgb_path, groundtruth, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento do modelo de segmentação.")
    parser.add_argument('--rgb', type=str, required=True, help='Corresponde ao caminho do diretório que contém as imagens RGB em blocos')
    parser.add_argument('--groundtruth', type=str, required=True, help='Corresponde ao caminho do diretório que contém as imagens RGB em blocos')
    parser.add_argument('--modelpath', type=str, required=True, help='Modelo final salvo.') 
    args = parser.parse_args()

    if not os.path.exists(args.modelpath):
        os.makedirs(args.modelpath)
        print(f'Criado diretório de output: {args.modelpath}')
    
    main(args.rgb, args.groundtruth, args.modelpath)