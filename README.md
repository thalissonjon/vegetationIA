# Artificial Neural Network Model for Vegetation Detection 
This project was created to segment images for vegetation detection. 
It includes functions for data augmentation, image splitting, network training, image binarization, and model inference.  

### Prerequisites  

```
pip install -r requirements.txt
```

### augmentation.py  
This function performs data augmentation on images and masks for image segmentation tasks. It applies augmentation transformations such as random rotation, horizontal and vertical flipping, and transposition to generate new variations of the images and their respective masks.  

```
python augmentation.py --rgb path/to/rgb_images --groundtruth path/to/masks
```

### binarize_images.py  
Script to binarize RGB images, converting them into black-and-white binary images.  

```
python binarize_images.py --input path/to/images --output path/to/binarized_output
```

### divide_orthomosaic.py  
Script to divide an image into smaller chunks and save them as separate files.  

```
python divide_orthomosaic.py --input path/to/images --output path/to/chunk_output
```

### train_model.py  
Trains a segmentation model using the UNet network with a ResNet34 backbone.  

```
python train_model.py --rgb path/to/images --groundtruth path/to/masks --modelpath path/to/generated_model
```

### inference.py  
Performs inference using a pre-trained segmentation model (`.h5` file) on a set of RGB images and saves the resulting predictions.  

```
python inference.py --rgb path/to/images --modelpath path/to/model --output path/to/output
```

