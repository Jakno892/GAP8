import os

# Code to rename image files

IMAGE_PATH = "classification/training_data/validation/"

img_folder = f"{IMAGE_PATH}class_1/"
num_images = len(os.listdir(img_folder))-99
for i in range(1,num_images+1):
    os.rename(os.path.join(img_folder,f'img_{1100+i:06d}.png'), os.path.join(img_folder,f'img_{i:06d}.png'))

img_folder = f"{IMAGE_PATH}class_2/"
num_images = len(os.listdir(img_folder))
for i in range(1,num_images+1):
    os.rename(os.path.join(img_folder,f'img_{400+i:06d}.png'), os.path.join(img_folder,f'img_{i:06d}.png'))
