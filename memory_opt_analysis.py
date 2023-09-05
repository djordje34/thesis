import math
import os
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

root = 'GasHisSDB'
resolutions = ['80', '120', '160']
datasets = {res: {'images': [], 'labels': []} for res in resolutions}

#Batchovanje podataka zbog memorije
def process_images_batch(image_paths, batch_size, resolution):
    batch_images = []
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        batch_images.extend(process_images(batch_image_paths, resolution))
    return np.array(batch_images)

def process_images(image_paths, resolution):
    batch_images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (int(resolution), int(resolution)))
        image = image.astype('float32') / 255.0 
        batch_images.append(image)
    return np.array(batch_images)

for res in resolutions:
    root_res = os.path.join(root, res)
    for folder in ['Abnormal', 'Normal']:
        folder_path = os.path.join(root_res, folder)
        label = 0 if folder == 'Abnormal' else 1  # abnormal=0, normal=1

        image_files = os.listdir(folder_path)
        image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
        num_images = len(image_paths)
        batch_size = 100

        for i in range(0, num_images, batch_size):
            batch_image_paths = image_paths[i:i+batch_size]
            batch_images = process_images(batch_image_paths,res)

            datasets[res]['images'].extend(batch_images)
            datasets[res]['labels'].extend([label] * len(batch_images))
            del batch_images
    #slike i labele u np.array sa manjim dtype
    datasets[res]['images'] = np.array(datasets[res]['images'], dtype=np.float16)
    datasets[res]['labels'] = np.array(datasets[res]['labels'], dtype=np.uint8)
    
#del batch_images
print("PART 1")

split_ratio = 0.15  # 70/15/15

for val,res in enumerate(resolutions):
    print(f"PART {val+1}")
    images = datasets[res]['images']
    labels = datasets[res]['labels']

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=split_ratio, stratify=labels, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    datasets[res]['X_train'] = X_train
    datasets[res]['X_val'] = X_val
    datasets[res]['X_test'] = X_test
    datasets[res]['y_train'] = y_train
    datasets[res]['y_val'] = y_val
    datasets[res]['y_test'] = y_test

    del datasets[res]['images']
    del datasets[res]['labels']

np.savez('arrays/80.npz', X_train=datasets['80']['X_train'], X_val=datasets['80']['X_val'], X_test=datasets['80']['X_test'],
         y_train=datasets['80']['y_train'], y_val=datasets['80']['y_val'],y_test=datasets['80']['y_test'])

np.savez('arrays/120.npz', X_train=datasets['120']['X_train'], X_val=datasets['120']['X_val'], X_test=datasets['120']['X_test'],
         y_train=datasets['120']['y_train'], y_val=datasets['120']['y_val'],y_test=datasets['120']['y_test'])

np.savez('arrays/160.npz', X_train=datasets['160']['X_train'], X_val=datasets['160']['X_val'], X_test=datasets['160']['X_test'],
         y_train=datasets['160']['y_train'], y_val=datasets['160']['y_val'],y_test=datasets['160']['y_test'])

