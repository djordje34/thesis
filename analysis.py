import os
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

print(f"Радни директоријум: {os.getcwd()}")    
print("Путање до фајлова и датотека у оквиру радне датотеке:",*[f"*{file}*" for file in os.listdir(os.getcwd())\
                                                                if not file.endswith('.png')],\
                                                                  sep='\n')

types = ["Normal","Abnormal"]
folders = ["80","120","160"]
images = []
paths = [f"{folder}/{image_type}" for folder, image_type in product(folders, types)]
print("Путање до датотека које садрже скупове података:",*[f"GasHisSDB/{p}/*назив неке слике*" for p in paths], sep='\n')

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for j in range(axes.shape[1]): 
    for i in range(axes.shape[0]): 
        index = j * axes.shape[0] + i  
        if index >= len(paths):  
            break
        print(f"GasHisSDB/{paths[index]}/{paths[index].split('/')[1]}-00001.png")
        image = Image.open(f"GasHisSDB/{paths[index]}/{paths[index].split('/')[1]}-00001.png")
        ax = axes[i, j] 
        ax.imshow(image)
        ax.set_title(f"{paths[index]}/{paths[index].split('/')[1]}-00001.png")
        width, height = image.size
        ax.set_xlabel(f"Ширина: {width}px")
        ax.set_ylabel(f"Висина: {height}px")
plt.tight_layout()
plt.show()

el_count = {}

for path in paths:
    files = os.listdir(f"GasHisSDB/{path}")
    file_count = len(files)
    el_count[path] = file_count

folder_names = list(el_count.keys())
counts = list(el_count.values())

bar_width = 0.5
bar_padding =2.0

bar_positions = range(len(folder_names))

plt.bar(bar_positions, counts, width=bar_width, align='center', edgecolor='black')

plt.xticks(bar_positions, folder_names)
plt.tick_params(axis='x', labelsize=8)
plt.margins(0.0)
plt.subplots_adjust(bottom=0.2)
plt.xlabel('Датотеке - подскупови')
plt.ylabel('Број слика - података')
plt.title('Број слика у сваком подскупу')
plt.show()

#или на други начин

res = [str(x) for x in sorted([int(x) for x in list(set([x.split("/")[0] for x in paths]))])]
tp = list(set([x.split("/")[1] for x in paths]))
plt.bar(res,[x for ind, x in enumerate(counts) if ind%2])
plt.bar(res,[x for ind, x in enumerate(counts) if not ind%2], bottom=[x for ind, x in enumerate(counts) if ind%2])
plt.legend(tp)

plt.xlabel('Подскупови груписани по резолуцији')
plt.ylabel('Број слика - података')
plt.title('Број слика у сваком груписаном подскупу')

plt.show()

#рефактористање листе која представља фодлере и подфолдере
root = 'GasHisSDB'
resolutions = ['80','120','160']
datasets = {res: {'images': [], 'labels': []} for res in resolutions}
for res in resolutions:
    rootres = os.path.join(root, res)
    for folder in ['Abnormal', 'Normal']:
        folder_path = os.path.join(rootres, folder)
        label = 0 if folder == 'Abnormal' else 1  #abnormal=0, normal=1
        
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            
            image = cv2.imread(image_path)
            image = cv2.resize(image, (int(res), int(res)))
            
            datasets[res]['images'].append(image)
            datasets[res]['labels'].append(label)
            
split_ratio = 0.15  # 70/15/15
for res in resolutions:
    images = np.array(datasets[res]['images'])
    labels = np.array(datasets[res]['labels'])
    
    # сплит та тренинг и (тест + валдиационе) скупове
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=split_ratio, stratify=labels, random_state=42)
    
    # валдиација и тест 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Assign the split datasets to the respective resolution
    datasets[res]['X_train'] = X_train
    datasets[res]['X_val'] = X_val
    datasets[res]['X_test'] = X_test
    datasets[res]['y_train'] = y_train
    datasets[res]['y_val'] = y_val
    datasets[res]['y_test'] = y_test
    
    
    
subset_labels = ['train', 'val', 'test']
srb_labels = ['тренинг','валидациони','тестни']
# Create a grid layout for the subplots
fig, axes = plt.subplots(nrows=len(resolutions), ncols=len(subset_labels), figsize=(12, 9))

# Iterate over each dataset and its subsets to plot the bar plots
for i, res in enumerate(resolutions):
    for j, subset_label in enumerate(subset_labels):
        # Count the number of normal and abnormal images in the subset
        subset_images = datasets[res]['X_'+subset_label]
        subset_labels_data = datasets[res]['y_'+subset_label]
        
        num_normal = sum(subset_labels_data == 1)
        num_abnormal = sum(subset_labels_data == 0)
        
        # Plot the bar plot for the current subset
        axes[i, j].bar(['Нормално ткиво', 'Абнормално ткиво'], [num_normal, num_abnormal])
        axes[i, j].set_title(f'Скуп података {res}x{res} слика \n *{srb_labels[j]}* подскуп')
        axes[i, j].set_ylabel('Број примерака')

        print(res, subset_label)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

#стандардни приступ захтева 16.8 гигабајта РАМ-а за нормализацију тренинг сета, тако да мора по бечевима...
batch_size = 300
for res in resolutions:
    total_images = datasets[res]['X_train'].shape[0]
    num_batches = int(np.ceil(total_images / batch_size))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        datasets[res]['X_train'][start_idx:end_idx] = datasets[res]['X_train'][start_idx:end_idx] / 255.0
    
    #datasets[res]['X_val'] = datasets[res]['X_val'] / 255.0
    #datasets[res]['X_test'] = datasets[res]['X_test'] / 255.0
    
    total_images = datasets[res]['X_val'].shape[0]
    num_batches = int(np.ceil(total_images / batch_size))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        datasets[res]['X_val'][start_idx:end_idx] = datasets[res]['X_val'][start_idx:end_idx] / 255.0
        
        
    total_images = datasets[res]['X_test'].shape[0]
    num_batches = int(np.ceil(total_images / batch_size))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        datasets[res]['X_test'][start_idx:end_idx] = datasets[res]['X_test'][start_idx:end_idx] / 255.0
        

print(datasets['80']['X_train'])