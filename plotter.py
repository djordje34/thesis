import math
import os
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model

class Plotter:
    @staticmethod
    def loadSubset(size=40):
        data = np.load('data_arrays.npz')
        X_train = data['X_train']
        y_train = data['y_train']

        #print(y_train[:size])
        return X_train[:size],y_train[:size]
    
    
    @staticmethod
    def processColors(*args, **kwargs):
        subset_images = []
        subset_images.append(Plotter.loadSubset()[0][35]) #абн
        subset_images.append(Plotter.loadSubset()[0][5]) #н
        for img in subset_images:
            red_channel = img[:, :, 0].ravel().astype('float64')
            green_channel = img[:, :, 1].ravel().astype('float64')
            blue_channel = img[:, :, 2].ravel().astype('float64') #проблем око варијансе и флоата - потребно фл64
            print(red_channel,green_channel,blue_channel)
                
            plt.figure(figsize=(10, 6))
            # црвена
            plt.subplot(1, 3, 1)
            sns.kdeplot(red_channel, color='red', fill=True)
            plt.title('Црвени канал')
            plt.xlabel('Вредност пиксела')
            plt.ylabel('Густина')
            # зелена
            plt.subplot(1, 3, 2)
            sns.kdeplot(green_channel, color='green', fill=True)
            plt.title('Зелени канал')
            plt.xlabel('Вредност пиксела')
            plt.ylabel('Густина')
            # плава
            plt.subplot(1, 3, 3)
            sns.kdeplot(blue_channel, color='blue', fill=True)
            plt.title('Плави канал')
            plt.xlabel('Вредност пиксела')
            plt.ylabel('Густина')
            plt.tight_layout()
            plt.show()
            
    @staticmethod
    def processContours(*args, **kwargs):
        images = []
        images.append(Plotter.loadSubset()[0][35])
        images.append(Plotter.loadSubset()[0][5])
        for image in images:
            gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)  # претворена у нијансе беле-црне
            
            #binary_image = cv2.adaptiveThreshold(gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 137, 60) #најбоље ради
            
            normalized_gray_image = gray_image.astype('float32') / 255.0
            threshold_value = 0.6
            binary_image = (normalized_gray_image > threshold_value).astype(np.uint8)
            
            original_image = (image * 255).astype(np.uint8)
            binary_image = (binary_image * 255).astype(np.uint8)

            # Контуре
            contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # 1 3
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title('Оригинална слика')

            # Прикажи бин. слику и контуре
            plt.subplot(1, 2, 2)
            plt.imshow(binary_image, cmap='gray') 
            for contour in contours:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
                plt.xlim(2,158)
                plt.ylim(158,2)
            plt.title('Детектоване ивице и контуре')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def processGrayscale(*args, **kwargs):
        images = []
        gray_images = []
        images.append(Plotter.loadSubset()[0][35])
        images.append(Plotter.loadSubset()[0][5])
        for image in images:
            gray_images.append(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
        
        plt.figure(figsize=(10, 5))
        for i,image in enumerate(images):
            plt.subplot(2, 2, i+1)
            plt.imshow((image * 255).astype(np.uint8))
            plt.title(f"Слика {'абнормалног ткива ' if i==0 else 'нормалног ткива '}у RGB формату")

            plt.subplot(2, 2, i+3)
            plt.imshow(gray_images[i], cmap='gray') 
            plt.title(f"Слика {'абнормалног ткива ' if i==0 else 'нормалног ткива '}у Grayscale формату")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def GLCM(*args, **kwargs):
        #Load grayscale
        images = []
        images.append(Plotter.loadSubset()[0][35])
        images.append(Plotter.loadSubset()[0][5])
        for image in images:
            gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            #GLCM
            distance = 1
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray_image, [distance], angles=angles, symmetric=True, normed=True)

            #GLCM params
            contrast = graycoprops(glcm, 'contrast')
            energy = graycoprops(glcm, 'energy')
            homogeneity = graycoprops(glcm, 'homogeneity')
            print(contrast,energy,homogeneity,sep=" | ",end="\n")
            #Plot glcm param vals
            plt.figure(figsize=(10, 6))
            plt.grid(True)
            #Kontrast
            plt.subplot(1, 3, 1)
            plt.bar(range(len(contrast[0])), contrast[0])
            plt.xticks(range(len(contrast[0])), ['0°', '45°', '90°', '135°'])
            plt.title('Контраст')
            plt.xlabel('Угао')
            plt.ylabel('Вредност')

            #Energija
            plt.subplot(1, 3, 2)
            plt.bar(range(len(energy[0])), energy[0])
            plt.xticks(range(len(energy[0])), ['0°', '45°', '90°', '135°'])
            plt.title('Енергија')
            plt.xlabel('Угао')
            plt.ylabel('Вредност')

            #Homogenost
            plt.subplot(1, 3, 3)
            plt.bar(range(len(homogeneity[0])), homogeneity[0])
            plt.xticks(range(len(homogeneity[0])), ['0°', '45°', '90°', '135°'])
            plt.title('Хомогеност')
            plt.xlabel('Угао')
            plt.ylabel('Вредност')

            plt.tight_layout()
            plt.show()

        
        
    @staticmethod
    def LBP(*args, **kwargs):
        
        def compute_lbp(image, radius=2, n_points=8):
            lbp_image = np.zeros_like(image)

            for y in range(radius, image.shape[0] - radius):
                for x in range(radius, image.shape[1] - radius):
                    center_pixel = image[y, x]
                    lbp_code = 0
                    for i in range(n_points):
                        angle = 2 * np.pi * i / n_points
                        x_i = x + int(radius * np.cos(angle))
                        y_i = y - int(radius * np.sin(angle))
                        lbp_code |= (image[y_i, x_i] >= center_pixel) << i
                    lbp_image[y, x] = lbp_code
            return lbp_image

        images = []
        images.append(Plotter.loadSubset()[0][35])
        images.append(Plotter.loadSubset()[0][5])
        lbp_images = [compute_lbp(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)) for image in images]

        for i in range(len(images)):
            rev_img = (images[i] * 255).astype(np.uint8)
            plt.subplot(2, 2, 2*i+1)
            plt.imshow(rev_img, cmap='gray')
            plt.title('Оригинална слика')
            
            rev_lbp = (lbp_images[i] * 255).astype(np.uint8)
            plt.subplot(2, 2, 2*i+2)
            plt.imshow(rev_lbp, cmap='gray')
            plt.title('Резултантна слика')
            
            # LBP histogram
            lbp_histogram, _ = np.histogram(lbp_images[i].ravel(), bins=np.arange(257))
            
            # Plot LBP histogram
            plt.figure(figsize=(8, 4))
            plt.bar(np.arange(256), lbp_histogram)
            plt.title('LBP Хистограм')
            plt.xlabel('LBP Вредност')
            plt.ylabel('Учестаност')
            
            plt.show()
        
        
    def basePlot(arr1,arr2,name,res):
        plt.plot(arr1, label='Тачност на тренинг скупу података')
        plt.plot(arr2, label = 'Тачност над валидационом скупу података')
        plt.xlabel('Епоха')
        plt.ylabel('Тачност')
        plt.title(f"Архитектура типа 2 над скупом {res}")
        plt.ylim([0.4, 1])
        plt.legend(loc='lower right')
        plt.show()
        #plt.savefig(f'generated/{name}_{res}.png')
        
        
    def confMatrix(m_path,X_test,y_test):
        model = load_model(m_path)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        confusion_mtx = confusion_matrix(y_test, y_pred_classes)
        class_names = ['Абнормално ткиво', 'Нормално ткиво']
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Матрица конфузије")
        plt.show()
       
       
def main2():
        Plotter.processColors()
def main():
    #Plotter.processColors(debug=True)
    #Plotter.processContours()
    #Plotter.LBP()
    #print(tf.config.list_physical_devices('GPU'))
    #Plotter.basePlot([0.578,0.6359,0.6738,0.6859,0.6929,0.6940,0.6986,0.7029,0.7052,0.6971,0.6904,0.6894,0.6887,0.6913,0.6926,0.6938],[0.567,0.6434,0.6615,0.6942,0.6968,0.6886,0.7024,0.6961,0.7049,0.6901,0.6934,0.6882,0.6912,0.6923,0.6937,0.6856],"Model_M","80x80")
    data = np.load("generated/CNN_model_XL_160_160.npy",allow_pickle=True).item()
    #print(data)
    
    #data={'val_accuracy':[0.6629213690757751, 0.7150152921676636, 0.7123595476150513, 0.7403472661972046, 0.7438202500343323, 0.7477017641067505, 0.7474974393844604, 0.7595505714416504, 0.7720122337341309, 0.7812052965164185, 0.800000011920929, 0.7738508582115173, 0.7679264545440674, 0.7805924415588379, 0.8200204372406006, 0.7875382900238037, 0.7926455736160278, 0.8143002986907959, 0.80715012550354, 0.7981613874435425], 'accuracy':[0.6297091245651245, 0.675117015838623, 0.7076594829559326, 0.7259093523025513, 0.7368050217628479, 0.7447554469108582, 0.7534466981887817, 0.766004741191864, 0.7756897807121277, 0.7890428900718689, 0.8040763735771179, 0.8191460371017456, 0.84005206823349, 0.8551397919654846, 0.879298210144043, 0.8892000913619995, 0.8978190422058105, 0.9081365466117859, 0.9221039414405823, 0.9274162650108337]}
    #sacuvano jer je prethodno bilo overloadovano od strane novog modela koji se sluzio samo za testiranje; dev mistake
    #EPOHA 14: 0.8200 na val, 0.8551 na train i 0.817 na test ukupno 20 epoha->model L 120x120;
    #EPOHA 16: 0.8168 na val, 0.8064 na train i 0.8114374 na test ukupno 21 epoha->model L 80x80;
    #Pokreni trening za 120x120!!!
    #tip3 80x80 originalno 23 epohe
    #data['accuracy'].insert(0, 0.6468)
    #data['val_accuracy'].insert(0, 0.6982)
    #data['accuracy'].insert(1, 0.7031)
    #data['val_accuracy'].insert(1, 0.6672)
    #print(data['val_accuracy'])
    print(data['val_accuracy'].index(max(data['val_accuracy'])),max(data['val_accuracy']),max(data['accuracy']),data['accuracy'].index(max(data['accuracy'])))
    print(data['val_accuracy'],data['accuracy'])
    arr1=data['accuracy']
    arr2=data['val_accuracy']
    Plotter.basePlot(arr1,arr2,"Model_XL","80x80")

if __name__=='__main__':
    main2()