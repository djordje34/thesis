import os
import random
import time
from datetime import datetime
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models,optimizers
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import visualkeras
from PIL import ImageFont
from plotter import Plotter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
import seaborn as sns

class CNNModel:
    """Wrapper klasa za objekat tf.keras.models.Sequential klase
    
    Attributes
    ----------
    model : tf.keras.models.Sequential
        Model CNN mreze
    init_input : bool
        Da li je inicijalizovan ulazni sloj mreze

     
    -------
    InputLayer(self,inputShape:tuple[int,...],batch_size:int)->tf.keras.models.Sequential:
        Ulazni sloj modela mreze.
        
    ConvBlock(self,inputShape:tuple[int,...],batch_size:int,numFilters:int, kernelSize:int = 3, doBatchNorm:bool = False, addDropout:bool = False,dropoutMultiplier:int=1)->tf.keras.models.Sequential:
        Dodaje standardni blok u CNN model. Ukljucuje Conv2D,MaxPool i opcionom BatchNorm i Dropout slojeve. Dodati faktori za promenu rate-a Dropout sloja.
        
    FC(self,numNeurons:int=64, doBatchNorm:bool = False, addDropout:bool = False,dropoutMultiplier:int=2)->ttf.keras.models.Sequential:
        Dodaje potpuni povezani (FC) sloj u model. Sadrzi Flatten za 'peglanje' tensor-a, relu FC za opcionim brojem neurona,sigmoid izlaz za binarnu klasifikaciju i
        opciono dodavanje BatchNorm i Dropout slojeve.
    
    DataLoader(self, subset:str, path:str="arrays", grayscale:bool=False)->None:
        Ocitavanje podataka iz skupova.
        
    Compile(self, optimizer: list[str] = 'adam', loss: list[str] = 'binary_crossentropy', metrics: list[str] = ['accuracy'], learning_rate: float = 0.001) -> tf.keras.models.Sequential:
        Kompajlira model sa navedenim optimizatorom, funkcijom gubitaka, i metrikom.
    
    PlotArchitecture(self,path:str='generated')->None:
        Iscrtava grafike koji prikazuju model.
    """
    def __init__(self,name:str="CNN_model"):
        """Pravi Sequential keras model.

        Args:
            name (str, optional): Podesavanje naziva modela. Defaults to "CNN_model".
        """
        self.model = models.Sequential(name=name)
        self.init_input=False
        
    def InputLayer(self,inputShape:tuple[int,...])->tf.keras.models.Sequential:
        """Ulazni sloj modela mreze

        Args:
            inputShape (tuple[int,...]): Dimenzija ulazne slike

        Returns:
            tf.keras.models.Sequential: Model mreze
        """
        
        self.model.add(layers.Input(shape=inputShape))
        self.init_input = True
        return self.model
        
        
    def ConvBlock(self,inputShape:tuple[int,...],numFilters:int, kernelSize:int = 3, doBatchNorm:bool = False, addDropout:bool = False,dropoutMultiplier:int=1,double:bool=False)->tf.keras.models.Sequential:
        """Dodaje standardni blok u CNN model. Ukljucuje Conv2D,MaxPool i opcionom BatchNorm i Dropout slojeve. Dodati faktori za promenu rate-a Dropout sloja.

        Args:
            inputShape (tuple[int,...]): Dimenzija ulazne slike
            numFilters (int): Broj filtera u konv. sloju
            kernelSize (int, optional): Velicina filtra konvolucije. Defaults to 3.
            doBatchNorm (bool, optional): Batch normalizacija. Defaults to False.
            addDropout (bool, optional): Dropout sloj. Defaults to False.
            dropoutMultiplier (int, optional): Faktor kojim se mnozi default Dropout rate. Defaults to 1.
            double(bool, optional): Dupla conv2d. Defaults to False.
        Returns:
            tf.keras.models.Sequential: Model sa dodatim slojevima.
        """
        if not self.init_input:
            self.model = self.InputLayer(inputShape)
            
            
        self.model.add(layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),activation='relu'))
        if double:
            self.model.add(layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),activation='relu'))
        if doBatchNorm:
            self.model.add(tf.keras.layers.BatchNormalization())
        
        self.model.add(layers.MaxPooling2D((2, 2)))
        if addDropout:
            self.model.add(layers.Dropout(0.1*dropoutMultiplier))
        return self.model
    
    
    def FC(self,numNeurons:int=64, doBatchNorm:bool = False, addDropout:bool = False,dropoutMultiplier:int=2,end:bool=True,flatten:bool=False)->tf.keras.models.Sequential:
        """Dodaje potpuni povezani (FC) sloj u model. Sadrzi Flatten za 'peglanje' tensor-a, relu FC za opcionim brojem neurona,sigmoid izlaz za binarnu klasifikaciju i
        opciono dodavanje BatchNorm i Dropout slojeve.

        Args:
            numNeurons(int, optional): Broj neurona u relu FC sloju. Defaults to 64.
            doBatchNorm (bool, optional): Batch Normalization sloj. Defaults to False.
            addDropout (bool, optional): Dropout sloj. Defaults to False.
            dropoutMultiplier(int, optional): Faktor kojim se mnozi default Dropout rate. Defaults to 2.
            end(bool, optional): Da li je dati layer zadnji

        Returns:
            tf.keras.models.Sequential: Model sa dodatim pratecim slojevima i izlaznim slojem.
        """
        
        if flatten:
            self.model.add(layers.Flatten())
            
        self.model.add(layers.Dense(numNeurons, activation='relu'))
        
        if doBatchNorm:
            self.model.add(tf.keras.layers.BatchNormalization())
            
        if addDropout:
            self.model.add(layers.Dropout(0.2*dropoutMultiplier))
            
        if end:
            self.model.add(layers.Dense(1,activation='sigmoid'))
        
        return self.model


    def DataLoader(self, subset:str, path:str="arrays", grayscale:bool=False)->None:
        """Ocitavanje podataka iz skupova.

        Args:
            grayscale(bool): Pretvaranje RGB u Grayscale. Defaults to False.
            subset (str): Koji subset podataka treba ocitati.
            path (str, optional): Putanja do datoteke sa skupovima podataka. Defaults to "arrays".
        """
        
        data = np.load(f"{path}/{subset}.npz")
        if grayscale:
            print("Conversion of X_train")
            X_train_gs = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in data['X_train']])/ 255.0
            print("Conversion of X_val")
            X_val_gs = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in data['X_val']])/ 255.0
            print("Conversion of X_test")
            X_test_gs = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in data['X_test']])/ 255.0
            print("Conversion done")
        self.X_train = data['X_train']/ 255.0 if not grayscale else X_train_gs
        self.y_train = data['y_train']
        self.X_val = data['X_val']/ 255.0 if not grayscale else X_val_gs
        self.y_val = data['y_val']
        self.X_test = data['X_test']/ 255.0 if not grayscale else X_test_gs
        self.y_test = data['y_test']
        
    def Compile(self, optimizer: list[str] = 'adam', loss: list[str] = 'binary_crossentropy', metrics: list[str] = ['accuracy'], learning_rate: float = 0.01) -> tf.keras.models.Sequential:
        """Kompajlira model sa navedenim optimizatorom, funkcijom gubitaka, i metrikom.

        Args:
            optimizer (str or tf.keras.optimizers.Optimizer, optional): Optimizator za treniranje. Defaults to 'adam'.
            loss (str or tf.keras.losses.Loss, optional): Funkcija gubitaka. Defaults to 'binary_crossentropy'.
            metrics (List of str or List of tf.keras.metrics.Metric, optional): Evaluaciona metrika. Defaults to ['accuracy'].
            learning_rate (float, optional): Stopa ucenja. Defaults to 0.01.
        Returns:
            tf.keras.models.Sequential: Kompajliran model.
        """
        
        
        optimizer = optimizers.get(optimizer)
        optimizer.learning_rate.assign(learning_rate)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

    def PlotArchitecture(self,path:str='generated/architectures')->None:
        """Iscrtava grafike koji prikazuju model.

        Args:
            path (str, optional): Putanja do datoteke gde se smestaju grafici. Defaults to 'generated'.
        """
        
        
        plot_model(self.model, to_file=f"{path}/CNNarch_{datetime.now().strftime('%d%m%Y%H%M%S-')}_{self.model.name}.png", show_shapes=True, show_layer_names=True)
        font = ImageFont.truetype("arial.ttf", 12)
        visualkeras.layered_view(self.model, legend=True, font=font,to_file=f"{path}/CNNvisual_{datetime.now().strftime('%d%m%Y_%H_%M_%S-')}_{self.model.name}.png")


    def SaveModel(self,path:str="models",name:str=f"model_{datetime.now().strftime('%d%m%Y%H%M%S-')+str(random.randrange(0000,9999))}",param:str="")->None:
        self.model.save(f"{path}/{self.model.name}_{name}_{param}")
        print(f"Model sacuvan: {path}/{self.model.name}_{name}_{param}")

def main():
    start_time = time.time()
    size = '160'
    model = CNNModel(name=f"CNN_model_XL_{size}")
    model.DataLoader(size,grayscale=True)
    
    print(f"Conversion time: {((conversion_time := time.time())-start_time)} sec.")
    
    # dodatna dim za grayscale conv zbog inputa
    model.X_train = np.expand_dims(model.X_train, axis=-1)
    model.X_val = np.expand_dims(model.X_val, axis=-1)
    model.X_test = np.expand_dims(model.X_test, axis=-1)
    # Fit the data generator on the training data
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}.h5',  # path
    monitor='val_loss',  # metrika
    save_best_only=False,  # sve verzije modela
    save_freq='epoch'  
    )
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model.ConvBlock(inputShape=(int(size),int(size),1),numFilters=64,kernelSize=3,doBatchNorm=False,addDropout=True,dropoutMultiplier=1,double=True)
    model.ConvBlock(inputShape=(int(size),int(size),1),numFilters=128,kernelSize=3,doBatchNorm=False,addDropout=True,dropoutMultiplier=2,double=True)
    model.ConvBlock(inputShape=(int(size),int(size),1),numFilters=256,kernelSize=3,doBatchNorm=False,addDropout=True,dropoutMultiplier=3, double=True)
    #model.ConvBlock(inputShape=(int(size),int(size),1),numFilters=256,kernelSize=3,doBatchNorm=False,addDropout=True,dropoutMultiplier=4, double=True)
    #model.FC(numNeurons=512,doBatchNorm=False,addDropout=True,dropoutMultiplier=2,end=False,flatten=True)
    model.FC(numNeurons=512,doBatchNorm=False,addDropout=True,dropoutMultiplier=2,end=False,flatten=True)
    model.FC(numNeurons=256,doBatchNorm=False,addDropout=True,dropoutMultiplier=1,end=True)
    #model.FC(numNeurons=64,doBatchNorm=False,addDropout=False,dropoutMultiplier=3,end=True)
    
    for layer in model.model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            layer.kernel_regularizer = tf.keras.regularizers.l2(0.001)
    
    model.Compile(optimizer='adam', learning_rate=0.0001)
    last_epoch = 0
    if any(os.scandir("checkpoints")):
        model.model = tf.keras.models.load_model(f"checkpoints/model_{(model_num:=str(last_epoch:=max([int(x.split('_')[1].split('.')[0]) if '0' != x.split('_')[1].split('.')[0][0] else int(x.split('_')[1].split('.')[0]) for x in os.listdir('checkpoints/')]))).zfill(2)}.h5")
        print(f"Loaded iteration {model_num}")
    model.PlotArchitecture()
    
    print(f"Model setup time: {((setup_time := time.time())-conversion_time)} sec.")
    
    history = model.model.fit(
        model.X_train, model.y_train, batch_size=128,
        steps_per_epoch=len(model.X_train) // 128,  #steps po epohi
        initial_epoch=last_epoch,  
        epochs=30,
        validation_data=(model.X_val, model.y_val),
        callbacks=[checkpoint_callback, reduce_lr_callback, early_stopping_callback]
    )
    
    print(f"Train time: {((train_time:=time.time())-setup_time)} sec.")
    print(history.history)
    np.save(f'generated/{model.model.name}_{size}.npy',history.history)
    """plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()"""
    if not len(history.history):
        np.save(f'generated/{model.model.name}_{size}.npy',history.history)
    test_loss, test_acc = model.model.evaluate(model.X_test,  model.y_test)
    model.SaveModel(param=str(round(test_acc,3)))
    print(f"Test time: {((test_time:=time.time())-train_time)} sec.") 
    
    print(test_loss,test_acc)
    
    
def cmplotter():
    size='160'
    wrapper = CNNModel(name=f"CNN_model_XL_{size}")
    wrapper.DataLoader(size,grayscale=True)
    m_path='models/CNN_model_XL_160_model_17082023114805-3049_0.853'
    X_test = wrapper.X_test
    y_test = wrapper.y_test
    model = load_model(m_path)
    model.summary()
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(test_loss,test_acc)
    
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred_classes=np.round(y_pred, decimals = 0)
    confusion_mtx = confusion_matrix(y_test, y_pred_classes)
    print(confusion_matrix)
    class_names = ['Абнормално ткиво', 'Нормално ткиво']
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title("Матрица конфузије")
    plt.show()
        
        
def tempCM():
    confusion_matrix = np.array([[50, 5], [5, 50]])
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    categories = ['Абнормално ткиво', 'Нормално ткиво']
    sns.set(font_scale=1.2)
    plt.figure(figsize=(9,6))
    heatmap = sns.heatmap(confusion_matrix, annot=False,cbar=False, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    # Set axis labels and title
    plt.xlabel('Предвиђена класа')
    plt.ylabel('Стварна класа')
    plt.title('Матрица конфузије')
    plt.text(0.5, 0.5, 'TN\n(Тачне негативне)', ha='center', va='center', color='white', fontsize=12)
    #plt.text(1.5, 0.5, 'FP', ha='center', va='center', color=fp, fontsize=12)
    #plt.text(0.5, 1.5, 'FN', ha='center', va='center', color=fp, fontsize=12)
    plt.text(1.5, 0.5, 'FP\n(Лажне позитивне)', ha='center', va='center', color=sns.color_palette("Blues")[5], fontsize=12)
    plt.text(0.5, 1.5, 'FN\n(Лажне негативне)', ha='center', va='center', color=sns.color_palette("Blues")[5], fontsize=12)
    plt.text(1.5, 1.5, 'TP\n(Тачне позитивне)', ha='center', va='center', color='white', fontsize=12)
    # Add labels to each cell
    #for i in range(2):
        #for j in range(2):
            #plt.text(j + 0.5, i + 0.5, str(confusion_matrix[i, j]), ha='center', va='center', color='black', fontsize=12)

    # Show the plot
    plt.show()
if __name__=='__main__':
    cmplotter()
    #tempCM()
    #main()
        