import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from audioSignal import AudioSignal
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size // 2)
def completArray():
    d=0
def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 20, frames = 11):
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('fold')[1].split('/')[0]
            tmp = AudioSignal(fn)
            tmp.normalizeEnergy() 
            tmp.signalFeatures(0.01, 0.025)
            mfccs.append(tmp.featuresMFCC[0:11] )  
            labels.append(label)       
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    return np.array(features), np.array(labels,dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,10))
    #print(one_hot_encode)
    #one_hot_encode[np.arange(n_labels), labels] = 1
    for i in range(n_labels):
        one_hot_encode[i][labels[0]-1] = 1
    
    return one_hot_encode

    # use this to process the audio files into numpy arrays
def save_folds(data_dir):
    for k in range(1,11):
        fold_name = 'fold' + str(k)
        print("\nSaving " + fold_name)
        features, labels = extract_features(parent_dir, [fold_name])
        labels = one_hot_encode(labels)
        
        #print("Features of", fold_name , " = ", features.shape)
        #print("Labels of", fold_name , " = ", labels.shape)
        
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        np.save(feature_file, features)
        #print("Saved " + feature_file)
        np.save(labels_file, labels)
        #print("Saved " + labels_file)
        #print(labels)

def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)
        
#uncomment this to recreate and save the feature vectors
parent_dir = "/home/ariel/TEC/patrones/proyecto2/audioRNN/" # Where you have saved the UrbanSound8K data set"       
save_dir = "/home/ariel/TEC/patrones/proyecto2/audioRNN/data"
assure_path_exists(save_dir)
save_folds(save_dir)

# this is used to load the folds incrementally
def load_folds(folds):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print (fold_name, "features: ", loaded_features.shape)

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True
        
    return features, labels
data_dir = "/home/ariel/TEC/patrones/proyecto2/audioRNN/data"

def extract_feature_array(filename, bands = 20, frames = 11):
    mfccs = []
    sound_clip,s = librosa.load(filename)
    tmp = AudioSignal(filename)
    tmp.normalizeEnergy() 
    tmp.signalFeatures(0.01, 0.025)
    mfccs.append(tmp.featuresMFCC[0:11])
            
    features = np.asarray(mfccs)
    return np.array(features)

sample_filename = "samples/us8k/music.wav"
features = extract_feature_array(sample_filename)
data_points, _ = librosa.load(sample_filename)
print ("IN: Initial Data Points =", len(data_points))
print ("OUT: Total features =", np.shape(features))    

tf.set_random_seed(0)
np.random.seed(0)

def evaluate(model):
    y_prob = model.predict_proba(test_x, verbose=0)
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(test_y, 1)

    #roc = roc_auc_score(test_y, y_prob)
    #print ("ROC:",  round(roc,3))

    # evaluate the model
    score, accuracy = model.evaluate(test_x, test_y, batch_size=32)
    print("\nAccuracy = {:.2f}".format(accuracy))

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print ("F-Score:", round(f,2))
    
    return accuracy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

data_dim = 20
timesteps = 11
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()

# returns a sequence of vectors of dimension 512
model.add(LSTM(300, return_sequences=True, input_shape=(timesteps, data_dim)))  

model.add(Dropout(0.2))

# return a single vector of dimension 128
model.add(LSTM(16))  

model.add(Dropout(0.2))

# apply softmax to output
model.add(Dense(num_classes, activation='softmax'))


# compile the model for multi-class classification
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def getTest(num,datax,datay):
    test_x=[]
    test_y=[]
    idx = np.random.randint(len(datax), size=num)
    test_x=datax[idx,:]
    test_y=datay[idx,:]
    train_x= np.delete(datax, idx,0)
    train_y= np.delete(datay, idx,0)
    print("shape train",train_x.shape,train_y.shape)
    return train_x,train_y,test_x,test_y

# load fold1 for testing
datax, datay = load_folds([1,2,3,4,5,6,7,8,9,10])
print(datax.shape,datay.shape)
train_x,train_y,test_x,test_y=getTest(30,datax,datay)
print("wwwwwww")
#print(train_x.shape)
#print(train_x)
# load fold2 for validation
valid_x, valid_y = load_folds([9])

# a stopping function to stop training before we excessively overfit to the training set
earlystop = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

model.fit(train_x, train_y, batch_size=128, nb_epoch=30, validation_data=(test_x, test_y))    

print("Evaluating model...")
acc = evaluate(model)

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

labels = ["uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve","diez"]
print ("Showing Confusion Matrix")
y_prob = model.predict_proba(test_x, verbose=0)
y_pred = y_prob.argmax(axis=-1)
y_true = np.argmax(test_y, 1)
cm = confusion_matrix(y_true, y_pred)
#df_cm = pd.DataFrame(cm, labels, labels)
plt.figure(figsize = (16,8))
sn.heatmap(cm, annot=True, annot_kws={"size": 14}, fmt='g', linewidths=.5)
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.show()
sound_file_paths = ["uno_22_1.wav","dos_11_2.wav","tres_26_2.wav","cuatro_18_2.wav","cinco_30_3.wav", 
"seis_27_3.wav","siete_7_3.wav","ocho_10_3.wav","nueve_25_3.wav","diez_24_1.wav"]
sound_names = ["uno","dos","tres","cuatro","cinco","seis",
               "siete","ocho","nueve","diez"]
parent_dir = 'samples/digits/'


# create predictions for each of the sound classes
for s in range(len(sound_file_paths)):

    print ("\n----- ", sound_names[s], "-----")
    # load audio file and extract features
    predict_file = parent_dir + sound_file_paths[s]
    predict_x = extract_feature_array(predict_file)
    
    # generate prediction, passing in just a single row of features
    predictions = model.predict(predict_x)
    
    if len(predictions) == 0: 
        print ("No prediction")
        continue
    
    #for i in range(len(predictions[0])):
    #    print sound_names[i], "=", round(predictions[0,i] * 100, 1)
    
    # get the indices of the top 2 predictions, invert into descending order
    ind = np.argpartition(predictions[0], -2)[-2:]
    ind[np.argsort(predictions[0][ind])]
    ind = ind[::-1]
    
    print ("Top guess: ", sound_names[ind[0]], " (",round(predictions[0,ind[0]],3),")")
    print ("2nd guess: ", sound_names[ind[1]], " (",round(predictions[0,ind[1]],3),")")