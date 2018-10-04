# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:05:11 2018

@author: b1016132
"""

import scipy.io.wavfile as wav
import librosa
from sklearn.svm import SVC
import numpy

def getMfcc(filename):
    y, sr = librosa.load(filename)
    return librosa.feature.mfcc(y=y, sr=sr)

animals = ["dog", "cat"]
voice_training = []
animal_training = []

#訓練データの作成
for animal in animals:
    print('Reading data of %s...' % animal)
    for number in range(1,41):
        mfcc = getMfcc('./audio_dogcat/%s (%s).wav' 
                       % (animal, number))
        voice_training.append(mfcc.T)
        label = numpy.full((mfcc.shape[1], ), #
            animals.index(animal), dtype=numpy.int)
        animal_training.append(label)
        print(number)
voice_training = numpy.concatenate(voice_training)
animal_training = numpy.concatenate(animal_training)
print(voice_training)
print(animal_training)
print(voice_training.shape)
print(animal_training.shape)


#学習
clf = SVC(C=1, gamma=1e-5)
clf.fit(voice_training, animal_training)
print('Learning Done')

#予測
mfcc = getMfcc('cat1.wav')
prediction = clf.predict(mfcc.T)
count = numpy.bincount(prediction)
result = animals[numpy.argmax(count-count.mean(axis=0))]
print('prediction:%s.'% result)