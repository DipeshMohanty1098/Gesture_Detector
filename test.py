import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from cv2 import error

dir = 'D:/train/train'
dir2 = 'D:/train/test'
size = 50

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'ok': return 1
    elif word_label == 'thumbs': return 0

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(dir)):
        label = label_img(img)
        path = os.path.join(dir,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (size,size))
        training_data.append([np.array(img), label])
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data

def process_test():
    testing_data = []
    for img in tqdm(os.listdir(dir2)):
        label = label_img(img)
        path = os.path.join(dir2, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (size, size))
        testing_data.append([np.array(img), label])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
final_train = train_data
test_data = process_test()
final_test = test_data