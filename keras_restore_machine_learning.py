from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import scipy
import os
import sys
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_image = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("im_01.jpg", flatten=True)))

def print_input(input):
    for it1, i in enumerate(input):
        #print('[', end='')
        for it2, u in enumerate(i):
            if it2 > 0:
                print(',', end='')
            #print('[', end='')
            for it, y in enumerate(u):
                if (it + 1) < len(u):
                    print(y, end=',')
                else:
                    print(y, end='\n')
    pass

#line = sys.stdin.readlines()
#for data in line:
#    test = data.split(',')
#    i = 0
#    while i < len(test):
#        a = 0
#        while a < 27:
#            print(float(test[i + a]) / 255, end=',')
#            a += 1
#        print(float(test[i + a]) / 255)
#        i += a

#img = cv2.imread('test5.png', 0)
#if img.shape != [28,28]:
#    img2 = cv2.resize(img,(28,28))
#    img = img2.reshape(28,28,-1);
#else:
#    img = img.reshape(28,28,-1);
#revert the image,and normalize it to 0-1 range
#img = 1.0 - img/255.0
#img = img.transpose(2, 0, 1)
#print(img)
#print_input(img)

def load_model(model_name, model_weights):
    # load json and create model
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model

def predict(loaded_model, input):
    predict = loaded_model.predict(input, verbose=0)
    print(np.argmax(predict))

def get_input():
    input_array = sys.stdin.readlines()
    values = [[]]
    idx = 0

    for line in input_array:
        data = line.split(',')
        if line == '\n':
            continue
        values[0].append([])
        for nbr in data:
            values[0][idx].append(float(nbr))
        idx += 1
    return np.array(values)

def main(argv):
    model_name = argv[1] #model.json
    model_weights = argv[2] #model.h5
    loaded_model = load_model(model_name, model_weights)
    input_data = get_input()
    predict(loaded_model, input_data)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('missing model or wights', file=sys.stderr)
        exit(1)
    main(sys.argv)