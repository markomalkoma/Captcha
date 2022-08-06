# go to .keras (Users/user) and change from 'tensorflow' to 'theano'
##conda install -n soft-env -c conda-forge opencv=3.4.1
##conda install -n soft-env -c conda-forge keras=2.1.5
##conda install -n soft-env -c conda-forge theano=1.0.4
##install fuzzywuzzy
##instal pandas
##install matplotlib
##install h5py (stable 2xx version, 3xx are problematic)

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import product
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
import json
import h5py
import pandas as pd
from fuzzywuzzy import fuzz


# resize image to 28 x 28
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

# select pixels with letters color
def letters_selection(path):
    img_origin = cv2.imread(path)
    rows, cols, _ = img_origin.shape
    cells = rows*cols
    # reshaping image from 3d to 2d (one row with RGB pixels)
    pixels = np.reshape(img_origin, (cells, 3))
    # unique colors, index of their first appearance, frequency
    color, index, count = np.unique(pixels, axis = 0, return_counts = True, return_index = True)

    c = []

    # letter color search - 4 conditions
    for i in range(len(color)):
        CON_1 = cells*0.0001 < count[i] < cells*0.30
        CON_2 = int(color[i][0]) + int(color[i][1]) + int(color[i][2]) < 700
        CON_3 = int(color[i][0]) + int(color[i][1]) + int(color[i][2]) > 150
        CON_4 = np.var(color[i]) > 50
        if CON_1 and CON_2 and CON_3 and CON_4:
            c.append((count[i], index[i]))

    c.sort(reverse = True)
    #print(f'\nImage number: {n_img}')
    #print(f'Number of colors: {len(c)}')
    if c:
        # most common RGB color for selected conditions
        hit = pixels[c[0][1]]
        #print(f'Letters color: {hit}')
        #colors.append((n_img, hit))

        l = hit - 10
        h = hit + 10

        # result is a dark background with originaly colored letters
        mask = cv2.inRange(img_origin, l, h)
        result = cv2.bitwise_and(img_origin, img_origin, mask=mask)

        return result, hit

# rotate inclined sentences
def rotation(result, hit):

    rows, cols, _ = result.shape
    
    X = []
    Y = []

    # finding sentence inclination - calculating mid-points for dilated graph
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    img_dil = cv2.dilate(result, kernel, iterations=5)
    '''
    cv2.imshow('', img_dil)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # we search for <black - letter color - black> switch (eliminate possible mistakes with letter color on edges)
    # IMPORTANT!!! STEP directly determine time of code
    # LARGER STEP - FASTER CODE!!!
    STEP = 10 # step (from 1 to 10) 1 - very slow!!
    for x in range(0,cols,STEP):
        tresh = []
        color = 'green'
        switch = 0
        for y in range(rows):
            if np.array_equal(img_dil[y][x], np.array([0,0,0])) and color == 'green':
                color = 'black'
                switch += 1
                tresh.append(y)
            if np.array_equal(img_dil[y][x], hit) and color == 'black':
                color = 'green'
                switch += 1
                tresh.append(y)
        if switch == 3:
            X.append([x])
            # mid-point
            Y.append(int((tresh[2]+tresh[1])/2))


    # linear fit - applying linear regression on mid-points           
    reg = LinearRegression().fit(X, Y)
    # tangens
    tan = reg.coef_[0]

    # rotation
    if abs(tan) > 0.03:
        #print(f'Tangens: {tan}')
        angle = np.degrees(np.arctan(tan))
        #print(f'Angle: {angle}')
        
        cx, cy = cols//2, rows//2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        result = cv2.warpAffine(result, M, (cols, rows))

    else:
        pass
        #print(f'No need for rotation.')

    return result

# color to gray   
def gray(result):
    img_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    return img_gray

# gray to bin
def binary(img_gray):
    ret, img_bin = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    return img_bin

# resizing all sentences to the same hight
def resize_complete(img_bin):

    rows, cols = img_bin.shape
    
    # applying erosion for detecting treshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    img_ero = cv2.erode(img_bin, kernel, iterations=1)

    '''
    cv2.imshow(f'Image{n_img}', img_ero)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    treshold = []

    for y, x in product(range(rows), range(cols)):
        if np.array_equal(img_ero[y][x], 255):
            treshold.append(y)
            break
    for y, x in product(range(rows-1,0,-1), range(cols)):
        if np.array_equal(img_ero[y][x], 255):
            treshold.append(y)
            break
    
    #print(img_ero)
    #print(f'Letters treshold: {treshold}')

    if treshold:
        img_bin = img_bin[treshold[0]:treshold[1]]
        rows, cols = img_bin.shape
        rows_n = 200
        cols_n = (cols*200)//rows
        img_bin = cv2.resize(img_bin, (cols_n, 200))

        return img_bin

    return None

# drawing rectangles around each letter
def rectangles(img_bin):
    # finding contours for binary image
    img_con, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # creating rectangulars arround each contour that fullfiles conditions        
    regions_array = []
    #print(f'Number of contures: {len(contours)}')

    bin_tester = img_bin.copy()
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10 and 15 < h < 500 and w > 10:
            region = img_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(bin_tester, (x,y), (x+w,y+h), 150, 2)
            
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    # grouping rectangles that belongs to the same letter
    sorted_regions = [[regions_array[0]]]
    for rec in regions_array[1:]:
        if rec[1][0] < sorted_regions[-1][0][1][0] + sorted_regions[-1][0][1][2] + 1:
            sorted_regions[-1].append(rec)
        else:
            sorted_regions.append([rec])

    letters_img = []
    letters_loc = []
    for letter in sorted_regions:
        let_img = []
        let_loc = []
        for i in letter:
           let_img.append(i[0])
           let_loc.append(i[1])
        letters_img.append(let_img)
        letters_loc.append(let_loc)
    '''
    cv2.imshow(f'Image', bin_tester)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return letters_img, letters_loc

# finding whitespaces in the sentence
def whitespace(letters_loc):
    spaces = []
    for inx in range(len(letters_loc)-1):
        if letters_loc[inx+1][0][0] - (letters_loc[inx][0][0] + letters_loc[inx][0][2]) > 30:
            spaces.append(inx)
            
    return spaces

# INPUTS - X
# alphabet images are flatenn (from 28 x 28 to 784) and normalized (from 0 to 1)
def preparation(regions):
    flat_normal = []
    for img in regions:
        img = img/255
        img = img.flatten()
        flat_normal.append(img)        
    return flat_normal

# OUTPUTS - Y
# converting alphabet (strings) into [0,0,0,...0,0] with one 1 at the place of letter index
def converter(alphabet):
    outputs = []
    for inx in range(len(alphabet)):
        l = np.zeros(len(alphabet))
        l[inx] = 1
        outputs.append(l)
    return np.array(outputs)

# load traind model
# pip install h5py (stable 2xx version, 3xx are problematic)
def load_trained_nn():
    try:
        json_file = open('nn.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        nn = model_from_json(loaded_model_json)
        nn.load_weights("nn.h5")
        return nn
    except Exception as e:
        return None

# 128 - hidden layer, 784 - number of pixels, O - number of posible outputs
def create_nn(O):
    nn = Sequential()
    nn.add(Dense(128, input_dim=784, activation='sigmoid'))
    nn.add(Dense(O, activation='sigmoid'))
    return nn

# train
def train_nn(nn, X, Y):
    X = np.array(X, np.float32) 
    Y = np.array(Y, np.float32) 
    sgd = SGD(lr=0.01, momentum=0.9)
    nn.compile(loss='mean_squared_error', optimizer=sgd)
    nn.fit(X, Y, epochs=500, batch_size=1, verbose = 0, shuffle=False)      
    return nn

# save model in json file
def serialize_nn(nn):
    model_json = nn.to_json()
    with open("nn.json", "w") as json_file:
        json_file.write(model_json)
    nn.save_weights("nn.h5")
    
# winning letter
def winner(output): 
    return max(enumerate(output), key=lambda x: x[1])[0]

# return 
def display_result(outputs, alphabet):
    for output in outputs:
        result = alphabet[winner(output)]
    return result

# TRAIN     
def train_or_load_character_recognition_model(paths):
        
    # try to load model
    nn = load_trained_nn()

    # if model is not yet trained
    if nn == None:

        #print('model does not exists yet')
        
        # 1 ALPHABET IMAGES PREPARATION
        alphabet_img = []

        result, hit = letters_selection(paths[1])
        img_gray = gray(result)
        img_bin = binary(img_gray)
        img_bin = resize_complete(img_bin)
        letters_img, letters_loc = rectangles(img_bin)
        
        for l in letters_img:
            img = l[0]
            alphabet_img.append(img)

        # as 784 x [0-1]
        X = preparation(alphabet_img)

        '''
        for img in alphabet_img:
            cv2.imshow(f'Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''

        # 2 CREATING ALPHABET
        ABC = ['A','B','C','Č','Ć','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','Š','T','U','V','W','X','Y','Z','Ž']
        abc = ['a','b','c','č','ć','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','š','t','u','v','w','x','y','z','ž']

        alphabet_str = abc

        # as [0,0,0...0,1,0...]
        Y = converter(alphabet_str)
        
        # 3 CREATING NEURAL NETWORK
        nn = create_nn(len(alphabet_str))
        
        # 4 TRAINING NEURAL NETWORK
        nn = train_nn(nn, X, Y)
        # save model
        serialize_nn(nn)

    #print(nn)
    return nn

# VALIDATION
def extract_text_from_image(model, path, dicto):

    matcher = []
    for word, frequency in dicto.items():
        matcher.append(word)
    
    try:
        extracted_text = ''

        result, hit = letters_selection(path)
        result = rotation(result, hit)
        img_gray = gray(result)
        img_bin = binary(img_gray)
        img_bin = resize_complete(img_bin)
        letters_img, letters_loc = rectangles(img_bin)

        first_img = []

        for l in letters_img:
            img = l[0]
            first_img.append(img)

        # as 784 x [0-1]
        X = preparation(first_img)
            
        spaces = whitespace(letters_loc)

        alphabet_str = ['a','b','c','č','ć','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','š','t','u','v','w','x','y','z','ž']

        for inx in range(len(X)):
            results = model.predict(np.array([X[inx]], np.float32))
            letter = display_result(results, alphabet_str)

            if letter in ['s', 'š']:
                if len(letters_loc[inx]) > 1 and letters_loc[inx][1][1] < letters_loc[inx][0][1]:
                    letter = 'š'
                else:
                    letter = 's'
                
            if letter in ['z', 'ž']:
                if len(letters_loc[inx]) > 1 and letters_loc[inx][1][1] < letters_loc[inx][0][1]:
                    letter = 'ž'
                else:
                    letter = 'z'
            
            if letter in ['c', 'ć', 'č']:
                if len(letters_loc[inx]) > 1 and letters_loc[inx][1][1] < letters_loc[inx][0][1]:
                    letter = 'č'
                else:
                    letter = 'c'

            if inx == 0:
                letter = letter.capitalize()
                if letter == 'L':
                    letter = 'I'

            extracted_text += letter
                
            if inx in spaces:
                extracted_text += ' '

        extracted_text = extracted_text.replace(' L ', ' I ')
        extracted_text = extracted_text.replace(' i ',' I ')

        sentence = []

        word_inx = 0
        for predict in extracted_text.split():
            hits = []
            for word in matcher:
                score = fuzz.ratio(predict, word)
                hits.append((score, word))
            #print(hits[0])
            hits.sort(reverse=True)
            print(hits[0])
            if not word_inx:
                for h in hits:
                    if h[1][0].isupper():
                        sentence.append(h[1])
                        break
            else:
                sentence.append(hits[0][1])
            word_inx+=1

        extracted_text = ' '.join(sentence)
                
        return extracted_text

    except:
        extracted_text = 'I gess this must be some long and boring sentence'
        return extracted_text


# TESTING

df = pd.read_csv("dataset/validation/annotations.csv")
sentences = df.Sentence.to_list()
#print(sentences)

efficiency = 0

dicto = dict()
with open('dataset/dict.txt', 'r', encoding='utf-8') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split()
        if len(cols) == 3:
            dicto[cols[1]] = int(cols[2])
            
for n_img in range(100):

    results = pd
    
    path_0 = 'dataset/train/alphabet0.png'
    path_1 = 'dataset/train/alphabet1.png'
    path = 'dataset/validation/train' + str(n_img) + '.png'

    paths = [path_0, path_1]

    model = train_or_load_character_recognition_model(paths)
    text = extract_text_from_image(model, path, dicto)

    print(f'image number {n_img}')
    print(text)

    calculated = set(text.split())
    real = set(sentences[n_img].split())

    eff = len(calculated.intersection(real))/len(calculated.union(real))
    efficiency+=eff
    
    print(efficiency/(n_img + 1))


   
        
    




















    
                
