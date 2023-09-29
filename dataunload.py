import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
folder_path = "D:\\Python\\python_learning\\neural network\\Car-Bike-Dataset"
catagories = ["Car","Bike"]



def normalize(X_train,X_test):
    X_train = X_train/255
    X_test = X_test/255

    return X_train,X_test

def flatten(X_train,X_test):
    X_train = X_train.reshape((X_train.shape[0],-1)).T
    X_test = X_test.reshape((X_test.shape[0],-1)).T

    return X_train,X_test

def generate_dataset(train_x,train_y,test_x,test_y):
    X_train = np.array(train_x)
    Y_train = np.array(train_y)
    X_test = np.array(test_x)
    Y_test = np.array(test_y)
 
    print("before reshaping \n")
    print("X_train ",X_train.shape)
    print("X_test ",X_test.shape)
    print("Y_train ",Y_train.shape)
    print("Y_test ",Y_test.shape)

    Y_train = Y_train.reshape((1,len(Y_train)))
    Y_test = Y_test.reshape((1,len(Y_test)))

    X_train,X_test = flatten(X_train,X_test)

    X_train,X_test = normalize(X_train,X_test)

    return X_train,Y_train,X_test,Y_test

def explicit_f_l(train_cars_bikes,test_cars_bikes):
    train_x=[]
    test_x=[]
    train_y=[]
    test_y=[]

    for f,l in train_cars_bikes:
        train_x.append(f)
        train_y.append(l)
    for f,l in test_cars_bikes:
        test_x.append(f)
        test_y.append(l)

    return train_x,train_y,test_x,test_y    

def split_train_test(image_data):
    half_len = len(image_data)//2

    cars = image_data[:half_len]
    bikes = image_data[half_len:len(image_data)]

    split_num = int(0.9*len(cars))

    train_cars = cars[:split_num]
    test_cars = cars[split_num:len(cars)]

    train_bikes = bikes[:split_num]
    test_bikes = bikes[split_num:len(bikes)]

    train_cars_bikes = train_cars + train_bikes 
    test_cars_bikes = test_cars + test_bikes   

    random.shuffle(train_cars_bikes)
    random.shuffle(test_cars_bikes)
    print(len(train_cars_bikes))
    print(len(test_cars_bikes))

    return train_cars_bikes, test_cars_bikes

def main():

    image_data = []

    for category in catagories:
        label = catagories.index(category)
        current_path = os.path.join(folder_path,category)
        print(len(os.listdir(current_path))) 

        for index, image_name in enumerate(os.listdir(current_path),start=1):
            if index == 5:
                break
            else:
        
                try:
                    image = cv2.imread(os.path.join(current_path,image_name))
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image,(64,64))
                    image_data.append([image,label])
                except:
                    pass
    
    return image_data
    



image_data = main() 
train_cars_bikes, test_cars_bikes = split_train_test(image_data)
train_x,train_y,test_x,test_y = explicit_f_l(train_cars_bikes,test_cars_bikes)
X_train,Y_train,X_test,Y_test = generate_dataset(train_x,train_y,test_x,test_y)

print("X_train ",X_train.shape)
print("X_test ",X_test.shape)
print("Y_train ",Y_train.shape)
print("Y_test ",Y_test.shape)