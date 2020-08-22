import csv
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import ndimage, misc
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping

#Load data from csv file and store row information in list 
def load_data(dir_name): 
    lines = []
    file = open(dir_name + '/driving_log.csv')
    reader = csv.reader(file)
    next(reader)
    for line in reader:
        lines.append(line)
        
    return lines

def extract_steer(data):
    angle = []
    for i in range(len(data)):
        if i == 0:
            continue
        else:
            row = data[i]
            steer = float(row[3])
            angle.append(steer)
    
    return np.array(angle)

def even_dist(data, angles):
### get distribution of steer angles and add more data to lines to even out distribution
### get the max value of a bin and bring every other bin up to 80% of that bin
    bin_val = np.linspace(min(angles), max(angles), 23)
    hist, bins = np.histogram(angles, bin_val)
    max_val = 1500
    print('Length of data before processing = ', len(data))
    new_data = []
    bin_count = 1
    for j in range(1, len(bin_val)):
        if abs(bin_val[j]) > 0.45:
            dist_factor = 10
        elif abs(bin_val[j]) > 0.3:
            dist_factor = 4
        else: 
            dist_factor = 1.5
        data_bin_count = hist[j-bin_count]
        count = 0
        prev_val = bin_val[j-bin_count]
        value  = bin_val[j]
        while count < data_bin_count*dist_factor:
            if count > max_val:
                break
            for k in range(len(data)):
                row = data[k]
                steer = float(row[3])
                if count > max_val:
                    break
                if steer <= value and steer >= prev_val:
                    new_data.append(row)
                    count += 1
    print('Length of new data after processing = ', len(new_data))
          
    return new_data
    
def flip_images(image, measurement):
    measurement = np.float(measurement)
    aug_image = cv2.flip(image,1)
    aug_measurement = measurement*-1.0

    
    return aug_image, aug_measurement


def rand_rot(image):
    image = ndimage.rotate(image, angle=random.randint(-5,5), reshape=False)
    return image

def rand_translate(image):
    x = np.random.randint(-5,5)
    y = np.random.randint(-5,5)
    num_rows, num_cols = image.shape[:2]
    translate_mat = np.float32([[1,0,x], [0,1,y]])
    #image = cv2.warpAffine(image, translate_mat, (num_cols, num_rows))
    image = cv2.warpAffine(image, translate_mat, (num_cols, num_rows))
    return image

def left_warp(image):
    rows = image.shape[0]
    cols = image.shape[1]
    img_size = (image.shape[1], image.shape[0])
    
    x1 = rows/4
    x2 = 3*rows/4
    y1 = cols/4
    y2 = 3*cols/4
      
    src_points = np.float32(
    [[x1,  y2],
     [x1,  y1],
     [x2,  y1],
     [x2, y2]])

    #left tilt
    dst_points = np.float32(
    [[x1+2,  y2-3],
     [x1+2,  y1+3],
     [x2+3,  y1],
     [x2+3, y2]])
      
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
    
def right_warp(image):
    rows = image.shape[0]
    cols = image.shape[1]
    img_size = (image.shape[1], image.shape[0])
    
    x1 = rows/4
    x2 = 3*rows/4
    y1 = cols/4
    y2 = 3*cols/4
      
    src_points = np.float32(
    [[x1,  y2],
     [x1,  y1],
     [x2,  y1],
     [x2, y2]])
    
    #right tilt
    dst_points = np.float32(
    [[x1+2,  y2-1],
     [x1+2,  y1+1],
     [x2-3,  y1+3],
     [x2-3, y2-3]])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def rand_augment(image):
    Augmentation = {
        'Random Rotation' : rand_rot,
        'Random Translate' : rand_translate,
        'Left Warp' : left_warp,
        'Right Warp' : right_warp
        }
    key = np.random.choice(list(Augmentation))
    new_image = Augmentation[key](image)
    
    return new_image

def adjust_steering(path, measurement, min_steer=0.05, max_steer=0.2):
    measurement = np.float(measurement)
    if 'left' in path:
        if np.abs(measurement) > 0.3:
            adj_meas = measurement + max_steer
        elif np.abs(measurement) > 0.1 :
            adj_meas = measurement + min_steer
        else:
            adj_meas = measurement
            
    if 'right' in path:
        if np.abs(measurement) > 0.3:
            adj_meas = measurement - max_steer
        elif np.abs(measurement) > 0.1:
            adj_meas = measurement - min_steer
        else:
            adj_meas = measurement            
    else:
        adj_meas = measurement
        
    return adj_meas

def train_data_generator(dir_name, data, batch_size, val_data=False):

    while 1:
        data_size = len(data)
        x_train = []
        y_train = []
        while len(x_train) <= batch_size-1:
            row_num = np.random.randint(data_size-1)
            row = data[row_num]
            #randomly use camera images
            #rand_camera = np.random.randint(0,3)
            og_measurement = float(row[3])
            
            
            #rand_camera = 0
            #load the image from directory using the random row and camera
            if og_measurement < -0.3: 
                camera = 2
            elif og_measurement > 0.3:
                camera = 1
            else:
                camera = 0
            source_path = row[camera]
            filename = source_path.split('/')[-1]
            image_path = dir_name + '/IMG/' + filename
            og_image = cv2.imread(image_path)
            
            if val_data:
                image = og_image
            else:
                image = rand_augment(og_image)
            #print('image path = ', image_path)
            #print('og_measurement = ', og_measurement)
            measurement = adjust_steering(image_path, og_measurement)

            
            if len(x_train) % 2 == 0:
                flip_image, flip_measurement = flip_images(image, measurement)
                x_train.append(flip_image)
                y_train.append(flip_measurement)
            else:
                x_train.append(image)
                y_train.append(measurement)
            
        yield(np.array(x_train), np.array(y_train))


def build_model(debugging=True):
    if not debugging:
        drop_rate = 0.25

        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))
        model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(drop_rate))
        model.add(Dense(50))
        model.add(Dropout(drop_rate))
        model.add(Dense(10))
        model.add(Dropout(drop_rate))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')
        #model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

        return model
def main():
    
    batch_size = 256
    dir_name = 'data'
    
    my_data = load_data(dir_name)
    steer_angles = extract_steer(my_data)
    new_data =  even_dist(my_data, steer_angles)
    
    train_data_gen = train_data_generator(dir_name, new_data, batch_size, val_data=False)
    validation_data_gen = train_data_generator(dir_name, new_data, batch_size, val_data=True)

    model = build_model(debugging=False)
    
    if model:
        model.fit_generator(train_data_gen, steps_per_epoch=15000//batch_size, validation_data=validation_data_gen, validation_steps=3000//batch_size, epochs=7  ,verbose=1)

        model.save('Test.h5')
    
    print('testing main')
    

if __name__ == '__main__':
    main()

        