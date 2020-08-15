import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import ndimage, misc

#Load data from csv file and store row information in list 
def load_data(dir_name): 
    lines = []
    with open(dir_name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


data = load_data("Train_Data")

def extract_steer(data):
    angle = []
    for i in range(len(data)):
        row = data[i]
        steer = float(row[3])
        angle.append(steer)
    
    return np.array(angle)

steer_angle = extract_steer(data)

#print(max(steer_angle))
#print(min(steer_angle))

num_bins = np.linspace(-1.0, 1.0, 10)

hist, bins = np.histogram(steer_angle, num_bins)

plt.hist(steer_angle, num_bins)
plt.show()
                   
