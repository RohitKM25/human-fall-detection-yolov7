import math
import os
import urllib
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from poseEstimation import image_to_pose
from tqdm import tqdm
from main import fall_detection
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

FALL = 1
NOT_FALL = 0


def get_data_from_images(folder_path):
    x = []
    y = []
    for folder in os.listdir(folder_path):
        for file in tqdm(os.listdir(folder_path + '/' + folder)):
            pose = image_to_pose(folder_path + '/' + folder + '/' + file)
            if len(pose) >= 1:
                x.append(pose)
                y.append(FALL if folder == 'fall' else NOT_FALL)
    return pd.DataFrame({'x': x, 'y': y})


def store_data(df, folder_path):
    df.to_csv(folder_path + '/data.csv', index=False)
    df.to_pickle(folder_path + '/data.pkl')


def load_data(folder_path):
    return pd.read_pickle(folder_path + '/data.pkl')


def check_model_on_data(from_images=False):
    if from_images:
        data = get_data_from_images('fall_dataset/old')
        store_data(data, 'fall_dataset/data')
    else:
        data = load_data('fall_dataset/data')
    dataset = data.values
    x = dataset[:, 0]
    y = dataset[:, 1]
    y_predict = []
    for i in range(len(x)):
        y_predict.append(int(fall_detection(x[i])[0]))

    y_predict = list(y_predict)
    y = list(y)

    matplotlib.use('TkAgg')

    cm = confusion_matrix(y, y_predict)

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[False, True])

    cm_display.plot()
    plt.show()
    precision = precision_score(y, y_predict)
    print('Precision: %f' % precision)
    recall = recall_score(y, y_predict)
    print('Recall: %f' % recall)
    f1 = f1_score(y, y_predict)
    print('F1 score: %f' % f1)


# Predict if the person in the image is falling or not. The image must be a web-url.
def predict(image_url):
    urllib.request.urlretrieve(image_url, "image.png")
    poses = image_to_pose("image.png")
    for pose in poses:
        if fall_detection(pose):
            return True
    return False


# predict('https://st2.depositphotos.com/1000393/9807/i/950/depositphotos_98078022-stock-photo-man-falling-down.jpg')
check_model_on_data(False)
# clean_images('fall_dataset/images/fall', 'fall_dataset/images/cropped_images')
# clean_images('fall_dataset/images/not-fall', 'fall_dataset/images/cropped_images')
