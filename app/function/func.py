from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import misc
from matplotlib.image import imsave
import cv2
from pathlib import Path
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
parent_dir = Path("D:/Rottenfanger/project/periodontitis_portable")


def ModelWeight(num):
    if num == 1:
        return "bl_detect"
    elif num == 2:
        return "lowercej"
    elif num == 3:
        return "lowerteeth"
    elif num == 4:
        return "uppercej"
    elif num == 5:
        return "upperteeth"
    

def Detection(imagename, choice):
    # insert id from project google drive
    # roots = '1JcFiSyoWo2EspIAzG4rd8OWk8TmZbX48'
    # file_list = drive.ListFile({'q': "'1JcFiSyoWo2EspIAzG4rd8OWk8TmZbX48' in parents and trashed=false"}).GetList()
    # for file in file_list:
    #     print('Title: %s, ID: %s' % (file['title'], file['id']))

    # load json and create model
    json_model = os.listdir(f"{parent_dir}/" + "models/")[5]
    json_file = open(f"{parent_dir}/models/{json_model}", "r")

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    weight = os.listdir(f"{parent_dir}/" + "models/")[choice-1]
    loaded_model.load_weights(f"{parent_dir}/models/{weight}")
    print("Loaded model from disk")

    # set the images and masks directories
    data_dir = f"{parent_dir}/insert_here/"
    all_images = os.listdir(data_dir)

    image_name = imagename
    original_img = cv2.imread(data_dir + image_name)
    print(original_img.shape)
    resized_img = cv2.resize(original_img, (512, 512))
    print(resized_img.shape)
    array_img = img_to_array(resized_img)/255
    array_img = np.array(array_img)

    predicted_mask = loaded_model.predict(array_img[np.newaxis, :, :, :])
    predicted_mask = predicted_mask.reshape(512, 512)

    threelevelmask = np.copy(predicted_mask)
    threshold1 = 0.6

    [rows, cols] = threelevelmask.shape
    newmask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            if threelevelmask[i,j]<threshold1:
                newmask[i,j] = 0
            if (threelevelmask[i,j]>threshold1):
                newmask[i,j] = 255

    newmask = np.dstack([newmask, newmask, newmask])
    print (np.unique(newmask))

    plt.figure(1)
    plt.imshow(array_img)
    plt.show()
    # loc_file = './' + 'inp_' + image_name
    # imsave(loc_file, array_img)

    if choice in [3,5]: 
        plt.figure(2)
        plt.imshow(predicted_mask)
        plt.show()
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin=predicted_mask.min(), vmax=predicted_mask.max())
        new_image = cmap(norm(predicted_mask))
        loc_file = f'{parent_dir}/output/' + f'{choice}_' + image_name[:-4] +'.png'
        imsave(loc_file, new_image)

    else:
        plt.figure(3)
        plt.imshow(np.uint8(newmask))
        plt.show()
        loc_file = f'{parent_dir}/output/' + f'{choice}_' + image_name
        imsave(loc_file, np.uint8(newmask))