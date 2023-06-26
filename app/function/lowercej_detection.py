from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import misc
from matplotlib.image import imsave
import cv2
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

# load json and create model
json_file = open('drive/MyDrive/Trained_weight/model_unet512.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("drive/MyDrive/Trained_weight/model_unet_lower_cej_140822.h5")
print("Loaded model from disk")

# set the images and masks directories
data_dir = "drive/MyDrive/Data_lama_raihan/Teeth_Ori_Pic/"
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

# plt.figure(2)
# plt.imshow(predicted_mask)
# plt.show()
# cmap = plt.cm.coolwarm
# norm = plt.Normalize(vmin=predicted_mask.min(), vmax=predicted_mask.max())
# new_image = cmap(norm(predicted_mask))
# loc_file = './' + 'bottom_cej_prob_' + image_name[:-4] +'.png'
# imsave(loc_file, new_image)

plt.figure(3)
plt.imshow(np.uint8(newmask))
plt.show()
loc_file = 'drive/MyDrive/Data_lama_raihan/' + '2_bottom_cej_' + image_name
imsave(loc_file, np.uint8(newmask))