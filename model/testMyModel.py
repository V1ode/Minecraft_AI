import pathlib

print(pathlib._local)

# import math
# import os
#
# import numpy as np
# import tensorflow as tf
# from keras.src.saving import load_model
#
#
# model = load_model("myModel.keras")
#
# PATH = 'C:/Users/Roma/PycharmProjects/ML_3models/static/media/'
# train_dir = os.path.join(PATH, 'train')
#
# BATCH_SIZE = 64
# IMG_SIZE = (96, 96)
#
# train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
#                                                             shuffle=True,
#                                                             batch_size=BATCH_SIZE,
#                                                             image_size=IMG_SIZE)
#
# classes = train_dataset.class_names
#
# print()
# # for filename in os.listdir("C:/Users/Roma/PycharmProjects/ML_3models/static/media/test"):
# #     img = tf.keras.utils.load_img(os.path.join("../static/media/test/", filename), target_size=IMG_SIZE)
# #     img_array = tf.keras.utils.img_to_array(img)
# #     print(img_array)
# #     img_array = tf.expand_dims(img_array, 0)
# #     print(img_array)
# #
# #     predictions = model.predict(img_array)
# #     print(predictions)
# #     predictions = predictions.flatten()
# #     predictions = int(math.floor(float(predictions[0] * len(classes))))
# #
# #
# #     print(
# #         f"{filename[:-4]} это {classes[predictions]}"
# #     )
# #
# # print("\n\n\n")
#
# img1 = tf.keras.utils.load_img(os.path.join("../static/media/test/block", "acacia-leaves.png"), target_size=IMG_SIZE)
# img2 = tf.keras.utils.load_img(os.path.join("../static/media/test/slab", "acacia-slab.png"), target_size=IMG_SIZE)
#
#
# img_array1 = tf.keras.utils.img_to_array(img1)
# img_array2 = tf.keras.utils.img_to_array(img2)
# # for i in img_array1:
# #     for j in i:
# #         print(j, "Different")
#
# img_array1 = tf.expand_dims(img_array1, 0)
# prediction = model.predict(img_array1).flatten()
# print(prediction)
# pred = tf.nn.softmax(prediction)
# print(
#     f"Это изображение похоже на {classes[np.argmax(pred)]}"
#     )
#
# img_array2 = tf.expand_dims(img_array2, 0)
# prediction2 = model.predict(img_array2).flatten()
# print(prediction2)
# pred = tf.nn.softmax(prediction2)
# print(
#     f"Это изображение похоже на {classes[np.argmax(pred)]}"
#     )