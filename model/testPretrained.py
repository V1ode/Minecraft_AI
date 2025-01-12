import math
import os
import tensorflow as tf
from keras.src.saving import load_model


model = load_model("../saved_models/pretrainedModel30.keras")

PATH = 'C:/Users/Roma/PycharmProjects/ML_3models/static/media/'
train_dir = os.path.join(PATH, 'train')

BATCH_SIZE = 32
IMG_SIZE = (96, 96)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

classes = train_dataset.class_names

print()
for folder in os.listdir("C:/Users/Roma/PycharmProjects/ML_3models/static/media/test"):
    if not os.path.isfile(folder):
        for filename in os.listdir(os.path.join("C:/Users/Roma/PycharmProjects/ML_3models/static/media/test", folder)):
            img = tf.keras.utils.load_img(os.path.join("C:/Users/Roma/PycharmProjects/ML_3models/static/media/test", folder, filename), target_size=(96, 96))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array).flatten()
            # predictions = tf.nn.sigmoid(predictions)
            print(predictions[0], end="   ")
            predictions = int(math.floor(float(predictions[0] * len(classes))))


            print(
                f"{filename[:-4]} это {classes[predictions]}"
            )

print("\n\n\n")