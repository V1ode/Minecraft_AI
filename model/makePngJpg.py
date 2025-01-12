import os
from PIL import Image

for filename in os.listdir("C:/Users/Roma/PycharmProjects/ML_3models/model"):
    if filename[-4:] == ".png":
        img = Image.open(filename)
        img = img.convert("RGB")
        img.save(filename[:-4]+".jpg", format="JPEG")
        os.remove(filename)
