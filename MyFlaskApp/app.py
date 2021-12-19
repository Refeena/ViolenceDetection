#from flask import Flask
from flask import *
import os
import cv2
import numpy as np
#import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from tensorflow.keras.applications.mobilenet import MobileNet


from PIL import Image, ImageOps
from numpy import asarray

print("Downloading the MobileNet........")
image_model = MobileNet(include_top=True, weights='imagenet')
print("Completed the Downloading of MobileNet")

in_dir_1 = "../test"
print("Directory is : " + in_dir_1)

# Frame size
img_size = 224

img_size_touple = (img_size, img_size)

# Number of frames per video
_images_per_file = 20

transfer_values_size = 1000

transfer_layer = image_model.get_layer('reshape_2')

image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)

def video_names(in_dir_1):
    # list containing video names
    names1 = []

    for current_dir, dir_names, file_names in os.walk(in_dir_1):

        for file_name1 in file_names:
            names1.append(file_name1)

            # shuffle(names1)

    return tuple(names1)

def get_frames(in_dir_1, file_name):
    in_file = os.path.join(in_dir_1, file_name)
    images = []

    vidcap = cv2.VideoCapture(in_file)

    success, image = vidcap.read()

    count = 0

    while count < _images_per_file:
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                         interpolation=cv2.INTER_CUBIC)

        images.append(res)

        success, image = vidcap.read()

        count += 1

    resul = np.array(images)

    resul = (resul / 255.).astype(np.float16)


    return resul

# helper Function
def get_transfer_values(current_dir, file_name):
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)

    image_batch = np.zeros(shape=shape, dtype=np.float16)

    image_batch = get_frames(current_dir, file_name)

    # Pre-allocate output-array for transfer-values.
    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
        image_model_transfer.predict(image_batch)

    return transfer_values

# loading the model
model = keras.models.load_model('./model')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/success', methods=['POST','GET'])
def success():

    if request.method == 'POST':

        f = request.files['file']
        print(f.filename)

        f.save(f.filename)

        frames1 = get_frames(in_dir_1, f.filename)
        visible_frame = (frames1 * 255).astype('uint8')
        # plt.imshow(visible_frame[10])
        img_input = visible_frame[10]
        def add_border(input_image, output_image, border, color=0):
            img = Image.open(input_image)
            if isinstance(border, int) or isinstance(border, tuple):
                bimg = ImageOps.expand(img, border=border , fill=color)
            else:
                raise RuntimeError('Border is not an integer or tuple!')
            bimg.save(output_image)

        Image.fromarray(img_input).save('violence_1.jpg')
        in_img = 'violence_1.jpg'

        video = get_transfer_values(in_dir_1, f.filename)
        video = video.reshape((1, video.shape[0], video.shape[1]))
        label = np.argmax(model.predict(video))

        if (label == 1):
            print('RESULT-VIOLENCE NOT PRESENT')
            status = 'VIOLENCE NOT PRESENT'
            color_br = 'green'
        else:
            print('RESULT-VIOLENCE PRESENT')
            status = 'VIOLENCE PRESENT'
            color_br = 'red'
        add_border(in_img, output_image='./static/violence_final.jpg', border=25, color=color_br)
        # load the image
        #image = np.array(Image.open('violence_1.jpg').resize((500, 500)))
        imgName = 'violence_final.jpg'
        return render_template("success.html", name=f.filename, result=status, imgPath=imgName)


if __name__ == "__main__":
        app.run()




