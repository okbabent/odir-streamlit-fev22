import streamlit as st 
from app import ui, utils
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.patches as patches
import pandas as pd

from PIL import Image
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from typing import Callable, List, NamedTuple, Tuple
import altair as alt
import tensorflow as tf
from tensorflow import keras
from keras.applications import (vgg19, xception)
from tempfile import NamedTemporaryFile
# from tf.keras.preprocessing.image import img_to_array, load_img
# import tf.keras.backend.tensorflow_backend as tb

# Hack
# I get a '_thread._local' object has no attribute 'value' error without this
# See https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
# tb._SYMBOLIC_SCOPE.value = True  # pylint: disable=protected-access

DISEASE_CLASSES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

IMAGE_TYPES = ["png", "jpg"]

MenuChoice = {
  "Métriques" : "A",
  "Matrices de confusions" : "B",
  "Inference" : "C"
}

class ImageUtils:

    @staticmethod
    def remove_black_pixels(image):
        # Creation d'un masque avec les pixels colorés
        mask = image > 0

        # Coordonnes des pixels avec une couleur
        coordinates = np.argwhere(mask)

        # Definition du rectangle contenant les pixels colorés
        x0, y0, s0 = coordinates.min(axis=0)
        x1, y1, s1 = coordinates.max(axis=0) + 1  # slices are exclusive at the top

        # Decoupage de l'image selon le rectangle
        cropped = image[x0:x1, y0:y1]

        return cropped

    
    @staticmethod
    def equalize_image(rgb_img):
        # conversion de l'image du format RGB au format YcrCb
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

        # equalization sur le canal Y
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # conversion de l'image du format YcrCb au format RBG
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        return equalized_img


    @staticmethod
    def resize_image(img, image_width, keep_ratio = False, flip = False):

        # Possibilité de garder le ratio
        if keep_ratio:
            width_perc = image_width/float(img.size[0])
            height_size = int((float(img.size[1]) * float(width_perc)))
            img = cv2.resize(img, (image_width , height_size))
        else:
            img = cv2.resize(img, (image_width , image_width ))

        # Permutation de l'image droit/gauche de l'oeil droit
        if flip:
            img.transpose(Image.FLIP_LEFT_RIGHT)

        return img

    @staticmethod
    def preprocess_image(image, image_width, equalize=False):
        # loading file
        # image = cv2.imread(join(folder_src, file))
        # cropping
        image = ImageUtils.remove_black_pixels(image)

        if equalize:
            # equalize
            image = ImageUtils.equalize_image(image)

        # resize
        image = ImageUtils.resize_image(image, image_width)
        return image

class XceptionFT:
    image_width = (299, 299)

    @staticmethod
    def preprocess_image(image):
        return ImageUtils.preprocess_image(image, XceptionFT.image_width[0])


    def decode_predictions(predictions):
        return predictions

    @staticmethod
    def predict(model, image, input_shape):
        img = tf.keras.applications.xception.preprocess_input(image)
        img = model.predict(img.reshape(1,input_shape[0],input_shape[1],3))
        return img


class Vgg19:

    image_width = (224, 224)
    CLASS_INDEX = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    # CLASS_INDEX = range(8)

    @staticmethod
    def preprocess_image(image):
        return ImageUtils.preprocess_image(image, Vgg19.image_width[0], True)

    def decode_predictions(predictions):
        return predictions

    @staticmethod
    def predict(model, image, input_shape):
        res = model.predict(image.reshape(1,input_shape[0],input_shape[1],3))
        return res



class Inference(NamedTuple):

    name: str
    model_h5: str
    # application: Callable
    input_shape: Tuple[int, int]
    preprocess_input_func: Callable
    predictions_func: Callable
    decode_predictions_func: Callable


    @st.cache(allow_output_mutation=True)
    def get_model(self) -> object:
        """The Keras model with weights="imagenet"

        Returns:
            [object] -- An instance of the keras_application with weights="imagenet"
        """
        model_file_name = utils.get_resource('data/models', self.model_h5)
        model = None
        try:
            model = keras.models.load_model(model_file_name)
        except:
            print('No model')
        return model

    def preprocess_input(self, image: Image) -> Image:
        """Prepares the image for classification by the classifier

        Arguments:
            image {Image} -- The image to preprocess

        Returns:
            Image -- The preprocessed image
        """
        # For an explanation see
        # https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
        # image = self.to_input_shape(image)
        # image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        image = self.preprocess_input_func(image)
        return image


    def get_top_predictions(self, image: Image = None, report_progress_func=print) -> List[Tuple[str, str, float]]:
        """[summary]

        Keyword Arguments:
            image {Image} -- An image (default: {None})
            report_progress_func {Callable} -- A function like 'print', 'st.write' or similar
            (default: {print})

        Returns:
            [type] -- The top predictions as a list of 3-tuples on the form
            (id, prediction, probability)
        """
        report_progress_func(
            f"Loading {self.name} model ... (The first time this is done it may take several "
            "minutes)",
            10,
        )
        model = self.get_model()
        if model is  None:
            return []

        report_progress_func(f"Processing image ... ", 67)
        image = self.preprocess_input(image)

        report_progress_func(f"Classifying image with '{self.name}'... ", 85)
        predictions = self.predictions_func(model, image, self.input_shape)
        top_predictions = self.decode_predictions_func(predictions)

        report_progress_func("", 0)

        return top_predictions[0]

    
    @staticmethod
    def to_main_prediction_string(predictions) -> str:
        idx = predictions.argmax(axis = -1)
        label, prob = DISEASE_CLASSES[idx], predictions[idx]
        return f"{label.capitalize()} avec une probabilité de {str(prob*100).format('.2f')}"

    @staticmethod
    def to_predictions_chart(predictions):
        
        fig = plt.figure(figsize=(20, 15))
        sns.set_theme(style="white", palette=None, font_scale= 2.5)
        x = DISEASE_CLASSES
        y = predictions*100

        plt.barh(x, y)
        plt.xlim([0, 100])

        idx = predictions.argmax(axis = -1)
        label, prob = DISEASE_CLASSES[idx], predictions[idx]
    
        t = f"Prédiction : {label.capitalize()} avec une probabilité de {round(prob*100,2)}%"
        fgc = ui.color("green-100")
        plt.title(t, color=fgc)


        return fig
     
ODIR_APPLICATIONS: List[Inference] = [
     Inference(
        name="vgg19",
        input_shape = (224, 224),
        model_h5='model_17_10_vgg19_a.h5',
        preprocess_input_func=Vgg19.preprocess_image,
        predictions_func = Vgg19.predict,
        decode_predictions_func=Vgg19.decode_predictions,
    ),
        Inference(
        name="Xception with fine tune",
        input_shape = (299, 299),
        model_h5='odir_model_weights_Xception_2022_10_21_multiclass_fine_tuning.h5',
        preprocess_input_func=XceptionFT.preprocess_image,
        predictions_func = XceptionFT.predict,
        decode_predictions_func=XceptionFT.decode_predictions)]

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)

def upload_image_and_predict(selected_model):
    
    c1, c2 = st.columns([1,1])
    predictions = None
    prediction_done = False
    with c1:
        uploadedFile = st.file_uploader("Chargez une image pour la classification", IMAGE_TYPES)
        if uploadedFile:
            st.image(uploadedFile, use_column_width=True)
        
            # open the image
            image = Image.open(uploadedFile)
            # convert to cv2
            cv2_img = np.array(image)
            image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
            progress_bar = st.empty()
            progress = st.empty()

            def report_progress(message, value, progress=progress, progress_bar=progress_bar):
                if value == 0:
                    progress_bar.empty()
                    progress.empty()
                else:
                    progress_bar.progress(value)
                    progress.markdown(message)
            predictions = selected_model.get_top_predictions(image=image, report_progress_func=report_progress)
            prediction_done = True
    with c2:
        # st.subheader("Main Prediction")
        # main_prediction = selected_model.to_main_prediction_string(predictions)
        # st.write(main_prediction)
        if prediction_done:
            # st.subheader(f"Predictions")
            if (len(predictions) > 0):
                idx = predictions.argmax(axis = -1)
                label, prob = DISEASE_CLASSES[idx], predictions[idx]
                t = f"Prédiction : {label.capitalize()} avec une probabilité de {round(prob*100,2)}%"
                fgc = ui.color("green-100")
                
                st.markdown(ui.title_label(t), unsafe_allow_html=True)
                predictions_chart = selected_model.to_predictions_chart(predictions)
                st.pyplot(predictions_chart)
        # st.altair_chart(predictions_chart)



def display_choice(menu_choice, args):
    if menu_choice == 'A':
        def choice_a():
            models = ['VGG16', 'VGG19', 'XCEPTION', 'INCEPTION', 'RESNET', 'XCEPTION-FT']
            x = models
            loss = [0.2742179334, 0.2896939516, 0.3918916285, 0.4019442201, 0.4535985589, 0.6573]
            accuracy = [0.9765625, 0.9796984792, 0.9744443297, 0.9728212953, 0.9625055194, 0.7532]
            f1_score = [0.9765625, 0.9796985035, 0.9744443222, 0.9728213028, 0.9625055018, 0.7475]
            fig = plt.figure()
            plt.plot(x, loss, '-bo', label='Loss')
            plt.plot(x, accuracy, 'go-', label='Accuracy')
            plt.plot(x, f1_score, 'r--',  label='F1-score')
            plt.legend()
            img_title = ui.title_label('Métriques pour les différents modèles implémentés')
            _,c,_ = st.columns([1,2,1])
            with c:
                st.markdown(img_title, unsafe_allow_html=True)
                st.pyplot(fig)
        return choice_a
    if menu_choice == 'B':
        def choice_b():
            _, c1, c2, _ = st.columns([1,2,2,1])
            with c1:
                img_title = 'Matrice de confusion - Xception-FT -'
                img_title = ui.title_label(img_title)
                st.markdown(img_title, unsafe_allow_html=True)
                st.image(utils.get_resource('assets', 'matrice_de_confusion_xception_ft_ob.png'), use_column_width=True)
            # _, c2, _ = st.columns([1,2,1])
            with c2:
                img_title = 'Matrice de confusion - RESNET -'
                img_title = ui.title_label(img_title)
                st.markdown(img_title, unsafe_allow_html=True)
                st.image(utils.get_resource('assets', 'matrice_de_confusion_resnet_yb.jpg'), use_column_width=True)
            _, c2, _ = st.columns([1,2,1])
            with c2:
                ui.add_vgap(2)
                img_title = 'Exemple de prédiction avec Xception-FT'
                img_title = ui.title_label(img_title)
                st.markdown(img_title, unsafe_allow_html=True)
                st.image(utils.get_resource('assets', 'predictions.png'), use_column_width=True)
            
        return choice_b

    if menu_choice == 'C':
        def choice_c():

            selected_model = st.sidebar.selectbox(
            "Sélectionne un modèle de classification d'image",
            options=ODIR_APPLICATIONS,
            index=0,
            format_func=lambda x: x.name)
            # st.sidebar.markdown(get_resources_markdown(selected_model))
            upload_image_and_predict(selected_model)

        return choice_c
    return None





def header():
    return  {'id': "Analyse et Performance des modèles", 'icon': 'graph-up', 'callback': display}


def display():

    ### Create Title
    ui.slide_header("Analyse des performance des modèles", gap=(None, 10, None))
    ui.sub_menus(MenuChoice, display_choice)
   

      
      




