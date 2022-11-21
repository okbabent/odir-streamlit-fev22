import streamlit as st 
from app import ui, utils



MenuChoice = {
  "CNN - Tensorflow/Keras" : "A", # Convolutional Neural Networks Réseau de neurone convolutif Keras
  "Transfert learning" : "B",
}


def display_choice(menu_choice, args):
    if menu_choice == 'A':
        def choice_a():
            _, c, _ = st.columns([1,10,1])
            with c:
                img_title = 'Schéma d\'une architecture d\'un réseau neuronal convolutif'
                img_title = ui.title_label(img_title)
                st.markdown(img_title, unsafe_allow_html=True)
                img_path = utils.get_resource('assets', 'cnn-deeplearning-arc.jpeg')
                st.image(img_path, use_column_width=True)
                ui.add_vgap(5)
            _, c, _ = st.columns([1,2,1])
            with c:
                img_title = 'Schéma d\'architecture  du modèle VGG16'
                img_title = ui.title_label(img_title)
                st.markdown(img_title, unsafe_allow_html=True)
                st.image(utils.get_resource('assets', 'vgg-16-network-architecture.png'), use_column_width=True)
        return choice_a
    if menu_choice == 'B':
        def choice_b():
            _ , c= st.columns([1,8])
            with c:
                ui.add_vgap(10)
                st.markdown(
                    """
                    
                    - ## Plusieurs architectures de modèles testés par 'Transfert learning' & 'Fine Tunning'
                    - ## VGG16, VGG19, RESNET50, INCEPTION, XCEPTION depuis KERAS
                    - ## Plusieurs tentatives de réglages des hperparamères
                    - ## Utilisation de Tensor Board
                    - ## Accuracy comme métrique
                    - ## Modèles entrainés sur + de 18000 images
                    - ## Couche de sortie avec 8 unités avec une fonction d'activation Softmax/Sigmoid selon le modèle
                    """
                )
        return choice_b
    return None


def header():
    return {'id': "Modélisations", 'icon': 'boxes', 'callback': display}

def display():

    ### Create Title
    ui.slide_header("Modélisation des données", gap=(None,2,None))
    ui.sub_menus(MenuChoice, display_choice)
   

      
      



