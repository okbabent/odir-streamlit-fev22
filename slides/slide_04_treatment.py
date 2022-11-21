import streamlit as st 
from app import ui, utils
from app import load_dataset
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import functools


# Load dataframe
df = load_dataset.read_odir_data()


MenuChoice = {
  "Traitement des mots clés" : "A",
  "Mots clés diagnostic" : "B",
  "Traitement des images" : "C",
}

def split_diag_key_words(pdSeries):
  replacements = ('，', ',')  
  left_diags = pdSeries.apply(lambda c: functools.reduce(lambda s, sep: s.replace(sep, ';'), replacements, c)).str.split(';',  expand=True).stack().reset_index(drop=True)
  return left_diags

def key_words_by_diganostic(diag_label, top=10):
    tmp1 = split_diag_key_words(df[(df[diag_label]==1)]['Left-Diagnostic Keywords'])
    tmp2 = split_diag_key_words(df[(df[diag_label]==1)]['Right-Diagnostic Keywords'])
    df_keywords = pd.concat([tmp1, tmp2], ignore_index=True)
    if top > 0:
        return df_keywords.value_counts().nlargest(top), df_keywords.value_counts().shape[0], df_keywords.shape[0]
    return df_keywords.value_counts(), df_keywords.value_counts().shape[0], df_keywords.shape[0]

def dignosis_key_words_pie(diag_label):
    top = 20
    data, unique, total = key_words_by_diganostic(diag_label[0], top)
    data = pd.DataFrame({'Diagnostic Keywords':data.index, 'count':data.values})
    fig = plt.figure(figsize=(20, 10))
    #sns.set_theme(style="white", palette="bright", font="arial", font_scale= 1.5)
 
    title = ''
   
    if top > 0:
        title = f'Top {top} de la répartition des mots clés diagnostic pour ({diag_label})'
    else:
        title = f'Répartition des mots clés diagnostic'
    # suptitle = plt.suptitle(title, fontsize = 14)
    # suptitle.set_fontweight('bold')
    # suptitle.set_fontname('serif')
    # suptitle.set_color(ui.color("blue-green-100"))
    # 
   #title = plt.title(f'Nombre unique de mots clés diagnostic = {unique}', pad=2, loc='center', fontsize = 14, fontname='sans-serif')
    # title.set_fontweight('bold')
    # title.set_fontstyle('italic')
    st.markdown(ui.title_label(title), unsafe_allow_html=True)
    st.markdown(ui.subtitle_label(f'Nombre unique de mots clés diagnostic = {unique}'), unsafe_allow_html=True)
    
    plt.pie(x=data['count'], labels = data["Diagnostic Keywords"], autopct='%.0f%%')
    st.pyplot(fig)
      

def diagnosis_fig_tab(tab, diag_label):
  with tab:
    _ , c, _ = st.columns([1,2,1])
    with c:
        ui.add_vgap(2)
        dignosis_key_words_pie(diag_label)



 


def diagnosis_key_words():
    diagnosis_labels = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others', 'Récap']
 
    tabs = st.tabs(diagnosis_labels)
  
    for i, tab in enumerate(tabs[:-1]):
        diagnosis_fig_tab(tab, diagnosis_labels[i])
    with tabs[-1]:
        # ui.add_vgap(10)
        _ , c, _ = st.columns([1,2,1])
        with c:
            ui.add_vgap(2)
            # dd = pd.DataFrame(data=np.array([[3500, 6066, 5905, 6279]]), columns=['Original', 'OB', 'YB', 'TV'])
            # st.dataframe(dd)
            #fig = plt.figure(figsize=(10, 4))
            #sns.set_style('whitegrid',{'grid.linestyle': ':'})
            sns.set_theme(style="white", palette=None, font_scale= 1.5)
            fig = plt.figure(figsize=(20, 10))
            # ax = fig.add_axes([0,0,1,1])
            # ax.bar(['Original', 'OB', 'YB', 'TV'], [3500, 6066, 5905, 6279])
            x = ['Original', 'OB', 'YB', 'TV']
            y = [3500, 6066, 5905, 6279]
            sns.barplot(x=x, y=y)
            # ax.grid(False)
            # ax.axis('off')
            st.markdown(ui.title_label('Répartition du nombre de données par fond d\'oeil'), unsafe_allow_html=True)
            st.pyplot(fig)

    



# def diagnosis_key_words_tab(tab, fn, in_column=True):
#   with tab:
#     if in_column:
#       _ , c, _ = st.columns([1,2,1])
#       with c:
#         fn()
#     else:
#       fn()

def display_choice(menu_choice, args):
    if menu_choice == 'A':
        def choice_a():
            _ , c= st.columns([1,8])
            with c:
                ui.add_vgap(15)
                st.markdown(
                    """
                    
                    - ##  Identifier le nombre de combinaisons de labels possibles et leur poids par rapport à l'ensemble des données
                    - ## Identifier l'ensemble des mots-clés diagnostic existants
                    - ## Ré-attribuer chaque label à leur(s) mot(s)-clé(s) correspondant 
                    - ##  Séparation des données par fond d'œil et non plus par patient.
                    """
                )
        return choice_a

    if menu_choice == 'B':
        def choice_b():
            diagnosis_key_words()
        return choice_b

    if menu_choice == 'C':
        def choice_c():
            _ , c, _= st.columns([1,6,1])
            with c:
                svg = utils.get_resource('assets', 'processing-images.png')
                img_title = ui.title_label('Processus de traitement et de générationd es données images')
                st.markdown(img_title, unsafe_allow_html=True)
                ui.add_vgap(2)
                st.image(svg, use_column_width=True)
                ui.add_vgap(5)
            _, c1, c2,_ = st.columns([1,2,2,1])
            with c1:
                img_title = ui.title_label('Exemple de données augmentées')
                st.markdown(img_title, unsafe_allow_html=True)
                st.image(utils.get_resource('assets', 'post-process-img.png'))
            with c2:
                img_title = ui.title_label('Exemple de données augmentées')
                st.markdown(img_title, unsafe_allow_html=True)
                st.image(utils.get_resource('assets', 'post-process-img-2.png'))
        return choice_c

def header():
    return {'id': "Traitement des données", 'icon': 'bar-chart', 'callback': display}

def display():
    ### Create Title
    ui.slide_header("Traitement des données", gap=(5,10))
    ui.sub_menus(MenuChoice, display_choice)
   
   

      
      



