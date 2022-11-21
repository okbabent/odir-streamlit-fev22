import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
from app import load_dataset
import functools
from nltk.tokenize import PunktSentenceTokenizer
from wordcloud import WordCloud
from PIL import Image
from app import utils
from app import ui
from bokeh.plotting import figure
from bokeh.palettes import Category20c
from bokeh.layouts import row
import math


# st.markdown(
#     """
# <style>
# .reportview-container .markdown-text-container {
#     font-family: monospace;
# }
# .sidebar .sidebar-content {
#     background-image: linear-gradient(#2e7bcf,#2e7bcf);
#     color: white;
# }
# .Widget>label {
#     color: white;
#     font-family: monospace;
# }
# [class^="st-b"]  {
#     color: red;
#     font-family: monospace;
# }
# .st-bb {
#     background-color: transparent;
# }
# .st-at {
#     background-color: #0c0080;
# }
# .st-af {
#   font-size: 1.5rem;
# }
# footer {
#     font-family: monospace;
# }
# .reportview-container .main footer, .reportview-container .main footer a {
#     color: #0c0080;
# }
# header .decoration {
#     background-image: none;
# }

# </style>
# """,
#     unsafe_allow_html=True,
# )

DEFAULT_NUMBER_OF_ROWS = 5
DEFAULT_NUMBER_OF_COLUMNS = 5
DIAGNOSTIC_COL_NAMES = ['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']

def load_df():
  return load_dataset.read_odir_data()
  
def load_post_processed_df():
  dict1 = {0 : 'Normal',
        1 : 'Diabetes',
        2 : 'Glaucoma',
        3 : 'Cataract',
        4 : 'AMD',
        5 : 'Hypertension',
        6 : 'Myopia',
        7 : 'Others'}
  dict2 = {"C" : 'Cataract',
        "D" : 'Diabetes',
        "G" :'Glaucoma',
        "C" :'Cataract',
        "A" :'AMD',
        "M" :'Myopia',
        "N" :'Normal',
        "H" :'Hypertension',
        "O" :'Others'}
  df_OB = load_dataset.read_csv_data('df_OB.csv', 'Label', dict1) #Dataset Okba
  df_YB = load_dataset.read_csv_data('df_YB.csv', 'Diag', dict2) #Dataset Yannick
  df_TV = load_dataset.read_csv_data('df_TV.csv') #Dataset Thibaut
  # Order column as me
  # columns = df_OB.columns
  # print('OB',df_OB.head())
  # print('df_TV',df_TV.head())
  # print('df_YB',df_YB.head())
  # df_TV = df_TV.loc[:columns]
  # df_YB = df_YB.loc[:columns]


  return df_OB, df_YB, df_TV


df = load_df()

df_OB, df_YB, df_TV = load_post_processed_df()


def split_diag_key_words(pdSeries):
  replacements = ('，', ',')  
  left_diags = pdSeries.apply(lambda c: functools.reduce(lambda s, sep: s.replace(sep, ';'), replacements, c)).str.split(';',  expand=True).stack().reset_index(drop=True)
  return left_diags

left_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[0]])
right_diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[1]])

left_diag_keys_str = ','.join(left_diag_keys)
right_diag_keys_str = ','.join(right_diag_keys)


  
  #.str.split('，', expand=True).stack().reset_index(drop=True)
  #right_diags = df.right_diag_key.str.split('，', expand=True).stack().reset_index(drop=True)


#@st.cache(suppress_st_warning=True)
def word_cloud():

  mask_left = np.array(Image.open(utils.get_resource('assets', 'Leftbw2.jpg')))
  mask_right = np.array(Image.open(utils.get_resource('assets', 'Rightbw2.jpg')))
  wc_left = WordCloud(background_color="black", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_left)
  wc_right = WordCloud(background_color="black", max_words=1000, max_font_size=90, collocations=False, random_state=42, mask=mask_right)
  fig, ax = plt.subplots(1,2,figsize=(15,10))
  left=wc_left.generate(left_diag_keys_str) 
  right=wc_right.generate(right_diag_keys_str)
  fig.set_facecolor('black')
  fig.tight_layout(pad=2.0)
  alpha = 1
  fig.set_alpha(alpha)
  ax[1].set_facecolor('black')
  ax[0].set_facecolor('black')
  ax[1].set_alpha(alpha)
  ax[0].set_alpha(alpha)
  ax[0].imshow(wc_left) 
  ax[0].set_title('Left eye fundus')
  ax[1].imshow(wc_right) 
  ax[1].set_title('Right eye fundus')
  ax[0].grid(False)
  ax[1].grid(False)
  ax[0].axis('off')
  ax[1].axis('off')
  st.pyplot(fig)

def set_styles(results):
    table_styles = [
        dict(
            selector="table",
            props=[("font-size", "150%"), ("text-align", "center"), ("color", "red")],
        ),
        dict(selector="caption", props=[("caption-side", "bottom")]),
    ]
    return (
        results.style.set_table_styles(table_styles)
        .set_properties(**{"background-color": "blue", "color": "white"})
        .set_caption("This is a caption")
    )

@st.cache
def _filter_results(results, number_of_rows, number_of_columns) -> pd.DataFrame:
    return results.iloc[0:number_of_rows, 0:number_of_columns]

def filter_results(results, number_of_rows, number_of_columns, style) -> pd.DataFrame:
    filter_table = _filter_results(results, number_of_rows, number_of_columns)
    if style:
        filter_table = set_styles(filter_table)
    return filter_table

def select_number_of_rows_and_columns(results: pd.DataFrame, key: str, select_rows=True, select_columns=True, select_style=True, default_number_of_rows=5, default_number_of_col=5):
    rows = default_number_of_rows
    columns = default_number_of_col
    style= False
    if select_rows:
      rows = st.selectbox(
          "Selectionnez le nombre de ligne à affciher",
          options = [a*100 for a in range(len(results))],
          #options=[5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, len(results)],
          key=key+'_rows',
      )
    if select_columns:
      columns = st.slider(
          "Selectionnez le nombre de colonne à afficher",
          0,
          len(results.columns) - 1,
          default_number_of_col,
          key=key+'_columns',
      )
    if select_style:
      style = st.checkbox("Style dataframe?", False, key=key)
    return rows, columns, style

def missing_values(df):
    flag=0
    for col in df.columns:
            if df[col].isna().sum() > 0:
                flag=1
                missing = df[col].isna().sum()
                portion = (missing / df.shape[0]) * 100
                st.text(f"'{col}': nombre de donnée manquante '{missing}' ==> '{portion:.2f}%'")
    if flag==0:
        st.success("Le dataset ne contient aucune donnée manquante.")


def header():
  return {'id': "Exploration des données", 'icon': 'binoculars', 'callback': display}

def get_keywork_table(df, side):
    diag_keys = None
    side_index = side
    diag_keys = split_diag_key_words(df[DIAGNOSTIC_COL_NAMES[side_index]]).unique()
    data = pd.DataFrame(diag_keys, columns=[DIAGNOSTIC_COL_NAMES[side_index]])
    line_to_plot = st.slider("Selectionez le nombre de mot-clé à afficher", min_value=1, max_value = data.shape[0], value=15)
    table = data.head(line_to_plot)
    return table

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def exploration1():
  ui.add_vgap(5)
  sns.set_style('whitegrid',{'grid.linestyle': ':'})

  fig, ax = plt.subplots(1,2,sharey=True,figsize=(25,20))
  fig.subplots_adjust(wspace=0.05)
  #fig.suptitle("\nDistribution de la population en fonction de l'âge : \npopulation totale (gauche) ou répartie selon le sexe (droite)", fontsize=13, fontweight="bold", y=0.02)
  fig.suptitle("\nDistribution de la population en fonction de l'âge", fontsize=25, fontweight="bold", y=0.02)
  ax[0].set_title("\nPopulation totale\n").set_fontsize(25)
  ax[1].set_title("\nRépartie selon le sexe\n").set_fontsize(25)
  b=sns.kdeplot(ax=ax[0],x='Patient Age', data=df, legend=False, color='black',linewidth=2.5, fill=True, alpha=.1)
  c=sns.kdeplot(ax=ax[1],x='Patient Age', hue='Patient Sex', data=df, legend=True, palette=['m', 'c'], linewidth=2.5)
  st.pyplot(fig)

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def exploration2():
  ui.add_vgap(5)
  sns.set_style('whitegrid',{'grid.linestyle': ':'})

  fig, ax = plt.subplots(4,4, sharex=True, sharey=True, figsize=(18,18))
  fig.subplots_adjust(hspace=0.4,wspace=0.06)
  fig.suptitle("Age-related distribution of each label based on \nthe overall population distribution (red & green) or based on the patient's sex (blue & purple)",  fontsize=18, fontweight="bold", y=0.95)

  ax1=sns.kdeplot(ax=ax[0, 0], x='Patient Age', hue='N', data=df, palette=['r', 'g'], linewidth=2)
  ax1.legend(labels=['Normal', 'non-Normal'])
  ax1.text(15, 0.02,'67.4 %', fontsize=9, color='red',  weight='bold')
  ax1.text(10, 0.01,'32.6 %', fontsize=9, color='green',  weight='bold')

  ax2=sns.kdeplot(ax=ax[0, 2], x='Patient Age', hue='D', data=df, palette=['r', 'g'], linewidth=2)
  ax2.legend(loc='upper left', labels=['Diabetes', 'non-Diabetes'])
  ax2.text(15, 0.02,'67.8 %', fontsize=9, color='red',  weight='bold')
  ax2.text(10, 0.01,'32.2 %', fontsize=9, color='green',  weight='bold')

  ax3=sns.kdeplot(ax=ax[1, 0], x='Patient Age', hue='O', data=df, palette=['r', 'g'], linewidth=2)
  ax3.legend(loc='upper left', labels=['Others', 'non-Others'])
  ax3.text(15, 0.02,'72.0 %', fontsize=9, color='red',  weight='bold')
  ax3.text(10, 0.01,'28.0 %', fontsize=9, color='green',  weight='bold')

  ax4=sns.kdeplot(ax=ax[1, 2], x='Patient Age', hue='G', data=df, palette=['r', 'g'], linewidth=2)
  ax4.legend(loc='upper left', labels=['Glaucoma', 'non-Glaucoma'])
  ax4.text(15, 0.02,'93.9 %', fontsize=9, color='red',  weight='bold')
  ax4.text(50, 0.004,'6.1 %', fontsize=9, color='green',  weight='bold')

  ax5=sns.kdeplot(ax=ax[2, 0], x='Patient Age', hue='C', data=df, palette=['r', 'g'], linewidth=2)
  ax5.legend(loc='upper left', labels=['Cataract', 'non-Cataract'])
  ax5.text(15, 0.02,'93.9 %', fontsize=9, color='red',  weight='bold')
  ax5.text(50, 0.004,'6.1 %', fontsize=9, color='green',  weight='bold')

  ax6=sns.kdeplot(ax=ax[2, 2], x='Patient Age', hue='M', data=df, palette=['r', 'g'], linewidth=2)
  ax6.legend(loc='upper left', labels=['Myopia', 'non-Myopia'])
  ax6.text(15, 0.02,'95.0 %', fontsize=9, color='red',  weight='bold')
  ax6.text(50, 0.004,'5.0 %', fontsize=9, color='green',  weight='bold')

  ax7=sns.kdeplot(ax=ax[3, 0], x='Patient Age', hue='A', data=df, palette=['r', 'g'], linewidth=2)
  ax7.legend(loc='upper left', labels=['AMD', 'non-AMD'])
  ax7.text(10, 0.015,'95.3 %', fontsize=9, color='red',  weight='bold')
  ax7.text(50, 0.004,'4.7 %', fontsize=9, color='green',  weight='bold')

  ax8=sns.kdeplot(ax=ax[3, 2], x='Patient Age', hue='H', data=df, palette=['r', 'g'], linewidth=2)
  ax8.legend(loc='upper left', labels=['Hypertension', 'non-Hypertension'])
  ax8.text(15, 0.02,'97.1 %', fontsize=9, color='red',  weight='bold')
  ax8.text(50, 0.004,'2.9 %', fontsize=9, color='green',  weight='bold')

  ax1.set_title("General population \n('Normal' label)", fontdict = {'fontweight':'semibold'})
  ax2.set_title("General population \n('Diabetes' label)", fontdict = {'fontweight':'semibold'})
  ax3.set_title("General population \n('Others' label)", fontdict = {'fontweight':'semibold'})
  ax4.set_title("General population \n('Glaucoma' label)", fontdict = {'fontweight':'semibold'})
  ax5.set_title("General population \n('Cataract' label)", fontdict = {'fontweight':'semibold'})
  ax6.set_title("General population \n('Myopia' label)", fontdict = {'fontweight':'semibold'})
  ax7.set_title("General population \n('AMD' label)", fontdict = {'fontweight':'semibold'})
  ax8.set_title("General population \n('Hypertension' label)", fontdict = {'fontweight':'semibold'})

  ax11=sns.kdeplot(ax=ax[0, 1], x='Patient Age', hue='Patient Sex', data=df[df.N == 1], legend=True, palette=['c', 'm'], linewidth=2)
  ax21=sns.kdeplot(ax=ax[0, 3], x='Patient Age', hue='Patient Sex', data=df[df.D == 1], legend=True, palette=['c', 'm'], linewidth=2)
  ax31=sns.kdeplot(ax=ax[1, 1], x='Patient Age', hue='Patient Sex', data=df[df.O == 1], legend=True, palette=['c', 'm'], linewidth=2)
  ax41=sns.kdeplot(ax=ax[1, 3], x='Patient Age', hue='Patient Sex', data=df[df.G == 1], legend=True, palette=['m', 'c'], linewidth=2)
  ax51=sns.kdeplot(ax=ax[2, 1], x='Patient Age', hue='Patient Sex', data=df[df.C == 1], legend=True, palette=['m', 'c'], linewidth=2)
  ax61=sns.kdeplot(ax=ax[2, 3], x='Patient Age', hue='Patient Sex', data=df[df.M == 1], legend=True, palette=['m', 'c'], linewidth=2)
  ax71=sns.kdeplot(ax=ax[3, 1], x='Patient Age', hue='Patient Sex', data=df[df.A == 1], legend=True, palette=['c', 'm'], linewidth=2)
  ax81=sns.kdeplot(ax=ax[3, 3], x='Patient Age', hue='Patient Sex', data=df[df.H == 1], legend=True, palette=['m', 'c'], linewidth=2)

  ax11.title.set_text("Sex-related \n('Normal' label)")
  ax21.title.set_text("Sex-related \n('Diabetes' label)")
  ax31.title.set_text("Sex-related \n('Others' label)")
  ax41.title.set_text("Sex-related \n('Glaucoma' label)")
  ax51.title.set_text("Sex-related \n('Cataract' label)")
  ax61.title.set_text("Sex-related \n('Myopia' label)")
  ax71.title.set_text("Sex-related \n('AMD' label)")
  ax81.title.set_text("Sex-related \n('Hypertension' label)")
 
  st.pyplot(fig)

# # @st.cache(suppress_st_warning=True, allow_output_mutation=True)
# def exploration4():
#   st.code(f'Original - nombre de ligne : {df.shape[0]}\n'
#   f'OB - nombre de ligne :  {df_OB.shape[0]}\n'
#   f'TV - nombre de ligne :  {df_TV.shape[0]}\n'
#   f'YB - nombre de ligne :  {df_YB.shape[0]}')

 
#   # df_OB['diagnosis'] = df_OB['Label'].replace(dict)
#   # df_YB['diagnosis'] = df_YB['Diag'].replace(dict)
#   sns.set_style('whitegrid',{'grid.linestyle': ':'})

#   fig, ax = plt.subplots(1,3, sharey=True,figsize=(10,3), squeeze=False)
#   fig.subplots_adjust(wspace=0.05)

#   a=sns.countplot(ax=ax[0,0], x='diagnosis', data=df_OB)
#   a.tick_params(axis='x', labelrotation=90)
#   b=sns.countplot(ax=ax[0,1], x='diagnosis', data=df_YB)
#   b.tick_params(axis='x', labelrotation=90)
#   c=sns.countplot(ax=ax[0,2], x='diagnosis', data=df_TV)
#   c.tick_params(axis='x',labelrotation=90)


#   ax[0,0].set_title('OB')
#   ax[0,0].set_xlabel('')
#   ax[0,0].set_ylabel('')

#   ax[0,1].set_title('YB')
#   ax[0,1].set_xlabel('')
#   ax[0,1].set_ylabel('')

#   ax[0,2].set_title('TV')
#   ax[0,2].set_xlabel('')
#   ax[0,2].set_ylabel('')

#   st.pyplot(fig)

  
def exploration3():
  
  ui.add_vgap(5)
  # df_OB = load_dataset.read_csv_data('df_OB.csv') #Dataset Okba
  # df_YB = load_dataset.read_csv_data('df_YB.csv') #Dataset Yannick
  # df_TV = load_dataset.read_csv_data('df_TV.csv') #Dataset Thibaut
  # name of the sectors
  sectors = df_OB['diagnosis'].value_counts().sort_index().index
  sectors2 = df_YB['diagnosis'].value_counts().sort_index().index  
  sectors3 = df_TV['diagnosis'].value_counts().sort_index().index 

  # % tage weightage of the sectors
  percentages = df_OB['diagnosis'].value_counts(normalize=True).round(3)*100
  percentages2 = df_YB['diagnosis'].value_counts(normalize=True).round(3)*100  
  percentages3 = df_TV['diagnosis'].value_counts(normalize=True).round(3)*100  

  # converting into radians
  radians = [math.radians((percent / 100) * 360) for percent in percentages]
  radians2 = [math.radians((percent / 100) * 360) for percent in percentages2]
  radians3 = [math.radians((percent / 100) * 360) for percent in percentages3]

  # starting angle values
  start_angle = [math.radians(0)]
  prev = start_angle[0]
  for i in radians[:-1]:
      start_angle.append(i + prev)
      prev = i + prev

  start_angle2 = [math.radians(0)]
  prev2 = start_angle2[0]
  for i in radians2[:-1]:
      start_angle2.append(i + prev2)
      prev2 = i + prev2

  start_angle3 = [math.radians(0)]
  prev3 = start_angle3[0]
  for i in radians3[:-1]:
      start_angle3.append(i + prev3)
      prev3 = i + prev3
    
  # ending angle values
  end_angle = start_angle[1:] + [math.radians(0)]
  end_angle2 = start_angle2[1:] + [math.radians(0)]
  end_angle3 = start_angle3[1:] + [math.radians(0)]
    
  # center of the pie chart
  x = 0
  y = 0
    
  # radius of the glyphs
  # radius = 1
  # sectors.sort_values()
  # sectors2.sort_values()
  # sectors3.sort_values()

  # color of the wedges
  color=Category20c[len(sectors)]
  color2=Category20c[len(sectors2)]
  color3=Category20c[len(sectors3)]

  # instantiating the figure object  
  graph = figure(title = "Répartition des labels - OB -", x_range=(-.7, .7), plot_width=500, plot_height=500)
  graph2 = figure(title = "Répartition des labels - YB -", x_range=(-.7, .7), plot_width=500, plot_height=500)  
  graph3 = figure(title = "Répartition des labels - TV -", x_range=(-.7, .7), plot_width=500, plot_height=500)  

  # plotting the graph
  for i in range(len(sectors)):
      g1=graph.annular_wedge(x, y, inner_radius=0.45, outer_radius=0.65, direction="anticlock",
                            start_angle = start_angle[i], end_angle = end_angle[i], color = color[i],
                            legend_label = sectors[i], fill_alpha=0.7, line_color='gray')
      graph.axis.visible = False
      graph.grid.grid_line_color = None
      graph.title.align = 'center'
      graph.title.text_font_size = '16pt'
      graph.legend.location = 'center'
      graph.legend.click_policy = 'hide'

  for i in range(len(sectors2)):
      g2=graph2.annular_wedge(x, y, inner_radius=0.45, outer_radius=0.65, direction="anticlock", 
                              start_angle = start_angle2[i], end_angle = end_angle2[i], color = color2[i],
                              legend_label = sectors2[i], fill_alpha=0.7, line_color='gray')
      graph2.axis.visible = False
      graph2.grid.grid_line_color = None
      graph2.title.align = 'center' 
      graph2.title.text_font_size = '16pt'
      graph2.legend.location = 'center' 
      graph2.legend.click_policy = 'hide'

  for i in range(len(sectors3)):
      g3=graph3.annular_wedge(x, y, inner_radius=0.45, outer_radius=0.65, direction="anticlock", 
                              start_angle = start_angle3[i], end_angle = end_angle3[i], color = color3[i],
                              legend_label = sectors3[i], fill_alpha=0.7, line_color='gray')
      graph3.axis.visible = False
      graph3.grid.grid_line_color = None
      graph3.title.align = 'center' 
      graph3.title.text_font_size = '16pt'
      graph3.legend.location = 'center' 
      graph3.legend.click_policy = 'hide'

  # To display graphs separately :
  _,c,_ = st.columns([1,2,1])
  with c:
    st.bokeh_chart(row(graph, graph2, graph3))  
  
  # c1 ,_, c2,_, c3,_, c4 = st.columns([2,1,2,1,2,1,2])
  # with c1:
  #   st.bokeh_chart(graph)
  # with c2:
  #   st.bokeh_chart(graph2)
  # with c3:
  #   st.bokeh_chart(graph3)
  # with c4: 
  #   OB = pd.DataFrame(df_OB['diagnosis'].value_counts(normalize=True).head(14).round(5)*100)
  #   OB = OB.rename({'diagnosis': 'OB'}, axis=1)

  #   YB = pd.DataFrame(df_YB['diagnosis'].value_counts(normalize=True).head(14).round(5)*100)
  #   YB = YB.rename({'diagnosis': 'YB'}, axis=1)

  #   TV = pd.DataFrame(df_TV['diagnosis'].value_counts(normalize=True).head(14).round(5)*100)
  #   TV = TV.rename({'diagnosis': 'TV'}, axis=1)

  #   st.dataframe(pd.concat([OB, YB, TV], axis=1))
    

  

def eye_fundus_image(diag_label, figsize=(10,10)):
 
  images = utils.list_images(diag_label)

  if len(images) >= 4:
    fig, axs = plt.subplots(2,2,figsize=figsize)
    # print(len(ax))
    fig.set_facecolor('black')
    fig.tight_layout(pad=2.0)
    for i, ax in enumerate(axs.flat):
      img = utils.load_image(diag_label, images[i])
      if img.any():
        ax.imshow(img) 
        ax.set_title(images[i][:-4]).set_color('white')
        ax.grid(False)
        ax.axis('off')
      else:
        print("impossible de charger l'image:",  images[i])
    st.pyplot(fig)

def eye_fundus_image_tab(tab, diag_label):
  with tab:
    _ , c, _ = st.columns([1,2,1])
    with c:
      eye_fundus_image(diag_label)
 


def eye_fundus():
  diagnosis_labels = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia',	'Others']
  # [normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others] = st.tabs(diagnosis_labels)
  tabs = st.tabs(diagnosis_labels)
  for i, tab in enumerate(tabs):
    eye_fundus_image_tab(tab, diagnosis_labels[i].lower())

def exploration_tab(tab, fn, in_column=True):
  with tab:
    if in_column:
      _ , c, _ = st.columns([1,2,1])
      with c:
        fn()
    else:
      fn()

def explorations():
  explorations = [f'Exporation - {e+1}' for e in range(3)]
  tabs = st.tabs(explorations)
  for i, tab in enumerate(tabs):
    exploration_tab(tab, globals()[f"exploration{i+1}"], i != len(explorations)-1)


      

MenuChoice = {
  "Diagnostic keywords" : "A",
  "Nuage de mots clés" : "B",
  "Data Viz" : "C",
  "Exemple d'images de fond d'oeil" : "D"
}



def display_choice(menu_choice, args):
  if menu_choice == 'A':
    def choice_a(): 
      col1, col2 = st.columns(2)
      color = ui.color("blue-green-60")
      with col1:
        st.markdown(f"<h4 style='text-align: center;color: {color}'>Oeil gauche<h4>", unsafe_allow_html=True)
        table = get_keywork_table(df, 0)
        st.table(table)
      with col2:
        st.markdown(f"<h4 style='text-align: center;color: {color}'>Oeil droit<h4>", unsafe_allow_html=True)
        table = get_keywork_table(df, 1)
        st.table(table)

      txt = f"{left_diag_keys.nunique()} diagnostics pour les fonds d'oeil gauche.\n{right_diag_keys.nunique()} diagnostics pour les fonds d'oeil droit."
      ui.info(txt)

    return choice_a

  if menu_choice == 'B':
    def choice_b():
      word_cloud()
      st.markdown('#')
      ui.info("Prédominance des fonds d'oeil normaux à gauche comme à droite")
    return choice_b

  if menu_choice == 'C':
    def choice_c():
      explorations()
    return choice_c

  if  menu_choice == 'D':
    def choice_d():
      eye_fundus()
    return choice_d



def main():
  sub_menus = st.sidebar.radio("Selectionnez une rubrique", options=list(MenuChoice), format_func=lambda x: x.value)
  fn = display_choice(sub_menus)
  fn()
  
 


def display():
    ### Create Title
    ui.slide_header("Exploration du dataset", gap=2)
    ui.sub_menus(MenuChoice, display_choice)

  


      



