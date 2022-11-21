import streamlit as st
import pandas as pd 
from app import load_dataset
from app import ui
from streamlit_option_menu import option_menu

def get_dataframe_info(df):
    """
    input
       df -> DataFrame
    output
       df_null_counts -> DataFrame Info (sorted)
    """

    df_info = pd.DataFrame({
      "Column": df.columns, 
      "Non-Null count": len(df)-df.isnull().sum().values, 
      "Null count": df.isnull().sum().values, 
      "Dtype": df.dtypes.astype(str).values})

   
    

    return df_info



MenuChoice = {
  "Aperçu des données": 'A',
  "Information sur les données": 'B',
  "Description des données" : 'C'
}



def display_choice(menu_choice, args):
  df = args['dataset']
  if menu_choice == 'A':
    def choice_a():
      _,c,_ = st.columns([1,5,1])
      with c:
        line_to_plot = st.slider("Selectionner le nombre de lignes à visualiser", min_value=5, max_value = 100)
        st.dataframe(df.head(line_to_plot))
    return choice_a

  if menu_choice == 'B':
    def choice_b():
      _,c,_ = st.columns([1,5,1])
      with c:
        df_info = get_dataframe_info(df)
        st.dataframe(df_info.T)
        dtypes = df_info['Dtype'].value_counts()
        st.write(dtypes)
        st.code('Nombre de duplication est égal à [df.duplicated().sum()]  ' + str(df.duplicated().sum()) + '\nLa base de données semble ne présente aucunes données absentes ou manquantes')
    return choice_b

  if menu_choice == 'C':
    def choice_c():
      _,c,_ = st.columns([1,2,1])
      with c:
        df_desc = df.describe().T
        df_desc = df_desc.astype({'count':'int'})
        st.dataframe(df_desc)
    return choice_c

 

 
def header():
  return {'id': "Présentation des données", 'icon': 'book', 'callback': display}

def choice_value(x):
  print('X =',x)
  print('value of x =',x.value)
  return x.value



def display():

    ### Create Title
    ui.slide_header('Présentation des données', gap=(None,None,10), description='Description & information')
  
    df = load_dataset.read_odir_data()
    ui.sub_menus(MenuChoice, display_choice, dataset=df)

   

      
      



