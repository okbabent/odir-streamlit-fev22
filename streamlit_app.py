# Core Pkg
import streamlit as st
import importlib
import glob
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np



#from streamit_experiments import streamlit_experiments
# from demo_stream_titanic import demo_streamlit # Basic ML web app with stremlit






# Print results.
# for line in content.strip().split("\n"):
#     name, pet = line.split(",")
#     st.write(f"{name} has a :{pet}:")

def main():
    # Parameters
    STARTING_SLIDE = 0

    st.set_page_config(
    page_title="ODIR 2019",
    page_icon="https://streamlit.io/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",)

    # connect_google_strorage()
  

  
   

    slide_files = sorted(glob.glob("slides/slide_*.py"))
    #print(slide_files)
    number_of_slides = len(slide_files)

    if 'number_of_slides' not in st.session_state:
        st.session_state.number_of_slides = number_of_slides

    # import all slides
    module_str_list = [importlib.import_module(slide.replace("/", ".").replace(".py", "")) for slide in slide_files]
    menu_list = [module.header() for module in module_str_list]
    menus =  list(map(lambda x: x['id'], menu_list))
    icons = list(map(lambda x: x['icon'], menu_list))
  
    with st.sidebar:
        choose = option_menu("Présentation", menus,
                            icons=icons,
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#FFFFFF"},
            "icon": {"color": "#27DCE0", "font-size": "25px"}, 
            "menu-icon": {"color":"#27DCE0"},
            "nav": {"color": "#00FF"},
            "menu-title": {"font-size": "20px", "font-weight": "bold", "color": "black", "text-align": "left", "margin":"0px"},
            "nav-link": {"font-size": "16px", "color": "black", "text-align": "left", "margin":"0px", "--hover-color": "rgb(70, 40, 221)", "--hover-text-color":"#FFFFFF"},
            "nav-item": {"color": "#00FF"},
            "nav-link-selected": {"background-color": "#FFF1D3"},
        })
    selected_rubric = next(item for item in menu_list if item["id"] == choose)
    selected_rubric['callback']()




    # List of pages
    # option_menu = [
    #     {'id': "Titre", 'callback': streamlit_odir_title},
    #     {'id': "Introduction", 'callback': streamlit_odir_intro},
    #     {'id': "Présentation des données", 'callback': streamlit_data_presentation},
    #     {'id': "Exploration des données", 'callback': streamlit_data_exploration},
    #     {'id': "Traitement des données", 'callback': streamlit_data_treatment},
    #     {'id': "Modélisations", 'callback': streamlit_data_modelization},
    #     {'id': "Analyse et Performance des modèles", 'callback': streamlit_model_analysis},
    #     {'id': "Perspectives", 'callback': streamlit_odir_perspectives},
    #     {'id': "Conclusion", 'callback': streamlit_odir_conclusion}
    #     ]
  
    # # Page navigation
    # menu_ids =  list(map(lambda x: x['id'], option_menu))
   
    # # Sidebar
    # selected_menu_id = st.sidebar.selectbox("Présentation", menu_ids)
    # selected_rubric = next(item for item in option_menu if item["id"] == selected_menu_id)
    
    # # call function
    # selected_rubric['callback']()

    
    # List of rubric
    # menu_list = [
    #     {'id': "Titre", 'icon': 'house', 'callback': null},
    #     {'id': "Introduction", 'icon': 'easel', 'callback': streamlit_odir_intro},
    #     {'id': "Présentation des données", 'icon': 'book', 'callback': streamlit_data_presentation},
    #     {'id': "Exploration des données", 'icon': 'binoculars', 'callback': streamlit_data_exploration},
    #     {'id': "Traitement des données", 'icon': 'bar-chart', 'callback': streamlit_data_treatment},
    #     {'id': "Modélisations", 'icon': 'boxes', 'callback': streamlit_data_modelization},
    #     {'id': "Analyse et Performance des modèles", 'icon': 'graph-up', 'callback': streamlit_model_analysis},
    #     {'id': "Perspectives", 'icon': 'eyeglasses', 'callback': streamlit_odir_perspectives},
    #     {'id': "Conclusion", 'icon': 'list-task', 'callback': streamlit_odir_conclusion},
    #     {'id': "Debug", 'icon':'', 'callback': streamlit_odir_debug}
    #     ]


    # menus =  list(map(lambda x: x['id'], menu_list))
    # icons = list(map(lambda x: x['icon'], menu_list))

    # with st.sidebar:
    #     choose = option_menu("Présentation", menus,
    #                         icons=icons,
    #                         menu_icon="app-indicator", default_index=0,
    #                         styles={
    #         "container": {"padding": "5!important", "background-color": "#FFFFFF"},
    #         "icon": {"color": "#27DCE0", "font-size": "25px"}, 
    #         "menu-icon": {"color":"#27DCE0"},
    #         "nav": {"color": "#00FF"},
    #         "menu-title": {"font-size": "20px", "font-weight": "bold", "color": "black", "text-align": "left", "margin":"0px"},
    #         "nav-link": {"font-size": "16px", "color": "black", "text-align": "left", "margin":"0px", "--hover-color": "rgb(70, 40, 221)", "--hover-text-color":"#FFFFFF"},
    #         "nav-item": {"color": "#00FF"},
    #         "nav-link-selected": {"background-color": "#FFF1D3"},
    #     })
    # selected_rubric = next(item for item in menu_list if item["id"] == choose)
    # selected_rubric['callback']()

  


if __name__ == '__main__':
    main()