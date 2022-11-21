import streamlit as st
#from app.utils import utils

def header():
   return {'id': "Titre", 'icon': 'house', 'callback': display}

def display():
    c0, c1, c2, _ = st.columns([1,8,3,2])
    with c1:
        
        st.markdown('<br><br><br><br><br><br>', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: white;'>Ocular Disease Intelligent Recognition</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: grey;'>Compétition Internationale de la \"Peking University\" sur la reconnaissance intelligente de pathologie oculaires (ODIR-2019)</h3>", unsafe_allow_html=True) 

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write('')
        st.markdown('<br><br><br><br><br><br>', unsafe_allow_html=True)
        li_str = ['Equipe : Okba BENTOUMI, Yannick BODIN, Thibaut VAUSSELIN',
        'Mentor de projet : Anthony JAILLET',
        'Chef de Cohorte : Gaspard GRIMM',
        'Promo : Fev22 Continu DS']
        style = "style='text-align: left; color: grey;'"
        for li in li_str:
          st.markdown(f"<h4>{li}</h4>", unsafe_allow_html=True)
      
        
        #st.write("Repositorio de la presentación: https://github.com/sebastiandres/talk_2021_11_pyconcl")

    with c0:
        #st.image("https://i0.wp.com/datascientest.com/wp-content/uploads/2022/03/logo-2021-1.png")
        st.image("assets/datascientest-logo.png")
    with c2:
      st.image("assets/eye_fundus.png")