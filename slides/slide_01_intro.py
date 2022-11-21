import streamlit as st
from app import utils
from app import ui

def header():
  return {'id': "Introduction", 'icon': 'easel', 'callback': display}

def display():
  ui.slide_header('Objectif: Elaboration d\'un modèle de classification des maladies ophtalmiques sur la base des fonds d\'oeil', gap=(15,5), description='Sujet soumis en 2019 par la Peking University - Compétition international -')

    