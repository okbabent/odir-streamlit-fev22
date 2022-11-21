import streamlit as st
import base64
import os
from PIL import Image
import cv2


"""
Utility functions for:
    1. reading data
    2. setting background
    3. writing head and body
"""

@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_working_directory():
  return os.getcwd()

def get_directory(folder_name):
  # return folder_name
  return os.path.join(get_working_directory(), folder_name);

def get_data_directory():
  return get_directory('data');

def get_assets_directory():
  return get_directory('assets');

def get_resource(res_folder, res_name):
  s = os.path.join(get_directory(res_folder), res_name);
  #print(s)
  return s

def show():
  print('Hello world')

def list_images(diag_label):
  cwd = get_working_directory()
  # print(cwd)
  source_folder = os.path.join(cwd, f'data/{diag_label}')
  #source_folder = f'{cwd}/data/{diag_label}'
  # print(source_folder)
  files = [f for f in os.listdir(source_folder) if f[-3:] == 'jpg' and os.path.isfile(os.path.join(source_folder, f))]
  return files

@st.cache(suppress_st_warning=True)
def load_image(path, name):
  image_path = f'data/{path}/{name}'
  # print('load image = ', image_path)
  # image = Image.open(image_path)
  cv_image = cv2.imread(image_path)
  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
  return cv_image

