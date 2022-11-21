import streamlit as st
import pandas as pd
import requests
import io



def url_to_id(url):
    x = url.split("/")
    return x[5]


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def read_odir_data():
  #xls_file = utils.get_ressource('data', 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
  xls_file = 'data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
  df = pd.read_excel(xls_file)
  df['Patient Sex'] = df['Patient Sex'].replace(['Female','Male'],[0,1])
  return df
  

@st.cache(suppress_st_warning=True)
def read_csv_data(csv_file_name, label_name=None, dict=None):
  csv_file = 'data/' + csv_file_name
  df = pd.read_csv(csv_file)
  if label_name and dict:
    df['diagnosis'] = df[label_name].replace(dict)
  return df
 


  

# @st.cache(suppress_st_warning=True)
# def read_data_csv(file_csv_name):
#   # url = 'https://drive.google.com/drive/u/0/folders/1mGluoMZ12iJ1kyDtDNT8nWf441baBitZ/'+file_csv_name
#   #url = 'https://drive.google.com/file/d/1rvfpO2KXS-kd-Nt-78DeO1nYaAs0aPSb/edit'
#   url = 'https://drive.google.com/file/d/1rvfpO2KXS-kd-Nt-78DeO1nYaAs0aPSb/view?usp=sharing'

#   file_id = '1rvfpO2KXS-kd-Nt-78DeO1nYaAs0aPSb'
#   s=requests. get(url).content
#   print(s)
#   csv_content = io.StringIO(s.decode('utf-8'))

#   df = pd.read_csv(csv_content)
#   return df
