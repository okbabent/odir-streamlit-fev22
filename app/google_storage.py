import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)





# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    global content
    content = bucket.blob(file_path).download_to_filename('./tmp')
    return content

def connect_google_strorage():
    # Create API client.
   

    bucket_name = "odir-datascientest"
    #file_path = "odir_model_weights_Xception_2022_10_21_multiclass_fine_tuning.h5"
    file_path = 'requirements.txt'

    content = read_file(client, bucket_name, file_path)

    print(content)