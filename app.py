import nltk
import re
import streamlit as st
import joblib
from train.deploy_pipeline import Dep_pipeline

pipeline = Dep_pipeline()
#Streamlit app

def main():
    st.title('Resume Screening App')
    uploaded_file = st.file_uploader('Upload Resume', type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_txt = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt = resume_bytes.decode('latin-1')
        
        prediction_name= pipeline.run(resume_txt)
        st.write('Predicting Category is : ',prediction_name)
if __name__ == '__main__':
    main()