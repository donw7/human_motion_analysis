import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import cv2


st.set_page_config(layout="wide")

st.title("Physiotherapy Analytics Dashboard")

st.sidebar.header("Context")
context_name = st.sidebar.selectbox("Choose a context", ["demo", "participant", "clinician"])


if context_name == "demo":
  with st.expander("Help"):
    st.markdown(f'''
          1. Choose a context in left sidebar
          2. An example demo video from youtube is provided
          3. Select analysis
          4. Click `Run` to start processing and analysis
          5. View analysis
          ''')
  st.video("https://www.youtube.com/watch?v=4zgjRBQEkeg&t=5s")
  
  # Load the input image.
  # gif = Image.open('dance_input.gif')
  st.write(gif)

if context_name == "participant":
  with st.expander("Help"):
    st.markdown(f'''
          1. Choose a context in left sidebar
          2. Input data
          3. Upload a video (h.264)
          4. Select analysis
          5. Click `Run` to start processing and analysis
          6. View analysis
          ''')
  
  st.title("Upload a video file")
  video_file = st.file_uploader("h.264")
  if video_file:
    st.video("video_file")

  st.title("Enter data")

  with st.form('my_form'):
      participant_name = st.text_input("Name", "Jane Doe")
      exercise_type = st.selectbox("Exercise Type", ["Arm raise", "Arm curl"])

      # Every form must have a submit button
      submitted = st.form_submit_button('Submit')

  if submitted:
      st.markdown(f'''
          You have entered:
          - Participant Name: {participant_name}
          - Exercise Type: {exercise_type}
          ''')
  else:
      st.write("Enter data above")

if context_name == "clinician":
  st.write("todo")
  with st.expander("Help"):
    st.markdown(f'''
          1. Choose a context in left sidebar
          2. Select participant
          3. Select analysis
          4. Click `Run` to start processing and analysis
          5. View analysis
          ''')





