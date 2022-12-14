import streamlit as st
import pandas as pd
from PIL import Image, ImageFile
import base64

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import yaml
import os
import glob
import pickle as pkl
import sys
import importlib

from src.io import iomod
from src.inference import inference
from src.visualization import draw
from src.analysis import analysisphysics
from src.tests import testdraw, structshape

def load_config(path):
		with open(path) as file:
				config = yaml.full_load(file)
		return config

# set up paths
CONFIG_KEYPOINTS = load_config(r"configs/keypoints.yaml")
CONFIG_EDGE_COLORS = load_config(r"configs/edge_colors.yaml")
PARAMS = load_config(r"configs/analysis_parameters.yaml")

# confidence threshold to determine whether a keypoint prediction is reliable, e.g. for cropping algorithm during inference
KEYPOINT_THRESH_SCORE_CROP = load_config("configs/analysis_parameters.yaml")["KEYPOINT_THRESH_SCORE_CROP"]


st.set_page_config(layout="wide")

st.title("Physiotherapy Analytics Demo")

st.sidebar.header("Context")
context_name = st.sidebar.selectbox("Choose a context", ["participant", "clinician"])


if context_name == "participant":
	with st.expander("Help"):
		st.markdown(f'''
					1. Choose a context in left sidebar
					2. An example demo video is provided
					3. Select analysis
					4. Click `Run` to start processing and analysis
					5. View analysis
					''')
	# st.video("https://www.youtube.com/watch?v=4zgjRBQEkeg&t=5s")

	'''load model'''
	module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
	input_size = 192

	st.experimental_memo()
	def movenet(input_image):
		"""Runs detection on an input image.

		Args:
			input_image: A [1, height, width, 3] tensor represents the input image
				pixels. Note that the height/width should already be resized and match the
				expected input resolution of the model before passing into this function.

		Returns:
			A [1, 1, 17, 3] float numpy array representing the predicted keypoint
			coordinates and scores.
		"""
		model = module.signatures['serving_default']

		# SavedModel format expects tensor type of int32.
		input_image = tf.cast(input_image, dtype=tf.int32)
		# Run model inference.
		outputs = model(input_image)
		# Output is a [1, 1, 17, 3] tensor.
		keypoints_with_scores = outputs['output_0'].numpy()
		return keypoints_with_scores
		
	"""### demo dance video"""
	file_ = open("dance_input.gif", "rb")
	contents = file_.read()
	data_url = base64.b64encode(contents).decode("utf-8")
	file_.close()

	st.markdown(
			f'<img src="data:image/gif;base64,{data_url}" alt="demo gif">',
			unsafe_allow_html=True,
	)

	if st.button("Click to Run Motion Analytics"):

		test_path = 'dance_input.gif'
		samplevid = iomod.load_video(test_path)

		st.write("processing")

		# # Make a gif and save
		fgifpath = f"{test_path}_test_thunder6.gif"
		iomod.imageio.mimsave(fgifpath, samplevid)

		# Read & decode from saved file
		imagesgif = tf.io.read_file(fgifpath)
		images = tf.image.decode_gif(imagesgif)

		# Run inference
		out_keypoints, out_edges = \
				inference.inference_video(movenet, images, input_size, \
																	CONFIG_KEYPOINTS, CONFIG_EDGE_COLORS, KEYPOINT_THRESH_SCORE_CROP, PARAMS)

		# Get mask for labeling edges based on velocity
		mask_edge = analysisphysics.compute_edge_velocities(out_edges, PARAMS["EDGE_VEL_THRESH"])


		output_images, output_images2 = \
				draw.wrap_draw_subplots(images, out_keypoints, out_edges, CONFIG_EDGE_COLORS, \
																mask_edge=mask_edge, figsize=(10,10))

		# Prepare gif visualization.
		output2 = np.stack(output_images2, axis=0)
		iomod.convert_to_gif(output2, fgifpath, fps=10)

		"""### processed - body keypoints detected; anomalous velocities of motion highlighted in red"""
		file_processed = open(fgifpath, "rb")
		contents = file_processed.read()
		data_url_processed = base64.b64encode(contents).decode("utf-8")
		file_processed.close()

		with open('out_keypoints.pkl', 'wb') as file:
			pkl.dump(out_keypoints, file)

		with open('out_edges.pkl', 'wb') as file:
			pkl.dump(out_edges, file)

		st.markdown(
				f'<img src="data:image/gif;base64,{data_url_processed}" alt="demo gif processed">',
				unsafe_allow_html=True,
		)
	else:
		pass


if context_name == "todo: upload":
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

@st.experimental_memo
def load_edges():
		with open('out_keypoints.pkl', 'rb') as file:
			out_keypoints = pkl.load(file)
		
		with open('out_edges.pkl', 'rb') as file:
			out_edges = pkl.load(file)
		return out_keypoints, out_edges

if context_name == "clinician":
	with st.expander("Help"):
		st.markdown(f'''
					1. Choose a context in left sidebar
					2. Select participant
					3. Select analysis
					4. Click `Run` to start processing and analysis
					5. View analysis¡™¡
					''')

	out_keypoints, out_edges = load_edges()

	out_edges = out_edges.reshape(-1, 72).astype('float32')

	feature_idx = st.multiselect(
		"Select features to plot",
		[0, 1, 2, 3])

	fig, ax = plt.subplots()
	ax.plot(out_edges[:,feature_idx])
	st.pyplot(fig)