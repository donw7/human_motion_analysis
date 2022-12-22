import streamlit as st
import pandas as pd
from PIL import Image, ImageFile
import base64

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2
import imageio
import yaml
import os
import glob
import pickle as pkl
import sys
import importlib
from pathlib import Path

from src.io import iomod
from src.io.utils import Config
from src.inference import inference
from src.visualization import draw
from src.analysis import analysisphysics
from src.tests import testdraw, structshape

st.set_page_config(layout="wide")
st.title("Physiotherapy Analytics Demo")
st.sidebar.header("Context")
context_name = st.sidebar.selectbox("Choose a context", ["participant", "clinician"])
filename = st.sidebar.selectbox("Choose a file", ["dance_demo", "000077_demo"])

config = Config()

if context_name == "participant":
	with st.expander("Help"):
		st.markdown(f'''
					1. Choose a context in left sidebar - participant here would be the mockup interface for a participant having uploaded a video of themselves engaging in PT
					2. An example demo video is provided
					3. Click `Click to Run Motion Analytics` to start processing and analysis
					4. View example visualization of keypoint and anomaly detection below
					5. View in "clinician" context for more detailed analysis
					''')
	# st.video("https://www.youtube.com/watch?v=4zgjRBQEkeg&t=5s")

	module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
	input_size = 192

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
		
	"""### select demo video in sidebar"""
	file_ = open(Path("test_examples", f"{filename}.gif"), "rb")
	contents = file_.read()
	data_url = base64.b64encode(contents).decode("utf-8")
	file_.close()

	st.markdown(
			f'<img src="data:image/gif;base64,{data_url}" alt="demo gif">',
			unsafe_allow_html=True,
	)

	if st.button("Click to Run Motion Analytics"):

		path = Path("test_examples", f"{filename}.gif")
		samplevid = iomod.load_video(str(path))

		st.write("processing")

		# # Make a gif and save
		fgifpath = str(Path("test_examples", f"{filename}_inference_lightning.gif"))
		iomod.imageio.mimsave(fgifpath, samplevid)

		# Read & decode from saved file
		imagesgif = tf.io.read_file(fgifpath)
		images = tf.image.decode_gif(imagesgif)

		# Run inference
		out_keypoints, out_edges = \
				inference.inference_video(movenet, images, input_size, \
																	config.kpts, config.edges, config.params["KEYPOINT_THRESH_SCORE_CROP"], config.params)

		# Get mask for labeling edges based on velocity
		mask_edge = analysisphysics.compute_edge_velocities(out_edges, config.params["EDGE_VEL_THRESH"])


		out_images_draw, out_images_drawsubplots = \
				draw.wrap_draw_subplots(images, out_keypoints, out_edges, config.edges, \
																mask_edge=mask_edge, figsize=(10,10))

		# Prepare gif visualization.
		output2 = np.stack(out_images_drawsubplots, axis=0)
		iomod.convert_to_gif(output2, fgifpath, fps=10)

		"""### processed - body keypoints detected; anomalous velocities of motion highlighted in red"""
		file_processed = open(fgifpath, "rb")
		contents = file_processed.read()
		data_url_processed = base64.b64encode(contents).decode("utf-8")
		file_processed.close()

		with open(Path("test_examples", f"{filename}_out_keypoints.pkl"), 'wb') as file:
			pkl.dump(out_keypoints, file)
		with open(Path("test_examples", f"{filename}_out_edges.pkl"), 'wb') as file:
			pkl.dump(out_edges, file)
		with open(Path("test_examples", f"{filename}_out_images_draw.pkl"), 'wb') as file:
			pkl.dump(out_images_draw, file)
		with open(Path("test_examples", f"{filename}_out_images_drawsubplots.pkl"), 'wb') as file:
			pkl.dump(out_images_drawsubplots, file)

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
					4. Click `Run` to start processing and analysis
					5. View analysis
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
def load_edges(filename):
	with open(Path("test_examples", f"{filename}_out_keypoints.pkl"), 'rb') as file:
		out_keypoints = pkl.load(file)
	with open(Path("test_examples", f"{filename}_out_edges.pkl"), 'rb') as file:
		out_edges = pkl.load(file)
	with open(Path("test_examples", f"{filename}_out_images_draw.pkl"), 'rb') as file:
		out_images_draw = pkl.load(file)
	with open(Path("test_examples", f"{filename}_out_images_drawsubplots.pkl"), 'rb') as file:
		out_images_drawsubplots = pkl.load(file)
	return out_keypoints, out_edges, out_images_draw, out_images_drawsubplots

if context_name == "clinician":
	with st.expander("Help"):
		st.markdown(f'''
					1. Choose a context in left sidebar
					2. Select frame
					3. Select keypoints to plot (by default, notable upper and lower extremity keypoints are selected)
					4. The x-y positions of the selected keypoints will be plotted, and the inferenced image will display in sidebar accordingly
					5. Move the slider to select a different frame (black line indicates the selected frame on plots). Resize the sidebar as needed.
					6. Pink highlights are areas of interest where anomalous velocities of motion have been detected by the model. This type of visualization can show clearly that there is more motion activity and potentially more anomalous motion in the upper extremities.
					7. Clinician can then use this objective data to quickly determine if the participant is performing the motion correctly - data which is typically not accessible.
					''')

	out_keypoints, out_edges, out_images_draw, out_images_drawsubplots = load_edges(filename)
	mask_edge = analysisphysics.compute_edge_velocities(out_edges, config.params["EDGE_VEL_THRESH"])
	mask_edge = mask_edge.reshape(-1, 36).astype('float32') # numframes, 18 joints x 2 points
	anom_idx = np.max(mask_edge, axis=1).astype("int")

	out_edges = out_edges.reshape(-1, 72).astype('float32')
	name_combinations = config.get_name_combinations()
	df_edgenames = pd.DataFrame(name_combinations, columns=["name"])

	frame_idx = st.slider(
		"Select frame (use left and right arrows to scroll through)",
		0, 40, 0
	)
	st.header("Upper extremities position")
	feature_query_upper = st.multiselect(
		"Select keypoints to plot",
		name_combinations,
		default=[
			"left_elbow-left_wrist-start_y",
			"left_elbow-left_wrist-end_y",
			"right_elbow-right_wrist-start_y",
			"right_elbow-right_wrist-end_y",
			"left_knee-left_ankle-start_y",
		],
		key="upper"
	)
	idx = df_edgenames.index[df_edgenames.name.isin(feature_query_upper)].values

	fig_upper, ax_upper = plt.subplots(figsize=(5, 2))
	ax_upper.plot(out_edges[:,idx])
	ax_upper.set_xlabel('Frame')
	ax_upper.set_ylabel('xy position')
	ax_upper.legend(feature_query_upper, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')

	segments = analysisphysics.get_segments(anom_idx)
	for segi in segments:
		analysisphysics.plot_patch(ax_upper, segi)

	analysisphysics.plot_patchline(ax_upper, frame_idx)

	# todo: compute anomalies on the fly for only displayed plots
	st.pyplot(fig_upper)


	st.header("Lower extremities position")
	feature_query_lower = st.multiselect(
		"Select keypoints to plot",
		name_combinations,
		default=[
			"right_knee-right_ankle-start_y",
			"left_knee-left_ankle-start_y",
		],
		key="lower"
	)

	idx = df_edgenames.index[df_edgenames.name.isin(feature_query_lower)].values
	fig_lower, ax_lower = plt.subplots(figsize=(5, 2))
	ax_lower.plot(out_edges[:,idx])
	ax_lower.set_xlabel('Frame')
	ax_lower.set_ylabel('xy position')
	ax_lower.legend(feature_query_lower, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')

	# segments = analysisphysics.get_segments(anom_idx)
	# for segi in segments:
	# 	analysisphysics.plot_patch(ax_lower, segi)

	analysisphysics.plot_patchline(ax_lower, frame_idx)
	st.pyplot(fig_lower)

	image = out_images_drawsubplots[frame_idx]
	with st.sidebar:
		sidebar_image = st.image(image)