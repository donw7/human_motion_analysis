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
import cv2 as cv
import os
import glob
import pickle as pkl
import sys
import importlib
import tempfile
from pathlib import Path

from src.io import iomod
from src.io.utils import Config
from src.inference import inference
from src.visualization import draw
importlib.reload(draw)
from src.analysis import analysisphysics
from src.tests import testdraw, structshape


# set up layout
st.set_page_config(layout="wide")
st.title("Physiotherapy Analytics Demo")
st.sidebar.header("Context")
context_name = st.sidebar.selectbox("Choose a context", ["participant", "clinician"])
filename = st.sidebar.selectbox("Choose a file", ["upload", "dance_demo", "armraise_000077_demo", "armraise_000010_demo", "armraise_000045_demo", "curl_000011_demo"])

# set configs
config = Config()
input_size = config.params["INPUT_SIZE"]
model_name = config.params["MODEL_NAME"]

# util functions (cached)
@st.cache
def load_model(model_name):
	model = tf.saved_model.load(Path("models", "movenet_lightning"))
	return model

@st.experimental_memo
def load_edges(filename):
	with open(str(Path("test_examples", f"{filename}_out_keypoints.pkl")), 'rb') as file:
		out_keypoints = pkl.load(file)
	with open(str(Path("test_examples", f"{filename}_out_edges.pkl")), 'rb') as file:
		out_edges = pkl.load(file)
	with open(str(Path("test_examples", f"{filename}_out_images_draw.pkl")), 'rb') as file:
		out_images_draw = pkl.load(file)
	with open(str(Path("test_examples", f"{filename}_out_images_drawsubplots.pkl")), 'rb') as file:
		out_images_drawsubplots = pkl.load(file)
	return out_keypoints, out_edges, out_images_draw, out_images_drawsubplots

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
	# model = module.signatures['serving_default']
	input_image = tf.cast(input_image, dtype=tf.int32) # SavedModel format expects tensor type of int32
	outputs = model.signatures['serving_default'](input_image) # run inference
	keypoints_with_scores = outputs['output_0'].numpy() # [1, 1, 17, 3] tensor --> np
	return keypoints_with_scores

def inference_analyze_pipe(images: np.array, filename: str, config=config) -> None:
	st.write(f"decoded video, inference starting")
	out_keypoints, out_edges = inference.inference_video(
		movenet, images, config.params["INPUT_SIZE"],
		config.kpts, config.edges, config.params["KEYPOINT_THRESH_SCORE_CROP"], config.params
	)

	st.write("inference done, compute_edge_velocities starting") # --> mask for labeling anomalous edges
	_, mask_edge = analysisphysics.compute_edge_velocities(out_edges, config.params["EDGE_VEL_THRESH"])

	st.write("compute_edge_velocities done, draw subplots starting")
	out_images_draw, out_images_drawsubplots = draw.wrap_draw_subplots(
		images, out_keypoints, out_edges, config.edges,
		mask_edge=mask_edge, figsize=(5,5)
	)

	st.write("draw done, prepare gif visualization starting")
	output2 = np.stack(out_images_drawsubplots, axis=0)
	fgifpath = str(Path("test_examples", f"{filename}_inference_lightning.gif"))
	iomod.convert_to_gif(output2, fgifpath, fps=10)

	"""### processed - body keypoints detected; anomalous velocities of motion highlighted in red (if any)"""
	file_processed = open(fgifpath, "rb")
	contents = file_processed.read()
	data_url_processed = base64.b64encode(contents).decode("utf-8")
	file_processed.close()
	st.markdown(
			f'<img src="data:image/gif;base64,{data_url_processed}" alt="demo gif processed">',
			unsafe_allow_html=True,
	)

	# encode video to mp4 and save all outputs
	video_draw_path = str(Path("test_examples", f"{filename}_out_images_draw.mp4"))
	iomod.encode_video(out_images_draw, video_draw_path)
	with open(Path("test_examples", f"{filename}_out_keypoints.pkl"), 'wb') as file:
		pkl.dump(out_keypoints, file)
	with open(Path("test_examples", f"{filename}_out_edges.pkl"), 'wb') as file:
		pkl.dump(out_edges, file)
	with open(Path("test_examples", f"{filename}_out_images_draw.pkl"), 'wb') as file:
		pkl.dump(out_images_draw, file)
	with open(Path("test_examples", f"{filename}_out_images_drawsubplots.pkl"), 'wb') as file:
		pkl.dump(out_images_drawsubplots, file)


# page content start
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

	model = load_model(model_name)

	if filename == "upload":
		st.write("upload video")
		file = st.file_uploader("Choose a file")
		if file is not None:
			contents = file.getvalue()
			with tempfile.NamedTemporaryFile(delete=False) as f:
				f.write(contents)
				temp_file = f.name
			cap = cv.VideoCapture(temp_file)
			fps = cap.get(cv.CAP_PROP_FPS)
			frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
			width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

			images = np.empty((frames, height, width, 3), dtype=np.uint8)
			for fri in range(frames):
				ret, frame = cap.read()
				if ret:
					image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
					images[fri] = image
				else:
					break

			cap.release()
			os.remove(temp_file)
			images = tf.convert_to_tensor(images)

			if st.button("Run Motion Analytics"):
				st.write("processing")
				inference_analyze_pipe(images, filename)

	else: # e.g. demo videos 
		"""### select or upload video in sidebar"""
		file_ = open(Path("test_examples", f"{filename}.gif"), "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()
		st.markdown(
				f'<img src="data:image/gif;base64,{data_url}" alt="demo gif">',
				unsafe_allow_html=True,
		)

		if st.button("Run Motion Analytics"):
			path = Path("test_examples", f"{filename}.gif")
			samplevid = iomod.load_video(str(path)) # --> np.array
			images = tf.convert_to_tensor(samplevid)
			inference_analyze_pipe(images, filename)


if context_name == "clinician":
	with st.expander("Help"):
		st.markdown(f'''
					1. Choose a context in left sidebar
					2. Select frame
					3. Select keypoints to plot (by default, notable upper and lower extremity keypoints are selected)
					4. The x-y positions of the selected keypoints will be plotted, and the inferenced image will display in sidebar accordingly
					5. Move the slider to select a different frame (black line indicates the selected frame on plots). Resize the sidebar as needed.
					6. Can then view video from selected areas of interest, such as pink highlights where anomalous velocities of motion have been detected by the model. This type of visualization can show clearly that there is more motion activity and potentially more anomalous motion in the upper extremities.
					7. Clinician can then use this objective data to quickly determine if the participant is performing the motion correctly - data which is typically not accessible.
					''')

	out_keypoints, out_edges, out_images_draw, out_images_drawsubplots = load_edges(filename)
	edge_vel, mask_edge = analysisphysics.compute_edge_velocities(out_edges, config.params["EDGE_VEL_THRESH"])
	mask_edge = mask_edge.reshape(-1, 36).astype('float32') # numframes, 18 joints x 2 points
	anom_idx = np.max(mask_edge, axis=1).astype("int")

	out_edges = out_edges.reshape(-1, 72).astype('float32')
	name_combinations = config.get_name_combinations()
	df_edgenames = pd.DataFrame(name_combinations, columns=["name"])

	frame_idx = st.sidebar.slider(
		"Select frame (black bar in plots indicates selected frame - scroll to area of interest for start of video)",
		0, len(out_images_drawsubplots), 0
	)

	st.subheader("Upper extremities position")
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
	fig_upper, ax_upper = plt.subplots(figsize=(5, 1))
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
	st.subheader("Lower extremities position")
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
	fig_lower, ax_lower = plt.subplots(figsize=(5, 1))
	ax_lower.plot(out_edges[:,idx])
	ax_lower.set_xlabel('Frame')
	ax_lower.set_ylabel('xy position')
	ax_lower.legend(feature_query_lower, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')

	# todo: divide into axis-specific anomaly highlights rather than overall
	# segments = analysisphysics.get_segments(anom_idx)
	# for segi in segments:
	# 	analysisphysics.plot_patch(ax_lower, segi)

	analysisphysics.plot_patchline(ax_lower, frame_idx)
	st.pyplot(fig_lower)

	# plot Y velocity of edges
	st.subheader("Y velocity of forearm")
	feature_query_vel = st.multiselect(
		"Select keypoints to plot",
		config.edge_names,
		default=[
			"left_elbow-left_wrist",
 			"right_elbow-right_wrist",
		],
		key="vel_y"
	)
	idx = [config.edge_names.index(x) for x in feature_query_vel]
	fig_vel_y, ax_vel_y = plt.subplots(figsize=(5, 1))
	ax_vel_y.plot(edge_vel[:,idx,1]) # Y only
	ax_vel_y.set_xlabel('Frame')
	ax_vel_y.set_ylabel('Y velocity')
	ax_vel_y.legend(feature_query_vel, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
	ax_vel_y.set_ylim([0, 20])
	for segi in segments:
		analysisphysics.plot_patch(ax_vel_y, segi)
	analysisphysics.plot_patchline(ax_vel_y, frame_idx)
	st.pyplot(fig_vel_y)

	# plot X velocity of edges
	st.subheader("X velocity of forearm")
	feature_query_vel = st.multiselect(
		"Select keypoints to plot",
		config.edge_names,
		default=[
			"left_elbow-left_wrist",
 			"right_elbow-right_wrist",
		],
		key="vel_x"
	)
	idx = [config.edge_names.index(x) for x in feature_query_vel]
	fig_vel_x, ax_vel_x = plt.subplots(figsize=(5, 1))
	ax_vel_x.plot(edge_vel[:,idx,0]) # X only
	ax_vel_x.set_xlabel('Frame')
	ax_vel_x.set_ylabel('X velocity')
	ax_vel_x.legend(feature_query_vel, loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
	ax_vel_x.set_ylim([0, 20])
	for segi in segments:
		analysisphysics.plot_patch(ax_vel_x, segi)
	analysisphysics.plot_patchline(ax_vel_x, frame_idx)
	st.pyplot(fig_vel_x)

	with st.sidebar:
		# todo: handle if file does not exist
		placeholder = st.empty()

		if st.button("Show Video from Selected Frame"):
			placeholder.video(str(Path("test_examples", f"{filename}_out_images_draw.mp4")), start_time=int(frame_idx/25)) # hardcoded fps conversion for now - todo


# todo: upload feature characteristics
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