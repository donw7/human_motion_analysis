from tensorflow_docs.vis import embed
import yaml
import numpy as np
import cv2
import imageio
import pickle as pkl


# write yaml file from dict
def write_yaml(data, file_name):
		with open(file_name, 'w') as f:
				yaml.dump(data, f, default_flow_style=False, sort_keys=False)

'''example usage:
iomod.write_yaml(KEYPOINT_DICT, 'keypoints.yaml')
		'''


def to_gif(images, fps):
	"""Converts image sequence (4D numpy array) to gif."""
	imageio.mimsave('./animation.gif', images, fps=fps)
	return embed.embed_file('./animation.gif')

def convert_to_gif(images, fpath, fps):
	"""Converts image sequence (4D numpy array) to gif."""
	imageio.mimsave(fpath, images, fps=fps)
	pass

def crop_center_square(frame):
	y, x = frame.shape[0:2]
	min_dim = min(y, x)
	start_x = (x // 2) - (min_dim // 2)
	start_y = (y // 2) - (min_dim // 2)
	return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, image_size=(224, 224)):
	cap = cv2.VideoCapture(path)
	frames = []
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame = crop_center_square(frame)
			frame = cv2.resize(frame, image_size)
			frame = frame[:, :, [2, 1, 0]]
			frames.append(frame)

			if len(frames) == max_frames:
				break
	finally:
		cap.release()
	return np.array(frames)# / 255.0

def encode_video(images, out_path, fps=25, image_size=(224, 224)):
	"""Encodes a video from a sequence of frames."""
	images = images.astype(np.uint8)
	fourcc = cv2.VideoWriter_fourcc(*'H264')
	writer = cv2.VideoWriter(out_path, fourcc, fps, image_size)
	for fri in range(images.shape[0]):
		frame = cv2.cvtColor(images[fri], cv2.COLOR_BGR2RGB)
		writer.write(frame)
	writer.release()

'''another example (used in 3D action net?)
def to_gif(images):
	converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
	imageio.mimsave('./animation.gif', converted_images, fps=25)
	return embed.embed_file('./animation.gif')
'''

