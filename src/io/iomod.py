"""Convenience functions for modifying, encoding and decoding videos and related files"""

from tensorflow_docs.vis import embed
import numpy as np
import cv2 as cv
import imageio


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
	cap = cv.VideoCapture(path)
	frames = []
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame = crop_center_square(frame)
			frame = cv.resize(frame, image_size)
			frame = frame[:, :, [2, 1, 0]]
			frames.append(frame)

			if len(frames) == max_frames:
				break
	finally:
		cap.release()
	return np.array(frames)

def encode_video(images, out_path, fps=25, image_size=(224, 224)):
    """Encodes a video from a sequence of frames."""
    images = images.astype(np.uint8)
    writer = imageio.get_writer(out_path, fps=fps)
    for fri in range(images.shape[0]):
        frame = images[fri]
        writer.append_data(frame)
    writer.close()

