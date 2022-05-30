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
  
def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)

      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

'''another example (used in 3D action net?)
def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
  return embed.embed_file('./animation.gif')
'''

def compile_data_wrapper(paths):
  # try and catch filenotfounderror
  skippedct = 0
  skippedpaths = []
  data_all = {}
  data_all["out_keypoints"] = {}
  data_all["out_edges"] = {}

  for path in paths:
    try:
      with open(f"{path}_out_keypoints.pkl", 'rb') as f:
        out_keypoints = pkl.load(f)
      with open(f"{path}_out_edges.pkl", 'rb') as f:
        out_edges = pkl.load(f)
      data_all["out_keypoints"][path] = out_keypoints
      data_all["out_edges"][path] = out_edges
    except FileNotFoundError:
      print("FileNotFoundError, Skipping", path)
      skippedct += 1
      skippedpaths.append(path)
  
  return data_all, skippedct, skippedpaths

