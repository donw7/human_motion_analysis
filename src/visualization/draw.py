# !pip install -q imageio
# !pip install -q opencv-python
# !pip install -q git+https://github.com/tensorflow/docs

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection # likely limited to draw module
import matplotlib.patches as patches
import cv2

from src.analysis import analysisphysics
import imageio



def _keypoints_and_edges_for_display(keypoints_with_scores,
                                    CONFIG_EDGE_COLORS,
                                    height,
                                    width,
                                    keypoint_threshold=0.01):
  """Returns high confidence keypoints and edges for visualization.
  
  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.
      # note initial OTB keypoint_threshold=0.11

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in CONFIG_EDGE_COLORS.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

# image-wise
def draw_prediction_on_image(
    image, keypoints_with_scores, CONFIG_EDGE_COLORS, crop_region=None, close_figure=True,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, CONFIG_EDGE_COLORS, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  if close_figure:
    plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

# image-wise
def _draw_subplot2(image, image_from_plot, keypoints_with_scores, CONFIG_EDGE_COLORS, mask_edge=None, figsize=(10,10)):
  '''like draw_prediction_on_image, but with an additional subplot of abstracted image
  expects image_from_plot input from draw_prediction_on_image'''

  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize) # fig w/ 2 subplots
  axs[0].imshow(image_from_plot) # first subplot display image already passed in

  # Remove axes
  axs[0].axis('off')
  
  line_segments = LineCollection([], linewidths=(2), linestyle='solid')
  axs[1].add_collection(line_segments)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, CONFIG_EDGE_COLORS, height, width)

  # add edge mask based on e.g. velocity
  # shape = (18, 2)
  # some edge colors will get lost d/t confidence threshold
  #   and for these the whole frame is skipped and mask not applied
  if mask_edge is not None: # and mask_edge.shape[1] - 1 == len(edge_colors):
    for i in range(1, mask_edge.shape[0] - 1):
      if sum(mask_edge[i,:].reshape(-1)) != 0:
        try:
          edge_colors[i] = 'r'
        except IndexError:
          print(len(edge_colors), i)
        
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    axs[1].plot(keypoint_locs[:, 0], keypoint_locs[:,1], 'o')

  # Remove axes
  axs[1].axis('off')

  # Set right subplot's aspect ratio to that of left (from imshow)
  axs[1].set_aspect(aspect_ratio) 

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))


  plt.close(fig)

  # if output_image_height is not None:
  #   output_image_width = int(output_image_height / height * width)
  #   image_from_plot = cv2.resize(
  #       image_from_plot, dsize=(output_image_width, output_image_height),
  #        interpolation=cv2.INTER_CUBIC)
  return image_from_plot

# images(i.e. decoded)-wise
def wrap_draw_subplots(images, out_keypoints, out_edges, CONFIG_EDGE_COLORS, mask_edge=None, figsize=(10,10)):
  '''wrapper for draw_prediction_on_image and draw_subplot2'''
  
  num_frames, image_height, image_width, _ = images.shape
  output_images = [] # 1 plot on raw
  output_images2 = [] # with 2 subplot

  # Draw with or without annotations (mask & subplot 2)
  for frame_idx in range(num_frames):

    # Load keypoints from previous model inference (note with time axis)
    keypoints_with_scores = out_keypoints[frame_idx,:,:,:,:]

    # Draw the coordinates on the raw frames
    images_from_plot = draw_prediction_on_image(
        images[frame_idx, :, :, :].numpy().astype(np.int32),
        keypoints_with_scores, CONFIG_EDGE_COLORS, crop_region=None,
        close_figure=True, output_image_height=image_height) #output_image_height=300)
    
    # Gather a list of images after plotted on raw frames
    output_images.append(images_from_plot)

    # Feed in edge velocities in as mask (if run inference already)
    # ETC: try to make this compatible with feeding in any mask and up/downstream processing
    if sum(out_edges.reshape(-1)) != 0:
      # Draw other representation on right and append to image stack
      output_images2.append(
        _draw_subplot2(images[frame_idx, :, :, :],
        images_from_plot, keypoints_with_scores, CONFIG_EDGE_COLORS, mask_edge[frame_idx,:,:]))

    else: # Don't feed in mask_edge, default = None
      # Draw other representation on right and append to image stack
      output_images2.append(
      _draw_subplot2(images[frame_idx, :, :, :],
      images_from_plot, keypoints_with_scores, CONFIG_EDGE_COLORS)) 
    
  return output_images, output_images2
