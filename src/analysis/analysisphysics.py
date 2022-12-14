import numpy as np
from itertools import groupby
from matplotlib.path import Path
import matplotlib.patches as patches

def compute_edge_velocities(out_edges, EDGE_VEL_THRESH):
    # Calculate edge velocities and feed back in as mask (if run inference already)
    
    if sum(out_edges.reshape(-1)) != 0:
        # (numframes, 18 joints, 2 points, 2 coords)
        # (42, 18, 2, 2) shape
        edge_means = np.mean(out_edges, axis=2) # mean of start-stop points => (42, 18, 2)

        # Get velocity by taking diff across time axis  => (41, 18, 2) + pad first frame => (42, 18, 2)
        edge_vel = np.zeros(edge_means.shape)
        edge_vel[1:,:,:] = abs(np.diff(edge_means, axis=0))
        
        # Set threshold for edge mask based on e.g. velocity after visualizing via histogram as below
        mask_edge = edge_vel > EDGE_VEL_THRESH

    return mask_edge


def get_segments(anom_idx) -> list:
    segments = []
    for k, g in groupby(enumerate(anom_idx), lambda x: x[1]):
        if k == 1:
            indices = list(map(lambda x: x[0], g))
            start = indices[0]
            stop = indices[-1]
            segments.append((start, stop))
    return segments

def plot_patch(ax, start_stop) -> None:
	verts = [
		(start_stop[0], 0.),  # left, bottom
		(start_stop[0], 200.),  # left, top
		(start_stop[1], 200.),  # right, top
		(start_stop[1], 0.),  # right, bottom
		(0., 0.),  # ignored
	]
	codes = [
			Path.MOVETO,
			Path.LINETO,
			Path.LINETO,
			Path.LINETO,
			Path.CLOSEPOLY,
	]
	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor='red', lw=0, alpha=0.2)
	ax.add_patch(patch)

def plot_patchline(ax, idx) -> None:
	verts = [
		(idx, 0.),  # left, bottom
		(idx, 200.),  # left, top
		(idx, 200.),  # right, top
		(idx, 0.),  # right, bottom
		(0., 0.),  # ignored
	]
	codes = [
			Path.MOVETO,
			Path.LINETO,
			Path.LINETO,
			Path.LINETO,
			Path.CLOSEPOLY,
	]
	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor='black', lw=3)
	ax.add_patch(patch)

