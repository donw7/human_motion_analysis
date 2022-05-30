import numpy as np

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




