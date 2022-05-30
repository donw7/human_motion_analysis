def edges_subplots(frame_idx, out_edges):
    # function to troubleshoot computations and shape manipulations of previous draw tools e.g. _draw_subplot2()
    # previously seemed to have issues mean goes to zero for some reason frame_idx = 11, 33

    # load out_edges collected from previous from previous model inference
    print(f"out_edges[frame_idx, :, :, :] = {out_edges[frame_idx, :, :, :]}") 
    print(f"out_edges[frame_idx, :, :, :].shape = {out_edges[frame_idx, :, :, :].shape}") # (18, 2, 2)

    # calculate mean of start-stop points => (18, 2)
    print(f"np.mean(out_edges, axis=2)[frame_idx,:,:] = {np.mean(out_edges, axis=2)[frame_idx,:,:]}") 
    print(f".shape = {np.mean(out_edges, axis=2)[frame_idx,:,:].shape}") # (18, 2)

    # check saved variable
    print(f"edge_means[frame_idx,:,:] = {edge_means[frame_idx,:,:]}")
    print(f"edge_means[frame_idx,:,:].shape = {edge_means[frame_idx,:,:].shape}")

    # absolute value of edge velocities
    print(f"abs(edge_vel[frame_idx,:,:]) = {abs(edge_vel[frame_idx,:,:])}")
    print(f"abs(edge_vel[frame_idx,:,:]).shape = {abs(edge_vel[frame_idx,:,:]).shape}")

    # mask of edges
    print(f"mask_edge[frame_idx,:,:] = {mask_edge[frame_idx,:,:]}")
    print(f"mask_edge[frame_idx,:,:].shape = {mask_edge[frame_idx,:,:].shape}")

    # sum of logical vector
    print(f"sum(mask_edge[1,17,:].reshape(-1)) = {sum(mask_edge[1,17,:].reshape(-1))}")