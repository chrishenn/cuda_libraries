import torch as t

from numba import jit, njit, prange
import numpy as np

import time



if t.cuda.is_available():
    try:
        import frnn_cuda
    except:
        print("using cpu . . .")

# pts = [y, x, z, angle, scale, pt_id]
def get_frnn( pts, radius, scale_radius, img_ids, device, add_identity=False ):
    """
    Fixed-radius nearest neighbor (get_frnn):
    Returns edges that define sets of fixed-radius nearsest neighbors, in a point-vector graph defined by pts.
    Each point [y, x, z, angle, scale, pt_id] has 3D coordinates [y, x, z] and a scale (size) used for comparison.
        Distance comparisons are done by a fixed radius (scale_radius) for scales after logarithmic binning, such that higher scale values
        have a wider range of scale values and (x, y) values that can be considered nearest neighbors.
    Nearest neighbors must also be within a certain distance in 3D space, as determined by radius*(harmonic mean of the
        two points' scales).
    A pair of points will only be considered nearest neighbors if they are in the same image, as mapped from their
        row in pts to their image identifier in img_ids.
    """

    if t.cuda.is_available():

        edges = frnn_gpu( pts, img_ids, radius, scale_radius, device )

    else:

        pts[:, 4] = pts[:, 4].log_()
        pts[:, 4] = pts[:, 4].sub_( pts[:, 4].min() )

        imgid_pts = t.cat([img_ids.float().unsqueeze(1), pts], dim=1)

        edges = frnn_cpu(imgid_pts, pts.size(0), radius, scale_radius)

        edges = t.tensor(edges)
        edge_ids = t.arange(edges.size(0))
        edge_ids = edge_ids.masked_select(edges[:, 0].ne(-1))
        edges = edges.index_select(0, edge_ids)
        edges = edges.sort(1)[0]

        if edges.size(0) > 0:
            edges = edges.unique(0)
        else:
            print('no comparisons binned')
            edges = t.tensor([], dtype=t.int, device=device)

    return edges


def frnn_gpu( pts, img_ids, radius, scale_radius, device ):

    xs = pts[:, 1]
    xmin = xs.min().item()
    xmax = xs.max()

    scales = pts[:, 4].log()
    scalemin = scales.min().item()
    scalemax = scales.max()
    scalemax = scalemax + 0.0000001

    num_scabin = scalemax.sub(scalemin).div(scale_radius).int().add(1).item()
    scabin_cutoffs = t.linspace(scalemin+scale_radius, scalemax, num_scabin, device=device)

    radius_at_each_scabin = t.exp(scabin_cutoffs).mul_(radius)

    num_xbins_each_scale = (xmax - xmin).div(radius_at_each_scabin).int().add(1)

    scabin_offsets = t.cat([t.tensor([0], dtype=t.int, device=device), num_xbins_each_scale])

    scabin_offsets = t.cumsum(scabin_offsets, 0, dtype=t.int)

    num_imgs = img_ids.max().item()+1
    num_bins_perimg = scabin_offsets[-1].item()
    img_ids = img_ids.int()

    if not isinstance(device, int): device = device.index

    t.cuda.synchronize(device)
    tic=time.time()

    edges = frnn_cuda.frnn_kernel(pts,
                                  img_ids,
                                  radius_at_each_scabin,
                                  scabin_offsets,

                                  num_imgs,
                                  num_bins_perimg,

                                  device,
                                  radius,
                                  scale_radius,
                                  xmin,
                                  scalemin
                                  )
    t.cuda.synchronize(device)
    print(time.time()-tic)

    try: edges = edges[0].long()
    except: edges = edges[0].long()

    return edges


@njit(parallel=True)
def frnn_cpu(imgid_pts, size, radius, scale_radius):
    """
    :param imgid_pts:   where imgid_pts[row] = [img_id, y, x, z, angle, scale, pt_id] for a given point
    """

    loops = size*size
    edges = np.subtract( np.zeros((loops, 2), dtype=np.int32), 1)

    for i in prange( loops ):

        i0 = int( np.divide( i, size ) )
        i1 = int( np.mod( i, size ) )

        p0 = imgid_pts[i0, :]
        p1 = imgid_pts[i1, :]

        ptid0 = p0[4]
        ptid1 = p1[4]
        imgid0 = p0[0]
        imgid1 = p1[0]

        if imgid0 == imgid1 and ptid0 != ptid1:

            diff_xyza = np.fabs( p0[1:5] - p1[1:5] )
            dist_xyz = np.sqrt( np.sum( np.power(diff_xyza[:-1], 2) ) )

            if diff_xyza[-1] < scale_radius and dist_xyz < np.multiply(radius, np.sqrt( np.multiply(p0[5], p1[5]) ) ):
                edges[i] = np.array( [ptid0, ptid1] )

    return edges


@njit(parallel=True)
def frnn_cpu_himem(imgid_pts, pairs, radius, scale_radius):
    """
    :param imgid_pts:   where imgid_pts[row] = [img_id, y, x, z, angle, scale, pt_id] for a given point
    :param pairs:       (n x 2): all pairs of indexes into batid_ptid_pt for brute force compares
    """

    loops = len(pairs)
    edges = np.subtract( np.zeros((loops, 2), dtype=np.int32), 1)

    for i in prange( loops ):

        i0 = pairs[i][0]
        i1 = pairs[i][1]

        p0 = imgid_pts[i0, :]
        p1 = imgid_pts[i1, :]

        ptid0 = p0[4]
        ptid1 = p1[4]
        imgid0 = p0[0]
        imgid1 = p1[0]

        if imgid0 == imgid1 and ptid0 != ptid1:

            diff_xyza = np.fabs( p0[1:5] - p1[1:5] )
            dist_xyz = np.sqrt( np.sum( np.power(diff_xyza[:-1], 2) ) )

            if diff_xyza[-1] < scale_radius and dist_xyz < np.multiply(radius, np.sqrt( np.multiply(p0[5], p1[5]) ) ):
                edges[i] = np.array( [ptid0, ptid1] )

    return edges


# Helper function for frnn_cpu_himem, generating all size**2 pairs of indexes for brute-force comparisons on cpu
# Faster, but with big time-space tradoff in system memory
@njit(parallel=True)
def get_pairs(size):

    pairs = np.empty((size*size, 2))

    for i in prange(size):
        for j in prange(size):
            pairs[i*size + j][0] = i
            pairs[i*size + j][1] = j

    return pairs