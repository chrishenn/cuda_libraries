import os
import sys

import torch
import torch as t
import time

from numba import jit, njit, prange
import numpy as np

import frnn_cuda


if t.cuda.is_available():
    try:
        import frnn_cuda
    except:
        print("using cpu . . .")

def get_frnn(data, lin_radius, scale_radius, batch_size):
    """
    Fixed-radius nearest neighbor (get_frnn):
    Returns edges that define sets of fixed-radius nearsest neighbors, in a point-vector graph defined by pts.
    Each point [y, x, z, angle, scale, pt_id] has 3D coordinates [y, x, z] and a scale (size) used for comparison.
        Distance comparisons are done by a fixed radius (scale_radius) for scales after logarithmic binning, such that higher scale values
        have a wider range of scale values and (x, y) values that can be considered nearest neighbors.
    Nearest neighbors must also be within a certain distance in 3D space, as determined by lin_radius*(harmonic mean of the
        two points' scales).
    A pair of points will only be considered nearest neighbors if they are in the same image, as mapped from their
        row in pts to their image identifier in imgid.
    """
    pts, imgid = data[1], data[2]

    if pts.is_cuda:

        try:
            aout = frnn_cuda.frnn_kernel(pts, imgid, lin_radius, scale_radius, batch_size)
            edges = aout[0]
        except:
            print("frnn fail. ")

    else:
        imgid_pts = t.cat([imgid.float().unsqueeze(1), pts], dim=1)

        edges, n_edges = frnn_cpu(imgid_pts.numpy(), pts.size(0), lin_radius, scale_radius)

        edges = t.tensor(edges)
        edge_ids = t.arange(edges.size(0))
        edge_ids = edge_ids.masked_select(edges[:, 0].ne(-1))
        edges = edges.index_select(0, edge_ids)

    assert edges.size(0) > 0
    return edges




@njit(parallel=True)
def frnn_cpu(imgid_pts, size, lin_radius, scale_radius):
    """
    :param imgid_pts:   where imgid_pts[row] = [img_id, y, x, z, angle, scale, pt_id] for a given point
    """

    loops = size*size
    edges = np.subtract( np.zeros((loops, 2), dtype=np.int32), 1)
    written = 0

    for i in prange( loops ):

        i0 = int( np.divide( i, size ) )
        i1 = int( np.mod( i, size ) )

        p0 = imgid_pts[i0, :]
        p1 = imgid_pts[i1, :]

        ptid0 = p0[6]
        ptid1 = p1[6]
        imgid0 = p0[0]
        imgid1 = p1[0]

        if imgid0 == imgid1 and ptid0 > ptid1:

            diff_yxzs = np.fabs( p0[np.array([1,2,3,5])] - p1[np.array([1,2,3,5])] )
            dist_yxz = np.sqrt( np.sum( np.power(diff_yxzs[:-1], 2) ) )

            if diff_yxzs[-1] < scale_radius and dist_yxz < np.multiply(lin_radius, np.sqrt( np.multiply(p0[5], p1[5]) ) ):
                edges[i] = np.array( [ptid0, ptid1] )
                written += 1

    return edges, written
