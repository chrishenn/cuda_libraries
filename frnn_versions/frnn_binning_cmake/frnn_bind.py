import os
import torch as t

from numba import njit, prange
import numpy as np

# import frnn_cuda
t.ops.load_library(os.path.split(__file__)[0] + "/build/libfrnn_ts.so")



def get_frnn(data, lin_radius, scale_radius, batch_size):

    pts, imgid = data[1], data[2]

    if pts.is_cuda:
        edges = t.ops.frnn_cmake_op.frnn_ts_kernel(pts, imgid, lin_radius, scale_radius, batch_size)[0]

    else:
        if isinstance(lin_radius, t.Tensor):   lin_radius = lin_radius.item()
        if isinstance(scale_radius, t.Tensor): scale_radius = scale_radius.item()
        if isinstance(batch_size, t.Tensor):   batch_size = batch_size.item()

        # img_counts = imgid.bincount()
        # edges = frnn_cpu(pts.numpy(),imgid.numpy(),img_counts.numpy(), lin_radius, scale_radius, batch_size)
        edges = frnn_cpu(pts.numpy(),imgid.numpy(), lin_radius, scale_radius)
        edges = t.from_numpy(edges)

    # assert edges.size(0) > 0
    return edges


@njit(parallel=True)
def frnn_cpu(pts, imgid, lin_radius, scale_radius):

    size = pts.shape[0]
    loops = size*size
    edges = np.subtract( np.zeros((loops, 2), dtype=np.int32), 1)

    for i in prange( loops ):

        i0 = int( np.divide( i, size ) )
        i1 = int( np.mod( i, size ) )

        p0 = pts[i0, :]
        p1 = pts[i1, :]

        ptid0 = i0
        ptid1 = i1
        imgid0 = imgid[i0]
        imgid1 = imgid[i1]

        if imgid0 == imgid1 and ptid0 > ptid1:

            diff_yxz = p0[np.array([0,1,2])] - p1[np.array([0,1,2])]
            dist_yxz = np.sqrt( np.sum( np.power(diff_yxz, 2) ) )

            size0 = p0[4]
            size1 = p1[4]
            diff_s = np.abs( np.log(size0) - np.log(size1) )

            if diff_s < scale_radius and dist_yxz < np.multiply(lin_radius, np.sqrt( np.multiply(size0, size1) ) ):
                edges[i] = np.array( [ptid0, ptid1] )

    edges = edges[edges[:, 0] + edges[:, 1] > 0]
    return edges

@njit(parallel=True)
def frnn_cpu_conditional(pts, imgid, img_counts, lin_radius, scale_radius, batch_size):

    total_size = np.power(img_counts, 2).sum()
    edges = np.subtract( np.zeros((total_size, 2), dtype=np.int64), 1)

    img_obj_offsets = img_counts.cumsum() - img_counts
    im_edge_offsets = np.power(img_counts, 2)
    im_edge_offsets = im_edge_offsets.cumsum() - im_edge_offsets

    for img in prange(batch_size):
        size = img_counts[img]
        im_loops = np.power(size, 2)

        im_edge_offset = im_edge_offsets[img]
        im_obj_offset = img_obj_offsets[img]

        for i in prange( im_loops ):

            ptid0 = int( np.divide( i, size ) ) +im_obj_offset
            ptid1 = int( np.mod( i, size ) ) +im_obj_offset

            imgid0 = imgid[ptid0]
            imgid1 = imgid[ptid1]

            if imgid0 == imgid1 and ptid0 > ptid1:

                p0 = pts[ptid0, :]
                p1 = pts[ptid1, :]

                diff_yx = p0[np.array([0,1])] - p1[np.array([0,1])]
                dist_yx = np.sqrt( np.sum( np.power(diff_yx, 2) ) )

                size0 = p0[4]
                size1 = p1[4]
                diff_s = np.abs( np.log(size0) - np.log(size1) )

                if diff_s < scale_radius and dist_yx < np.multiply(lin_radius, np.sqrt( np.multiply(size0, size1) ) ):
                    edges[im_edge_offset+i] = np.array( [ptid0, ptid1] )

    edges = edges[edges[:, 0] + edges[:, 1] > 0]
    return edges
