import torch as t

from numba import njit, prange
import numpy as np

if t.cuda.is_available():
    try: import frnn_bipart_cuda
    except: print("using cpu . . .")



def get_frnn_bipart(data, lin_radius, scale_radius, batch_size):
    '''
    where pts = data[1] gives by row [y,x,z,angle,size,group_id] where group_id is one of two values, and only edges between
        objects from different groups will be returned.
    '''

    pts, imgid = data[1], data[2]

    if pts.is_cuda:
        edges = frnn_bipart_cuda.frnn_bipart_kernel(pts, imgid, lin_radius, scale_radius, batch_size)[0]

    else:
        if isinstance(lin_radius, t.Tensor): lin_radius = lin_radius.item()
        if isinstance(scale_radius, t.Tensor): scale_radius = scale_radius.item()
        edges = frnn_bipart_cpu(pts.numpy(), imgid.numpy(), lin_radius, scale_radius)

        edges = t.tensor(edges)
        edges = edges[edges[:, 0] + edges[:, 1] >= 0]

    assert edges.size(0) > 0
    return edges


@njit(parallel=True)
def frnn_bipart_cpu(pts, imgid, lin_radius, scale_radius):
    """
    :param pts:   where pts[row] = [y, x, z, angle, scale, group_id] for a given point
    """
    size = pts.shape[0]

    loops = size*size
    edges = np.subtract( np.zeros((loops, 2), dtype=np.int32), 1)

    for i in prange( loops ):

        ptid_0 = int( np.divide( i, size ) )
        ptid_1 = int( np.mod( i, size ) )

        p0 = pts[ptid_0, :]
        p1 = pts[ptid_1, :]

        groupid_0 = p0[5]
        groupid_1 = p1[5]
        imgid0 = imgid[ptid_0]
        imgid1 = imgid[ptid_1]

        if imgid0 == imgid1 and groupid_0 != groupid_1 and ptid_0 > ptid_1:

            diff_yxz = p0[np.array([0,1,2])] - p1[np.array([0,1,2])]
            dist_yxz = np.sqrt( np.sum( np.power(diff_yxz, 2) ) )

            size_0 = p0[4]
            size_1 = p1[4]
            diff_s = np.abs( np.log(size_0) - np.log(size_1) )

            if diff_s < scale_radius and dist_yxz < np.multiply(lin_radius, np.sqrt( np.multiply(size_0, size_1) ) ):
                edges[i] = np.array( [ptid_0, ptid_1] )
    return edges




























