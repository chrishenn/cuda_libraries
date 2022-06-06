import os

import torch as t
t.ops.load_library(os.path.split(__file__)[0] + "/build/libsegmenter.so")


def get_segments(edges, imgid, batch_size, angles, seg_thresh):

        return t.ops.segmenter_op.segmenter_kernel(edges, imgid, batch_size, angles, seg_thresh)[0]