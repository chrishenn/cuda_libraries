from test.context import get_objects
from test.context import get_test_loader

import unittest
import time
import numpy as np

import torch as t

from frnn_brute.frnn_bind import get_frnn
from segmenter.seg_bind import get_segments






class segmenter_test(unittest.TestCase):

    def test_seg_timings(self):
        print('\n\nrunning test_seg_timings')
        dev = 0
        device = t.cuda.set_device(dev)

        n_loops =       8
        batch_size =    t.tensor(80,device=dev)
        scale_radius =  1.0
        seg_thresh = t.tensor(1.1,device=dev)

        loader = get_test_loader(batch_size)

        times = t.empty([0], dtype=t.float32)
        test = 0
        for lin_radius, p_drop in zip([1.45], [0.5]):
            for scale_multiplier in [1]:
                for vec_size in [2,4,16,32, 64, 128]:
                    lin_radius, scale_radius = t.tensor(lin_radius, device=dev), t.tensor(scale_radius, device=dev)

                    loop = 0
                    for batch_idx, (data, target) in enumerate(loader):

                        tex, pts, imgid = get_objects(data.cuda())
                        pts[:,4] = t.rand_like(pts[:,4]).mul_(scale_multiplier).add_(1)
                        pts[:,:2].add_( t.rand_like(pts[:,:2]).sub(0.5).mul(2) )

                        mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                        pts, imgid = pts[mask], imgid[mask]

                        edges = get_frnn((None,pts,imgid), lin_radius, scale_radius, batch_size)
                        angles = t.rand([pts.size(0), vec_size], device=dev).sub(0.5).mul(2).mul(np.pi)

                        t.cuda.synchronize(device)
                        tic = time.time_ns()
                        seg_ids = get_segments(edges, imgid, batch_size, angles, seg_thresh)
                        t.cuda.synchronize(device)
                        times = t.cat([times, t.tensor([time.time_ns() - tic])])
                        print("num seg_ids unique: ", seg_ids.unique().size(0))

                        loop += 1
                        if loop == n_loops: break

                    test += 1
                    print("test ",test)

        times = times[10:]
        times.mul_(1e-6)
        print('times: ', times, ' (msec)')
        print('times mean: ', times.mean().item(), '(msec)\n')



if __name__ == '__main__':
    unittest.main()



























