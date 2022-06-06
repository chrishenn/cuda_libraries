from test.context import get_test_loader
import oomodel.oovis as oovis

from frnn import frnn_bind
import edge_topim_cuda

import math
import unittest
import numpy as np
import time

from numba import njit, prange

import torch as t
import torch.autograd.profiler


def get_objects(data):
    device = data.device

    D1 = t.arange(data.size(2), dtype=t.int)
    D2 = t.arange(data.size(3), dtype=t.int)

    gridy, gridx = t.meshgrid(D1,D2)
    gridy = gridy.unsqueeze(0).unsqueeze(0).repeat(data.size(0),1,1,1).float().to(device)
    gridx = gridx.unsqueeze(0).unsqueeze(0).repeat(data.size(0),1,1,1).float().to(device)
    zeros = t.zeros_like(gridx)
    ones =  t.ones_like(gridx)

    angles = t.zeros_like(gridx)
    geom = t.cat([gridy, gridx, zeros, angles, ones, zeros], 1)

    imgid = t.arange(data.size(0), device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,data.size(2), data.size(3))

    texture =   data.permute([0,2,3,1])
    geom =      geom.permute([0,2,3,1])
    imgid =   imgid.permute([0,2,3,1])

    texture =   texture.reshape(-1,texture.size(3))
    geom =      geom.reshape(-1,geom.size(3))
    imgid =   imgid.reshape(-1,imgid.size(3))
    imgid =   imgid.squeeze(1)

    return texture.to(device), geom.to(device), imgid.to(device)


class edge_topim_test(unittest.TestCase):

    def test_timings(self):
        print('\n\nrunning test_timings')
        t.cuda.set_device(0)

        n_loops =       500
        batch_size =    t.tensor(80).cuda()
        lin_radius =    t.tensor(2.8).cuda()
        scale_radius =  t.tensor(1.0).cuda()

        loader = get_test_loader(batch_size)

        times = t.zeros(n_loops, dtype=t.float)
        sizes = t.zeros(n_loops, dtype=t.int)

        for loop, (data, target) in enumerate(loader):
            data = get_objects(data.cuda())

            edges = frnn_bind.get_frnn(data, lin_radius, scale_radius, batch_size)
            weight_b4 = t.rand_like(edges[:,0].float())

            t.cuda.synchronize()
            tic = time.time()
            keep_ids = edge_topim_cuda.edge_topim_kernel(edges, weight_b4, data[2], batch_size, 0.1)[0]
            t.cuda.synchronize()
            times[loop] = time.time() - tic
            sizes[loop] = keep_ids.size(0)

            if loop+1 == n_loops: break

        print('times: ', times.mul_(1000), ' (msec)')
        print('times mean: ', times.mean().item(), '(msec)\n')

        print('sizes: ', sizes)

if __name__ == '__main__':
    unittest.main()



























