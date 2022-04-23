from oomodel import oodl_draw
from test.context import get_test_loader

from conn_comp import conncom_bind
from frnn import frnn_bind

import unittest
import time

import torch as t

from numba import njit, prange


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


class conncom_test(unittest.TestCase):

    def test_timings(self):
        print('\n\nrunning test_timings')
        t.cuda.set_device(0)
        t.manual_seed(7)
        t.cuda.manual_seed_all(7)

        n_loops =       150
        batch_size =    t.tensor(80).cuda()
        lin_radius =    t.tensor(2).cuda()
        scale_radius =  t.tensor(1.0).cuda()

        loader = get_test_loader(batch_size)

        times = t.zeros(n_loops, dtype=t.float)
        sizes = t.zeros(n_loops, dtype=t.int)

        test = 0
        for p_drop_o in [0.25, 0.3, 0.5, 0.6, 0.75]:
            for p_drop_e in [0, 0.25, 0.5, 0.75, 0.9]:
                for loop, (data, target) in enumerate(loader):
                    data = get_objects(data.cuda())
                    mask = t.rand(data[0][:,0].size(0), device=data[0].device).ge(p_drop_o)
                    data = tuple(te[mask] for te in data)

                    edges = frnn_bind.get_frnn(data, lin_radius, scale_radius, batch_size)
                    mask = t.rand(edges[:,0].size(0), device=edges.device).ge(p_drop_e)
                    edges = edges[mask]

                    tic = time.time()
                    t.cuda.synchronize()
                    cc_ids = conncom_bind.get_conn_comp(edges, data[2])
                    t.cuda.synchronize()
                    times[loop] = time.time() - tic
                    sizes[loop] = cc_ids.size(0)

                    correct = check_ccids(edges.cpu().numpy(), cc_ids.cpu().numpy())
                    self.assertEqual(correct, edges.size(0), 'wrong cc_ids')

                    if (loop+1) % 10 == 0: print("loop: ", loop)
                    if loop+1 == n_loops: break
                print("\t\t test: ", test)
                test += 1

                # vis_merge.vis_seg(0, data[1], data[2], groupids=cc_ids, edges=edges, draw_obj=True, linewidths=0.9)

        print('times: ', times[2:].mul_(1000), ' (msec)')
        print('times mean: ', times[2:].mean().item(), '(msec)\n')

        print('sizes: ', sizes)

@njit(parallel=True)
def check_ccids(edges, cc_ids):

    correct = 0
    loops = edges.shape[0]
    for i in prange(loops):

        edge = edges[i]
        if cc_ids[edge[0]] == cc_ids[edge[1]]: correct += 1
        else: print(i)

    return correct

if __name__ == '__main__':
    unittest.main()



























