from context import frnn_cuda
from context import frnn_bin_cuda
from context import frnn
from context import oomodel

from oomodel import oogen
from oomodel.oogen import pix_object
import frnn.frnn_utils as fru

import unittest
import numpy as np
import warnings
import time
import os

from numba import jit, njit, prange
import numba.cuda as cuda

import torch as t
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.autograd.profiler
t.manual_seed(7)



def extract_sift_feat(minibatch, device):
    text, sha, geom, identi = oogen.sift_object(minibatch,
                                                already_0_255=True,
                                                contrastThreshold=0.01,
                                                edgeThreshold=20,
                                                sigma=2.0)

    return (text.to(device), sha.to(device), geom.to(device), identi.to(device))

def lamb(x): return np.multiply(x, 255)

def get_utest_loader(batch_size):
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Lambda(lamb)
         ])

    train_dataset = datasets.CIFAR10(root='~/dataset/brainkit/',
                                        train=True,
                                        download=True,
                                        transform=transform)
    train_dataset = data.random_split(train_dataset, [1000, len(train_dataset) - 1000])[0]
    train_loader = data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=False)
    return train_loader

@njit(parallel=True)
def count_binnable(edges, nebs2d):

    corr_count = 0
    loops = len(edges[:,0])

    for i in range(loops):
        ed = edges[i]
        mask0 = np.equal( nebs2d, ed[0] )
        mask1 = np.equal( nebs2d, ed[1] )

        ms = np.add(mask0, mask1).sum(1)

        if ms.max() > 1: corr_count += 1

    return corr_count


class frnn_bin_test(unittest.TestCase):

    def test_frnn_bin_kernel_timings(self):
        print('running test_frnn_bin_kernel_timings')
        warnings.filterwarnings("ignore")
        device = 0

        n_loops =           8
        batch_size =        120
        scale_radius =      1.0
        img_size =          32
        use_pix_objects =   True

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        loader = get_utest_loader(batch_size)

        times = t.zeros(n_loops, dtype=t.float32)
        i_outs = t.zeros(n_loops, dtype=t.int32)
        i_ins = t.zeros(n_loops, dtype=t.int32)
        loop = 0


        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for batch_idx, (data, target) in enumerate(loader):

                if use_pix_objects: texture, shape, pts, img_ids = pix_object(data, img_size)
                else: texture, shape, pts, img_ids = extract_sift_feat(data, device)
                texture, shape, pts, img_ids = texture.to(device), shape.to(device), pts.to(device), img_ids.to(device)

                scaled_pts = fru.get_scaled(pts, device)

                # t.cuda.synchronize(device)
                tic = time.time()

                nebs2d = fru.get_nebs2d_gpu( scaled_pts, img_ids, scale_radius, device )

                # t.cuda.synchronize(device)
                times[loop] = float(time.time() - tic)
                i_outs[loop] = nebs2d.size(1)
                i_ins[loop] = scaled_pts.size(0)


                print('loop: ',loop)
                loop += 1
                if loop == n_loops: break


        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        print('times: ', times)
        print('times mean: ', times.mean(), '\n')
        print('objects searched: ', i_ins)
        print('bin_ids widths: ', i_outs)


if __name__ == '__main__':
    unittest.main()