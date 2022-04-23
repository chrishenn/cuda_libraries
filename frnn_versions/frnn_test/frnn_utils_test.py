from context import frnn_cuda
from context import frnn
from context import oomodel

from oomodel import oogen
import oomodel.oogen as oogen
import frnn.frnn_utils as fru
from frnn.frnn_bind import frnn_cpu, get_pairs, get_frnn

import unittest
import numpy as np
import warnings
import time

from numba import jit, njit, prange

import torch as t
import torch.utils.data as data
from torchvision import datasets, transforms
t.manual_seed(7)



def lamb(x): return np.multiply(x, 255)

def get_loaders(batch_size, n_workers=11, train=True, datasize=300):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lamb)])

    dataset = datasets.CIFAR10(root='~/dataset/brainkit/',
                                        train=train,
                                        download=True,
                                        transform=transform)
    if datasize is not None:
        dataset = data.random_split(dataset, [datasize, len(dataset) - datasize])[0]

    loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=n_workers,
                                   pin_memory=False)
    return loader

@njit(parallel=True)
def count_binnable(edges, nebs2d):

    corr_count = 0
    loops = len(edges[:,0])

    for i in prange(loops):
        ed = edges[i]
        mask0 = np.equal( nebs2d, ed[0] )
        mask1 = np.equal( nebs2d, ed[1] )

        ms = np.add(mask0, mask1).sum(1)

        if ms.max() > 1: corr_count += 1

    return corr_count



class frnn_utils_test(unittest.TestCase):



    # Enforce correct ranges and shapes for nebs2d pointers into scaled_pts data
    def test_types_ranges(self):
        print('\n\nrunning test_types_ranges')
        device = t.cuda.current_device()

        batch_size =    100
        scale_radius =  0.03
        dim_size =      32

        loader = get_loaders(batch_size)

        for batch_idx, (data, target) in enumerate(loader):

            texture, shape, pts, img_ids = oogen.pix_object(data, dim_size)

            nebs2d, scaled_pts = fru.get_kernel_inputs(pts, img_ids, scale_radius, device)

            self.assertTrue( t.max(nebs2d).item() < scaled_pts.size(0) )
            self.assertTrue( t.min(nebs2d).item() > -2 )

            self.assertTrue( scaled_pts.size(1) == 3 )
            self.assertTrue( len(nebs2d.shape) == 2 )

            self.assertTrue( nebs2d.device.index == device )
            self.assertTrue( nebs2d.is_contiguous() )
            self.assertEqual( nebs2d.size(1)%2, 0, "nebs2d width is not an even integer; frnn_kernel requires an even width")

            self.assertTrue( pts.device.index == device )
            self.assertTrue( pts.is_contiguous() )

            print('pass')
            break


    # Enforce that all known-valid edges have points that are searchable from frnn_utils binning in nebs2d
    def test_valid_edges_binned_to_nebs(self):
        print('\n\nrunning test_valid_edges_binned_to_nebs')
        device = t.cuda.current_device()

        n_loops =           8
        batch_size =        30
        radius =            5.
        scale_radius =      1.0
        img_size =          25
        uese_pix_objects =  True

        loader = get_loaders(batch_size)

        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            if uese_pix_objects: texture, shape, pts, img_ids = oogen.pix_object(data, img_size)
            else: texture, shape, pts, img_ids = oogen.sift_object(data)
            texture, shape, pts, img_ids = texture.to(device), shape.to(device), pts.to(device), img_ids.to(device)

            nebs2d, scaled_pts = fru.get_kernel_inputs(pts, img_ids, scale_radius, device)


            ids = t.arange(scaled_pts.size(0), dtype=t.int32).unsqueeze(1)
            batid_ptid_pt = t.cat( [img_ids.float().unsqueeze(1).cpu(), ids.float(), scaled_pts.cpu()], dim=1)

            pairs = get_pairs(scaled_pts.size(0))
            b_edges = frnn_cpu(batid_ptid_pt.numpy(), pairs.astype(np.int32), radius, scale_radius)

            b_edges =   t.tensor(b_edges)
            edge_ids =  t.arange(b_edges.size(0))
            edge_ids =  edge_ids.masked_select( b_edges[:,0].ne(-1) )
            b_edges =   b_edges.index_select(0, edge_ids)
            b_edges =   b_edges.sort(1)[0]
            b_edges =   b_edges.unique(sorted=True, dim=0)



            corr_count = count_binnable(b_edges.numpy(), nebs2d.cpu().numpy())

            print('known-valid edges searchable from nebs2d: ', corr_count, ' / ', b_edges.size(0))
            self.assertTrue( corr_count == b_edges.size(0), 'frnn_utils returned a binning that would miss a valid edge')

            loop += 1
            if loop == n_loops: break



    # time runs of calls to frnn_utils; print times, average time
    def test_utils_timing(self):
        print('\n\nrunning test_utils_timing')
        device = t.cuda.current_device()

        n_loops =           8
        batch_size =        100
        scale_radius =      1.0
        img_size =          25
        use_pixel_objects = True

        loader = get_loaders(batch_size)

        times = t.zeros(n_loops, dtype=t.float32)
        i_ins = t.zeros(n_loops, dtype=t.int32)
        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            if use_pixel_objects: texture, shape, pts, img_ids = pix_object(data, img_size)
            else: texture, shape, pts, img_ids = extract_sift_feat(data, device)
            texture, shape, pts, img_ids = texture.to(device), shape.to(device), pts.to(device), img_ids.to(device)

            t.cuda.synchronize(device)
            tic = time.time()
            nebs2d, scaled_pts = fru.get_kernel_inputs(pts.to(device), img_ids.to(device), scale_radius, device)
            t.cuda.synchronize(device)
            times[loop] = time.time() - tic
            i_ins[loop] = scaled_pts.size(0)

            print('loop ', loop)
            loop += 1
            if loop == n_loops: break

        print('frnn_utils run times (s): ', times)
        print('average run time: ', times.mean().item())
        print('frnn_utils num objects binned (s): ', i_ins)


    # Enforce that points binned for comparison are all in the same image in this batch
    def test_compare_imgid_same(self):
        print('\n\nrunning test_compare_imgid_same')
        device = t.cuda.current_device()

        n_loops =           7
        batch_size =        100
        scale_radius =      15.0
        img_size =          25
        use_pixel_objects = True

        loader = get_loaders(batch_size)

        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            if use_pixel_objects: texture, shape, pts, img_ids = pix_object(data, img_size)
            else: texture, shape, pts, img_ids = extract_sift_feat(data, device)
            texture, shape, pts, img_ids = texture.to(device), shape.to(device), pts.to(device), img_ids.to(device)

            nebs2d, scaled_pts = fru.get_kernel_inputs(pts, img_ids, scale_radius, device)

            for row in nebs2d:

                bid = None
                for ptid in row:

                    if ptid == -1: continue

                    if bid is None:
                        bid = img_ids[ptid].item()
                    else:
                        self.assertTrue( bid == img_ids[ptid].item(), 'objects in different images binned for comparison')


            print('loop: ', loop)
            loop += 1
            if loop == n_loops:
                print('pass')
                break







if __name__ == '__main__':
    unittest.main()
