import math

from test.context import get_objects
from test.context import get_test_loader

import oomodel.oovis as oovis



# from frnn_cmake.frnn_bind import frnn_cpu, get_frnn
from frnn_brute.frnn_bind import frnn_cpu, get_frnn

import unittest
import numpy as np
import warnings
import time

from numba import njit, prange

import torch as t
import torch.autograd.profiler
import torch.nn as nn






# Counts the number of edges in edges that index points in scaled_pts that are within scale_radius of each other in
#   scale space, and within radius*(harmonic mean of the points' scales) in 2D space, and have the same batch id.
# imgid_pts[row] = [img_id, y, x, z, angle, scale, pt_id]
@njit(parallel=True)
def frnn_check_edges(edges, imgid, pts, radius, scale_radius):
    #[img_id, y, x, z, angle, scale, pt_id]

    corr_lin_count = 0
    corr_scale_count = 0

    loops = len(edges[:,0])

    for i in prange(loops):
        edge = edges[i]
        ptid_0, ptid_1 = edge[0], edge[1]
        imgid_0, imgid_1 = imgid[ptid_0], imgid[ptid_1]

        p0, p1 = pts[ptid_0], pts[ptid_1]

        if imgid_0 == imgid_1 and ptid_0 != ptid_1:

            diff_yxz = np.fabs( p0[np.array([0,1,2])] - p1[np.array([0,1,2])] )
            dist_yxz = np.sqrt( np.sum( np.power(diff_yxz, 2) ) )

            size_0, size_1 = p0[4], p1[4]
            diff_s = np.abs( np.log(size_0) - np.log(size_1) )

            if dist_yxz < np.multiply(radius, np.sqrt( np.multiply(size_0, size_1) ) ):
                corr_lin_count += 1
            if diff_s < scale_radius:
                corr_scale_count += 1

    return corr_lin_count, corr_scale_count

# count how many edges are drawn between points with the same img_id (in the same image)
@njit(parallel=True)
def check_edges_images(edges, imgid_pts):

    loops = len(edges[:,0])
    corr_same_image = 0

    for i in prange(loops):
        ed = edges[i]
        p0 = imgid_pts[ ed[0] ]
        p1 = imgid_pts[ ed[1] ]

        if abs(p0[0] - p1[0]) < 0.1: corr_same_image += 1

    return corr_same_image

@njit(parallel=False)
def verify_bid2ptid(nebs, bid2ptid, b_edges):

    correct = 0
    for edge in b_edges:
        row0 = (edge[0] == bid2ptid[:,1]).nonzero()[0].item()
        row1 = (edge[1] == bid2ptid[:,1]).nonzero()[0].item()

        bin0 = bid2ptid[row0,0]
        bin1 = bid2ptid[row1,0]

        if bin0 == bin1 and (bin0==nebs[:,0]).sum()>0:
            correct += 1
        elif (bin0==nebs[:,0]).sum()>0 and (bin1==nebs[nebs[:,0]==(bin0)]).sum()>0:
            correct += 1
        elif (bin1==nebs[:,0]).sum()>0 and (bin0==nebs[nebs[:,0]==(bin1)]).sum()>0:
            correct += 1

    return correct

@njit(parallel=False)
def verify_binptids(nebs, bin_ptids, bin_offsets, b_edges):

    correct = 0
    for edge in b_edges:
        row0 = (edge[0] == bin_ptids).nonzero()[0].item()
        row1 = (edge[1] == bin_ptids).nonzero()[0].item()

        bin0 = np.flatnonzero(bin_offsets <= row0).max()
        bin1 = np.flatnonzero(bin_offsets <= row1).max()

        if bin0 == bin1 and (bin0==nebs[:,0]).sum()>0:
            correct += 1
        elif (bin0==nebs[:,0]).sum()>0 and (bin1==nebs[nebs[:,0]==(bin0)]).sum()>0:
            correct += 1
        elif (bin1==nebs[:,0]).sum()>0 and (bin0==nebs[nebs[:,0]==(bin1)]).sum()>0:
            correct += 1

    return correct

class frnn_test(unittest.TestCase):

    # Enforce that no garbage point ids are returned by frnn kernel
    def test_kernel_garbage_outputs(self):
        print('\n\nrunning test_kernel_garbage_outputs')
        device = t.cuda.current_device()

        n_loops =       5
        batch_size =    100
        radius =        15.0
        scale_radius =  10.0
        dim_size =      32
        use_pix_objects = False

        loader = get_test_loader(batch_size, datasize=None)

        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            texture, pts, img_ids = get_objects(data)

            edges = get_frnn(pts, radius, scale_radius, img_ids, device)

            self.assertTrue(edges.max().item() < pts.size(0), 'kernel returned an invalid-large edge id')
            self.assertTrue(edges.min().item() > -1, 'kernel returned an invalid negative edge id')

            loop += 1
            if loop == n_loops: break


    # Count how many edges output from frnn_cuda_kernel are valid edges
    def test_frnn_kernel_correct(self):
        print('\n\nrunning test_frnn_kernel_correct')
        t.cuda.set_device(0)

        n_loops =           1
        batch_size =        40

        loader = get_test_loader(batch_size)

        for lin_radius, p_drop in zip([1.45, 3, 8, 16], [0.1,0.2,0.4,0.6]):
            for scale_radius in [1.]:
                for scale_multiplier in [1, 2*math.e,  math.e**3]:

                    loop = 0
                    for batch_idx, (data, target) in enumerate(loader):

                        tex, pts, imgid = get_objects(data.cuda())
                        pts[:,4] = t.rand_like(pts[:,4]).mul_(scale_multiplier).add_(1)
                        pts[:,:2].add_( t.rand_like(pts[:,:2]).sub(0.5).mul(2) )

                        mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                        data = tex[mask], pts[mask], imgid[mask]
                        data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
                        tex,pts,imgid = data

                        ## shuffle objects
                        shuffled_order = t.randint(pts.size(0), [pts.size(0)])
                        pts, imgid = pts[shuffled_order], imgid[shuffled_order]

                        edges = get_frnn((None,pts,imgid), t.tensor(lin_radius).cuda(), t.tensor(scale_radius).cuda(), t.tensor(batch_size).cuda())

                        corr_lin_count, corr_scale_count = frnn_check_edges(edges.cpu().numpy(), imgid.cpu().numpy(), pts.cpu().numpy(), lin_radius, scale_radius)

                        print('lin correct: ', corr_lin_count, ' / ', edges.size(0))
                        print('scale correct: ', corr_scale_count, ' / ', edges.size(0))
                        self.assertEqual(corr_lin_count, edges.size(0), 'kernel returned invalid neighbors per 2D radius compare')
                        self.assertEqual(corr_scale_count, edges.size(0), 'kernel returned invalid neighbors per scale_radius compare')

                        loop += 1
                        if loop == n_loops: break


    def test_frnn_timings(self):
        print('\n\nrunning test_frnn_timings')
        warnings.filterwarnings("ignore")
        device = t.cuda.current_device()

        n_loops =       8
        batch_size =    t.tensor(80).cuda()
        radius =        t.tensor(2.45).cuda()
        scale_radius =  t.tensor(1.2).cuda()

        loader = get_test_loader(batch_size)

        times = t.empty([0], dtype=t.float32)
        edges_sizes = t.zeros(n_loops, dtype=t.int32)
        pts_sizes = t.zeros(n_loops, dtype=t.int32)
        test = 0
        for lin_radius, p_drop in zip([1.45, 3, 8, 16], [0.3,0.5,0.7,0.9]):
            for scale_radius in [1., 1.2, 1.4]:
                for scale_multiplier in [1, 2*math.e,  math.e**3]:

                    loop = 0
                    for batch_idx, (data, target) in enumerate(loader):

                        tex, pts, imgid = get_objects(data.cuda())
                        pts[:,4] = t.rand_like(pts[:,4]).mul_(scale_multiplier).add_(1)

                        mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                        data = tex[mask], pts[mask], imgid[mask]
                        data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
                        tex,pts,imgid = data

                        batch_size = int(imgid[-1].item()) +1

                        t.cuda.synchronize(device)
                        tic = time.time_ns()
                        edges = get_frnn((None,pts,imgid), t.tensor(lin_radius).cuda(), t.tensor(scale_radius).cuda(), t.tensor(batch_size).cuda())
                        t.cuda.synchronize(device)

                        times = t.cat([times, t.tensor([time.time_ns() - tic])])
                        pts_sizes[loop] = pts.size(0)
                        edges_sizes[loop] = edges.size(0)

                        loop += 1
                        if loop == n_loops: break

                    test += 1
                    print("test ",test)

        times = times[20:]
        print('times: ', times.mul_(1e-6), ' (msec)')
        print('times mean: ', times.mean().item(), '(msec)\n')
        print('objects searched: ', pts_sizes)
        print('edges found: ', edges_sizes)


    def test_conv(self):
        device = t.cuda.current_device()

        mod = torch.nn.Conv2d(32,32,3).cuda().eval()

        for i in range(8):

            data = t.randint(0,255,[40,32,32,32], dtype=t.float, device=device)

            t.cuda.synchronize(device)
            tic = time.time()
            out = mod(data)
            t.cuda.synchronize(device)
            print( (time.time() - tic) * 1e3, " ms")


    # Enforces that the number of unique edges returned by frnn_cuda is equal to the number of unique edges returned by
    #   frnn_brute; enforces that frnn kernel returns the same edges as frnn_brute. It is assumed that frnn_brute
    #   returns all unique valid edges in scaled_pts.
    def test_kernel_finds_all_edges(self):
        print('\n\nrunning test_kernel_finds_all_edges')
        t.cuda.set_device(0)

        n_loops =           1
        batch_size =        8

        loader = get_test_loader(batch_size)

        test = 0
        for lin_radius, p_drop in zip([1.45, 3, 8, 16], [0.2,0.3,0.4,0.6]):
                for scale_radius in [1., 1.2]:
                    for scale_multiplier in [1, 3, 7]:

        # for lin_radius, p_drop in zip([2, 8, 16], [0.1, 0.4, 0.6]):
        #     for scale_radius in [1.]:
        #         for scale_multiplier in [math.e ** 3]:

                        loop = 0
                        for batch_idx, (data, target) in enumerate(loader):

                            tex, pts, imgid = get_objects(data.cuda())

                            ## randomize object-sizes, locations
                            pts[:,4] = t.rand_like(pts[:,4]).mul_(scale_multiplier).add_(1)
                            pts[:, :2].add_(t.rand_like(pts[:, :2]).sub(0.5).mul(2))

                            ## drop random
                            mask = t.rand(pts.size(0), device=pts.device).ge(p_drop)
                            pts,imgid = pts[mask], imgid[mask]

                            ## shuffle objects
                            shuffled_order = t.randint(pts.size(0), [pts.size(0)])
                            pts, imgid = pts[shuffled_order], imgid[shuffled_order]

                            edges = get_frnn((None,pts,imgid), t.tensor(lin_radius).cuda(), t.tensor(scale_radius).cuda(), t.tensor(batch_size).cuda())
                            edges = edges.sort(dim=1)[0].sort(dim=0)[0]

                            b_edges = get_frnn((None,pts.cpu(),imgid.cpu()), lin_radius, scale_radius, batch_size)
                            b_edges = b_edges.sort(dim=1)[0].sort(dim=0)[0]

                            settings = [lin_radius, scale_radius, p_drop, scale_multiplier]
                            size_diff = abs(b_edges.size(0) - edges.size(0))
                            if size_diff in [1,2]:
                                print("WARNING: kernel edges and cpu-edges vary in size by ",size_diff,". May be floating-point error; ensure binning is correct")
                            elif size_diff > 2:
                                print("brute (correct) edges larger than kernel edges; failed test ", test)
                                print("brute size: ", b_edges.size(0), " kernel size: ", edges.size(0), " size diff: ", size_diff)
                                print("failure settings: ", settings)
                                exit(1)
                            else:
                                print("settings: ", settings)
                                self.assertTrue( edges.eq(b_edges.int().cuda()).sum() == edges.numel(),
                                                 'brute frnn and frnn kernel returned different (same-sized) sets of edges. ')

                            print('test: ', test)
                            loop += 1
                            if loop == n_loops: break
                        test += 1


    def test_merge_finds_all_edges(self):
        t.manual_seed(7)
        t.cuda.manual_seed_all(7)

        print('\n\nrunning test_merge_finds_all_edges')
        device = 1
        t.cuda.empty_cache()

        n_loops =           400
        batch_size =        40

        lin_radius = t.tensor(np.sqrt(2), dtype=t.float, device=device)
        scale_radius = t.tensor(1, dtype=t.float, device=device)

        loader = get_merge_loader(batch_size, datasize=None)

        for loop, (data, target) in enumerate(loader):
            for p_drop in [0.8,0.5,0.]:

                tex,pts,imgid = (te.to(device) for te in data)
                batch_size = imgid[-1] +1

                mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                data = tex[mask], pts[mask], imgid[mask]
                data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
                tex,pts,imgid = data

                assert pts[:,:2].max() < 32
                assert pts[:,:2].min() >= 0

                aout = frnn_cuda.frnn_kernel( pts, imgid, lin_radius, scale_radius, batch_size )

                edges = aout[0]
                edges = edges.sort(dim=1)[0].unique(sorted=True, dim=0)

                b_edges = get_frnn((tex.cpu(),pts.cpu(),imgid.cpu()), lin_radius, scale_radius, batch_size)
                b_edges = b_edges.sort(dim=1)[0].unique(sorted=True, dim=0)

                settings = [lin_radius, scale_radius]
                size_diff = abs(b_edges.size(0) - edges.size(0))
                if 1 <= size_diff < 3:
                    print("WARNING: kernel edges and cpu-edges vary in size by ",size_diff,". May be floating-point error; ensure binning is correct")
                elif size_diff > 2:
                    print("brute (correct) edges larger than kernel edges; failed test ", 0)
                    print("brute size: ", b_edges.size(0), " kernel size: ", edges.size(0), " size diff: ", size_diff)
                    print("failure settings: ", settings)
                    exit(1)
                else:
                    self.assertTrue( edges.eq(b_edges.int().to(device)).sum() == edges.numel(),
                                     'brute frnn and frnn kernel returned different (same-sized) sets of edges')

            if loop == n_loops: break
            print("loop: ", loop)


    def test_edges_same_image_only(self):
        print('\n\nrunning test_edges_same_image_only')
        device = t.cuda.current_device()

        n_loops =           5
        batch_size =        53
        radius =            2.7
        scale_radius =      1.0
        dim_size =          76
        use_pix_objects =   True

        loader = get_test_loader(batch_size, datasize=None)

        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            texture, pts, img_ids = get_objects(data)

            edges = get_frnn(pts, radius, scale_radius, img_ids, device)

            imgids_pts = t.cat( [img_ids.float().unsqueeze(1).cpu(), pts.cpu()], 1)

            corr_same_image = check_edges_images(edges.cpu().numpy(), imgids_pts.numpy())

            print('correct edges with points in same image: ', corr_same_image, ' / ', edges.size(0))
            self.assertEqual(corr_same_image, edges.size(0), 'kernel returned invalid neighbors: points from different images')

            print('loop: ', loop)
            loop += 1
            if loop == n_loops: break



    def test_kernel_no_duplicate_edges(self):
        print('\n\nrunning test_kernel_no_duplicate_edges')
        device = t.cuda.current_device()

        n_loops =           2
        batch_size =        4
        radius =            4.7
        scale_radius =      1.0
        dim_size =          17
        use_pix_objects =   True

        loader = get_test_loader(batch_size, datasize=None)

        sizes_before = t.zeros(n_loops, dtype=t.int32)
        sizes_after = t.zeros(n_loops, dtype=t.int32)

        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            texture, pts, img_ids = get_objects(data)


            edges = get_frnn(pts, radius, scale_radius, img_ids, device)
            sizes_before[loop] = edges.size(0)

            edges = edges.sort(1)[0].unique(sorted=True, dim=0)
            sizes_after[loop] = edges.size(0)

            print('loop: ', loop)
            loop += 1
            if loop == n_loops: break

        print("edge sizes before sort: ", sizes_before)
        print("edge sizes after sort: ", sizes_after)

        self.assertTrue( all(sizes_before.eq(sizes_after)), "FAIL: kernel found redundant edges.")


    # test number of binnable edges from neighbors kernel (builds 'neighbors' or 'nebs'), and bid2ptid.
    # only works with debugging kernel that outputs all data
    def test_bid2ptid(self):
        print('\n\nrunning test_bid2ptid')
        device = 0

        n_loops =           1
        batch_size =        2
        # radius =            1.2
        # scale_radius =      1.2
        dim_size =          24
        use_pix_objects =   True

        loader = get_test_loader(batch_size, datasize=None)

        for lin_radius in [1., 1.2, 1.4, 1.8, 3.4, 6.5]:
            for scale_radius in [1., 1.2, 4.5, 7.6]:
                for p_drop in [0.9,0.5,0.2,0.1, 0]:
                    loop = 0
                    for batch_idx, (data, target) in enumerate(loader):

                        tex, pts, imgid = get_objects(data)
                        pts[:,4] = t.rand_like(pts[:,4]).mul_(math.e **3).add_(1)

                        mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                        data = tex[mask], pts[mask], imgid[mask]
                        data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
                        tex,pts,imgid = data

                        aout = frnn_all(pts, imgid, lin_radius, scale_radius)
                        b_edges = get_frnn((tex.cpu(),pts.cpu(),imgid.cpu()), lin_radius, scale_radius)

                        nebs = aout[2].cpu()
                        bid2ptid = aout[3].cpu()

                        correct = verify_bid2ptid(nebs.cpu().numpy(), bid2ptid.cpu().numpy(), b_edges.cpu().numpy())
                        print("(edges reachable from bid2ptid) / (correct total number of edges): {} / {}".format(correct, b_edges.shape[0]) )
                        print("edges unreachable due to kernel-binning: {}".format(b_edges.shape[0]-correct))
                        self.assertEqual(b_edges.shape[0], correct)

                        loop+=1
                        if loop==n_loops: break



    def test_bin_ptids(self):
        print('\n\nrunning test_bin_ptids')
        device = 0

        n_loops =           4
        batch_size =        2
        dim_size =          16
        use_pix_objects =   True

        loader = get_test_loader(batch_size, datasize=None)

        for lin_radius in [1.01, 1.2, 1.4, 1.8, 3.4, 6.5]:
            for scale_radius in [1., 1.2, 4.5, 7.6]:
                for p_drop in [0.9,0.5,0.2,0.1, 0]:

                    loop = 0
                    for batch_idx, (data, target) in enumerate(loader):

                        tex, pts, imgid = get_objects(data)
                        pts[:,4] = t.rand_like(pts[:,4]).mul_(math.e **3).add_(1)

                        mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                        data = tex[mask], pts[mask], imgid[mask]
                        data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
                        tex,pts,imgid = data

                        aout = frnn_all(pts, imgid, lin_radius, scale_radius)
                        b_edges = get_frnn((tex.cpu(),pts.cpu(),imgid.cpu()), lin_radius, scale_radius)

                        nebs = aout[2].squeeze()
                        bin_ptids = aout[4].squeeze()
                        bin_offsets = aout[5].squeeze()

                        correct = verify_binptids(nebs.cpu().numpy(), bin_ptids.cpu().numpy(), bin_offsets.cpu().numpy(), b_edges.cpu().numpy())

                        print("(edges reachable from bin_ptids) / (correct total number of edges): {} / {}".format(correct, b_edges.shape[0]) )
                        print("edges unreachable due to kernel-binning: {}".format(b_edges.shape[0]-correct))
                        self.assertEqual(b_edges.shape[0], correct)

                        # exit(0)
                        print('loop: ',loop)
                        loop += 1
                        if loop == n_loops: break



    def test_bin_offsets(self):
        print('\n\nrunning test_bin_offsets')
        device = 0

        ## batch 4 causes illegal memory access in frnn_bipart_kern
        n_loops =           3
        batch_size =        2
        radius =            1.6
        scale_radius =      1.0
        dim_size =          16
        use_pix_objects =   True

        loader = get_test_loader(batch_size, datasize=None)

        loop = 0
        for batch_idx, (data, target) in enumerate(loader):

            texture, pts, img_ids = get_objects(data)

            pts[:,4] = t.rand_like(pts[:,4]).mul_(2).add_(1)

            aout = frnn_all(pts, img_ids, radius, scale_radius, device)

            imgids_pts = t.cat( [img_ids.float().unsqueeze(1).cpu(), pts.cpu()], dim=1)
            b_edges, n_edges = frnn_cpu(imgids_pts.numpy(), pts.size(0), radius, scale_radius)
            b_edges = t.tensor(b_edges)
            b_edges = b_edges[b_edges[:, 0] + b_edges[:, 1] >= 0]

            nebs = aout[3].cpu()
            bin_ptids = aout[5].cpu().squeeze()
            bin_offsets = aout[6].cpu().squeeze()
            bin_counts = aout[2].cpu()
            bid2ptid = aout[4].cpu()

            bcount = bid2ptid[:,0].bincount()
            assert bcount.eq(bin_counts.squeeze()).sum() == bcount.size(0)

            for i,b_loc in enumerate(bin_counts):
                bid2pt = bid2ptid[:,1][ bid2ptid[:,0] == i ]

                if i+1 < bin_offsets.size(0):
                    bin_pt = bin_ptids[ bin_offsets[i]:bin_offsets[i+1]]
                else: bin_pt = bin_ptids[ bin_offsets[i]: ]

                n_corr = bid2pt.sort()[0].eq( bin_pt.squeeze().sort()[0] ).sum()
                if not n_corr == b_loc:
                    print("failed, bin: ", i, "n_found / bin correct: ", n_corr, " / ", b_loc )

            loop+=1
            if loop==n_loops: break



    # Verify that edges found by frnn_cpu are valid edges.
    def test_frnn_cpu(self):
        print('\n\nrunning test_frnn_cpu')
        t.cuda.set_device(0)

        n_loops = 1
        batch_size = 10

        loader = get_test_loader(batch_size)

        for lin_radius, p_drop in zip([1.45, 3, 8, 16], [0.1, 0.2, 0.4, 0.6]):
            for scale_radius in [1.]:
                for scale_multiplier in [1, 3, 6]:

                    loop = 0
                    for batch_idx, (data, target) in enumerate(loader):

                        tex, pts, imgid = get_objects(data.cuda())
                        pts[:, 4] = t.rand_like(pts[:, 4]).mul_(scale_multiplier).add_(1)
                        pts[:, :2].add_(t.rand_like(pts[:, :2]).sub(0.5).mul(2))

                        mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                        data = tex[mask], pts[mask], imgid[mask]
                        data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
                        tex, pts, imgid = data

                        edges = get_frnn((None, pts.cpu(), imgid.cpu()), t.tensor(lin_radius).cuda(), t.tensor(scale_radius).cuda(), t.tensor(batch_size).cuda())

                        corr_lin_count, corr_scale_count = frnn_check_edges(edges.cpu().numpy(), imgid.cpu().numpy(), pts.cpu().numpy(), lin_radius, scale_radius)

                        print('lin correct: ', corr_lin_count, ' / ', edges.size(0))
                        print('scale correct: ', corr_scale_count, ' / ', edges.size(0))
                        self.assertEqual(corr_lin_count, edges.size(0),
                                         ' returned invalid neighbors per 2D radius compare')
                        self.assertEqual(corr_scale_count, edges.size(0),
                                         ' returned invalid neighbors per scale_radius compare')

                        loop += 1
                        if loop == n_loops: break




if __name__ == '__main__':
    unittest.main()



























