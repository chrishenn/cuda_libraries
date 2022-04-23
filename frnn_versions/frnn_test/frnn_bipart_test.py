from test.context import get_test_loader
import oomodel.oovis as oovis

from frnn_bipart.frnn_bipart_bind import get_frnn_bipart

import math
import unittest
import numpy as np
import time

from numba import njit, prange

import torch as t


def get_bipart_objects(data):
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

    group_ids = t.randint_like(geom[:,0], low=0, high=2, device=geom.device)
    geom[:,5] = group_ids

    return texture.to(device), geom.to(device), imgid.to(device)


@njit(parallel=True)
def frnn_bipart_check_edges(edges, pts, imgid, lin_radius, scale_radius):
    #[y, x, z, angle, scale, group_id]

    corr_lin_count = 0
    corr_scale_count = 0

    loops = len(edges[:,0])

    for i in prange(loops):
        edge = edges[i]
        p0 = pts[ edge[0] ]
        p1 = pts[ edge[1] ]
        imgid_0 = imgid[edge[0]]
        imgid_1 = imgid[edge[1]]

        if imgid_0 == imgid_1 and p0[-1] != p1[-1] and edge[0] != edge[1]:

            diff_yxz = np.fabs( p0[np.array([0,1,2])] - p1[np.array([0,1,2])] )
            dist_yxz = np.sqrt( np.sum( np.power(diff_yxz, 2) ) )

            size_0, size_1 = p0[4], p1[4]
            diff_s = np.abs( np.log(size_0) - np.log(size_1) )

            if dist_yxz < np.multiply(lin_radius, np.sqrt( np.multiply(size_0, size_1) ) ):
                corr_lin_count += 1
            if diff_s < scale_radius:
                corr_scale_count += 1

    return corr_lin_count, corr_scale_count

# count how many edges are drawn between points with the same img_id (in the same image)
@njit(parallel=True)
def check_edges_images(edges, imgid):

    loops = len(edges[:,0])
    corr_same_image = 0

    for i in prange(loops):
        ed = edges[i]
        imgid_0 = imgid[ ed[0] ]
        imgid_1 = imgid[ ed[1] ]

        if imgid_0 == imgid_1: corr_same_image += 1

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



class frnn_bipart_test(unittest.TestCase):

    # Enforce that no garbage point ids are returned by frnn kernel
    def test_kernel_garbage_outputs(self):
        print('\n\nrunning test_kernel_garbage_outputs')
        t.cuda.set_device(0)

        n_loops =       5
        batch_size =    t.tensor(10).cuda()
        radius =        t.tensor(1.45).cuda()
        scale_radius =  t.tensor(1.7).cuda()

        loader = get_test_loader(batch_size)

        for loop, (data, target) in enumerate(loader):

            texture, pts, imgid = get_bipart_objects(data.cuda())
            edges = get_frnn_bipart((None,pts,imgid), radius, scale_radius, batch_size)

            self.assertTrue(edges.max().item() < pts.size(0), 'kernel returned an invalid-large edge id')
            self.assertTrue(edges.min().item() > -1, 'kernel returned an invalid negative edge id')

            if loop+1 == n_loops: break


    # Count how many edges output from frnn_cuda_kernel are valid edges
    def test_frnn_kernel_correct(self):
        print('\n\nrunning test_frnn_kernel_correct')
        t.cuda.set_device(0)

        n_loops =       10
        batch_size =    t.tensor(10).cuda()
        lin_radius =    t.tensor(1.8).cuda()
        scale_radius =  t.tensor(1.5).cuda()

        loader = get_test_loader(batch_size)

        times = t.zeros(n_loops, dtype=t.float32)
        i_outs = t.zeros(n_loops, dtype=t.int32)
        for loop, (data, target) in enumerate(loader):

            texture, pts, imgid = get_bipart_objects(data.cuda())

            pts[:,4] = t.rand_like(pts[:,4]).mul_(2).add_(1)

            t.cuda.synchronize()
            tic = time.time()
            edges = get_frnn_bipart((None, pts, imgid), lin_radius, scale_radius, batch_size)
            t.cuda.synchronize()
            times[loop] = float(time.time() - tic)

            edges = edges.sort(1)[0].unique(dim=0)
            i_outs[loop] = edges.size(0)
            print(time.time() - tic)

            corr_lin_count, corr_scale_count = frnn_bipart_check_edges(edges.cpu().numpy(), pts.cpu().numpy(),
                                                                       imgid.cpu().numpy(), lin_radius.item(),
                                                                       scale_radius.item())

            print('lin correct: ', corr_lin_count, ' \ ', edges.size(0))
            print('scale correct: ', corr_scale_count, ' \ ', edges.size(0))
            self.assertEqual(corr_lin_count, edges.size(0), 'kernel returned invalid neighbors per 2D radius compare')
            self.assertEqual(corr_scale_count, edges.size(0), 'kernel returned invalid neighbors per scale_radius compare')

            print('loop: ',loop)
            if loop+1 == n_loops: break

        print('times: ', times)
        print('times mean: ', times.mean(), '(msec)\n')
        print('count of edges found: ', i_outs)
        print('counts mean: ', i_outs.float().mean(), '\n')


    def test_frnn_bipart_timings(self):
        print('\n\nrunning test_frnn_timings')
        t.cuda.set_device(0)

        n_loops =       500
        batch_size =    t.tensor(80).cuda()
        radius =        t.tensor(2.8).cuda()
        scale_radius =  t.tensor(1.0).cuda()

        loader = get_test_loader(batch_size)

        times = t.zeros(n_loops, dtype=t.float32)
        edges_sizes = t.zeros(n_loops, dtype=t.int32)
        pts_sizes = t.zeros(n_loops, dtype=t.int32)

        for loop, (data, target) in enumerate(loader):
            texture, pts, imgid = get_bipart_objects(data.cuda())

            t.cuda.synchronize()
            tic = time.time()
            edges = get_frnn_bipart((None,pts,imgid), radius, scale_radius, batch_size)
            t.cuda.synchronize()
            times[loop] = time.time() - tic
            pts_sizes[loop] = texture.size(0)
            edges_sizes[loop] = edges.size(0)

            if loop+1 == n_loops: break

        print('times: ', times.mul_(1000), ' (msec)')
        print('times mean: ', times.mean().item(), '(msec)\n')
        print('objects searched: ', pts_sizes)
        print('edges found: ', edges_sizes)


    # Enforces that the number of unique edges returned by frnn_bipartite_cuda is equal to the number of unique edges returned by
    #   the cpu impl; enforces that frnn kernel returns the same edges as the cpu impl. It is assumed that the cpu impl is correct.
    def test_kernel_finds_all_correct_edges(self):
        print('\n\nrunning test_kernel_finds_all_edges')
        t.cuda.set_device(0)

        n_loops =    2
        batch_size = 10

        loader = get_test_loader(batch_size)

        test = 0
        for lin_radius, p_drop in zip([1.45, 3, 8, 16, 2, 4, 8, 18 ], [0.0,0.5,0.7,0.9, 0.0, 0.4, 0.6, 0.7]):
                for scale_radius in [1., 1.2]:
                    for scale_multiplier in [1, 2*math.e,  math.e**3]:

                        for loop, (data, target) in enumerate(loader):

                            tex, pts, imgid = get_bipart_objects(data.cuda())
                            pts[:,4] = t.rand_like(pts[:,4]).mul_(scale_multiplier).add_(1)

                            mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
                            data = tex[mask], pts[mask], imgid[mask]
                            tex,pts,imgid = data

                            edges = get_frnn_bipart((None,pts,imgid), t.tensor(lin_radius,device=pts.device), t.tensor(scale_radius,device=pts.device), imgid[-1]+1)
                            edges = edges.sort(dim=1)[0].unique(sorted=True, dim=0)

                            b_edges = get_frnn_bipart((tex.cpu(),pts.cpu(),imgid.cpu()), lin_radius, scale_radius, imgid[-1]+1)
                            b_edges = b_edges.sort(dim=1)[0].unique(sorted=True, dim=0)

                            settings = [lin_radius, scale_radius, p_drop, scale_multiplier]
                            size_diff = abs(b_edges.size(0) - edges.size(0))
                            if 1 <= size_diff <= 2:
                                print("WARNING: kernel edges and cpu-edges vary in size by ",size_diff,". May be floating-point error; ensure binning is correct")
                            elif size_diff > 2:
                                print("brute (correct) edges different size than kernel edges; failed test ", test)
                                print("brute size: ", b_edges.size(0), " kernel size: ", edges.size(0), " size diff: ", size_diff)
                                print("failure settings: ", settings)
                                exit(1)
                            else:
                                self.assertTrue( edges.eq(b_edges.int().cuda()).sum() == edges.numel(),
                                                 'brute frnn and frnn kernel returned different (same-sized) sets of edges')

                            print('test: ', test)
                            if loop+1 == n_loops: break
                        test += 1


    def test_edges_same_image_only(self):
        print('\n\nrunning test_edges_same_image_only')
        t.cuda.set_device(0)

        n_loops =       10
        batch_size =    t.tensor(120).cuda()
        lin_radius =    t.tensor(3.45).cuda()
        scale_radius =  t.tensor(1.0).cuda()

        loader = get_test_loader(batch_size)

        for loop, (data, target) in enumerate(loader):

            texture, pts, imgid = get_bipart_objects(data.cuda())

            edges = get_frnn_bipart((None,pts,imgid), lin_radius, scale_radius, batch_size)

            corr_same_image = check_edges_images(edges.cpu().numpy(), imgid.cpu().numpy())

            print('correct edges with points in same image: ', corr_same_image, ' / ', edges.size(0))
            self.assertEqual(corr_same_image, edges.size(0), 'kernel returned invalid neighbors: points from different images')

            print('loop: ', loop)
            loop += 1
            if loop == n_loops: break


    def test_kernel_no_duplicate_edges(self):
        print('\n\nrunning test_kernel_no_duplicate_edges')
        t.cuda.set_device(0)

        n_loops =       2
        batch_size =    t.tensor(20, dtype=t.long).cuda()
        lin_radius =    t.tensor(2.45).cuda()
        scale_radius =  t.tensor(1.2).cuda()

        loader = get_test_loader(batch_size)

        sizes_before = t.zeros(n_loops, dtype=t.int32)
        sizes_after = t.zeros(n_loops, dtype=t.int32)

        for loop, (data, target) in enumerate(loader):

            texture, pts, imgid = get_bipart_objects(data.cuda())

            edges = get_frnn_bipart((None,pts,imgid), lin_radius, scale_radius, batch_size)
            sizes_before[loop] = edges.size(0)

            edges = edges.sort(1)[0].unique(sorted=True, dim=0)
            sizes_after[loop] = edges.size(0)

            print('loop: ', loop)
            if loop+1 == n_loops: break

        print("edge sizes before sort: ", sizes_before)
        print("edge sizes after sort: ", sizes_after)

        self.assertTrue( all(sizes_before.eq(sizes_after)), "FAIL: kernel found redundant edges.")


    # Verify that edges found by frnn_cpu are valid edges.
    def test_frnn_bipart_cpu(self):
        print('\n\nrunning test_frnn_cpu')
        t.cuda.set_device(0)

        n_loops =           4
        batch_size =        20
        lin_radius =        t.tensor(4.6).cuda()
        scale_radius =      t.tensor(1.7).cuda()

        loader = get_test_loader(batch_size)

        times = t.zeros(n_loops, dtype=t.float)
        for loop, (data, target) in enumerate(loader):

            texture, pts, imgid = get_bipart_objects(data)

            tic = time.time()
            b_edges = get_frnn_bipart((texture,pts,imgid), lin_radius, scale_radius, imgid[-1]+1)
            times[loop] = (time.time()-tic)*1e-6

            assert(b_edges.size(0) == b_edges.sort(1)[0].unique(dim=0).size(0))

            corr_lin_count, corr_scale_count = frnn_bipart_check_edges(b_edges.cpu().numpy(), pts.numpy(),
                                                                       imgid.numpy(), lin_radius.item(),
                                                                       scale_radius.item())

            print('lin correct: ', corr_lin_count, ' / ', b_edges.size(0))
            print('scale correct: ', corr_scale_count, ' / ', b_edges.size(0))
            self.assertEqual(corr_lin_count, b_edges.size(0), 'brute returned invalid neighbors per 2D radius compare')
            self.assertEqual(corr_scale_count, b_edges.size(0), 'brute returned invalid neighbors per scale_radius compare')

            print('loop: ',loop)
            if loop+1 == n_loops: break

        print("time average (msec): ", times.mean())



    ########################################################################################################
    ## Not updated with new interface yet
    ########################################################################################################
    # test number of binnable edges from neighbors kernel (builds 'neighbors' or 'nebs'), and bid2ptid.
    # only works with debugging kernel that outputs all data
    # def test_bid2ptid(self):
    #     print('\n\nrunning test_bid2ptid')
    #     t.cuda.set_device(0)
    #
    #     n_loops =       1
    #     batch_size =    t.tensor(2).cuda()
    #
    #     loader = get_test_loader(batch_size, datasize=None)
    #
    #     for lin_radius in [1., 1.2, 1.4, 1.8, 3.4, 6.5]:
    #         for scale_radius in [1., 1.2, 4.5, 7.6]:
    #             for p_drop in [0.9,0.5,0.2,0.1, 0]:
    #                 loop = 0
    #                 for batch_idx, (data, target) in enumerate(loader):
    #
    #                     tex, pts, imgid = get_objects(data)
    #                     pts[:,4] = t.rand_like(pts[:,4]).mul_(math.e **3).add_(1)
    #
    #                     mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
    #                     data = tex[mask], pts[mask], imgid[mask]
    #                     data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
    #                     tex,pts,imgid = data
    #
    #                     aout = frnn_bipart_cuda.frnn_bipart_kernel(pts, imgid, t.tensor(lin_radius).cuda(), t.tensor(scale_radius).cuda(), batch_size)
    #                     b_edges = get_frnn_bipart((tex.cpu(),pts.cpu(),imgid.cpu()), lin_radius, scale_radius, batch_size)
    #
    #                     nebs = aout[2].cpu()
    #                     bid2ptid = aout[3].cpu()
    #
    #                     correct = verify_bid2ptid(nebs.cpu().numpy(), bid2ptid.cpu().numpy(), b_edges.cpu().numpy())
    #                     print("(edges reachable from bid2ptid) / (correct total number of edges): {} / {}".format(correct, b_edges.shape[0]) )
    #                     print("edges unreachable due to kernel-binning: {}".format(b_edges.shape[0]-correct))
    #                     self.assertEqual(b_edges.shape[0], correct)
    #
    #                     loop+=1
    #                     if loop==n_loops: break
    #
    # def test_bin_ptids(self):
    #     print('\n\nrunning test_bin_ptids')
    #     device = 0
    #
    #     n_loops =           4
    #     batch_size =        2
    #     dim_size =          16
    #     use_pix_objects =   True
    #
    #     loader = get_test_loader(batch_size, datasize=None)
    #
    #     for lin_radius in [1.01, 1.2, 1.4, 1.8, 3.4, 6.5]:
    #         for scale_radius in [1., 1.2, 4.5, 7.6]:
    #             for p_drop in [0.9,0.5,0.2,0.1, 0]:
    #
    #                 loop = 0
    #                 for batch_idx, (data, target) in enumerate(loader):
    #
    #                     tex, pts, imgid = get_objects(data)
    #                     pts[:,4] = t.rand_like(pts[:,4]).mul_(math.e **3).add_(1)
    #
    #                     mask = t.rand(tex.size(0), device=tex.device).ge(p_drop)
    #                     data = tex[mask], pts[mask], imgid[mask]
    #                     data[1][:, 5] = t.arange(data[1].size(0), dtype=t.float, device=data[1].device)
    #                     tex,pts,imgid = data
    #
    #                     aout = frnn_all(pts, imgid, lin_radius, scale_radius)
    #                     b_edges = get_frnn_bipart((tex.cpu(),pts.cpu(),imgid.cpu()), lin_radius, scale_radius)
    #
    #                     nebs = aout[2].squeeze()
    #                     bin_ptids = aout[4].squeeze()
    #                     bin_offsets = aout[5].squeeze()
    #
    #                     correct = verify_binptids(nebs.cpu().numpy(), bin_ptids.cpu().numpy(), bin_offsets.cpu().numpy(), b_edges.cpu().numpy())
    #
    #                     print("(edges reachable from bin_ptids) / (correct total number of edges): {} / {}".format(correct, b_edges.shape[0]) )
    #                     print("edges unreachable due to kernel-binning: {}".format(b_edges.shape[0]-correct))
    #                     self.assertEqual(b_edges.shape[0], correct)
    #
    #                     # exit(0)
    #                     print('loop: ',loop)
    #                     loop += 1
    #                     if loop == n_loops: break
    #
    # def test_bin_offsets(self):
    #     print('\n\nrunning test_bin_offsets')
    #     device = 0
    #
    #     ## batch 4 causes illegal memory access in frnn_bipart_kern
    #     n_loops =           3
    #     batch_size =        2
    #     radius =            1.6
    #     scale_radius =      1.0
    #     dim_size =          16
    #     use_pix_objects =   True
    #
    #     loader = get_test_loader(batch_size, datasize=None)
    #
    #     loop = 0
    #     for batch_idx, (data, target) in enumerate(loader):
    #
    #         texture, pts, imgid = get_objects(data)
    #
    #         pts[:,4] = t.rand_like(pts[:,4]).mul_(2).add_(1)
    #
    #         aout = frnn_all(pts, imgid, radius, scale_radius, device)
    #
    #         imgids_pts = t.cat( [imgid.float().unsqueeze(1).cpu(), pts.cpu()], dim=1)
    #         b_edges, n_edges = frnn_cpu(imgids_pts.numpy(), pts.size(0), radius, scale_radius)
    #         b_edges = t.tensor(b_edges)
    #         b_edges = b_edges[b_edges[:, 0] + b_edges[:, 1] >= 0]
    #
    #         nebs = aout[3].cpu()
    #         bin_ptids = aout[5].cpu().squeeze()
    #         bin_offsets = aout[6].cpu().squeeze()
    #         bin_counts = aout[2].cpu()
    #         bid2ptid = aout[4].cpu()
    #
    #         bcount = bid2ptid[:,0].bincount()
    #         assert bcount.eq(bin_counts.squeeze()).sum() == bcount.size(0)
    #
    #         for i,b_loc in enumerate(bin_counts):
    #             bid2pt = bid2ptid[:,1][ bid2ptid[:,0] == i ]
    #
    #             if i+1 < bin_offsets.size(0):
    #                 bin_pt = bin_ptids[ bin_offsets[i]:bin_offsets[i+1]]
    #             else: bin_pt = bin_ptids[ bin_offsets[i]: ]
    #
    #             n_corr = bid2pt.sort()[0].eq( bin_pt.squeeze().sort()[0] ).sum()
    #             if not n_corr == b_loc:
    #                 print("failed, bin: ", i, "n_found / bin correct: ", n_corr, " / ", b_loc )
    #
    #         loop+=1
    #         if loop==n_loops: break




if __name__ == '__main__':
    unittest.main()



























