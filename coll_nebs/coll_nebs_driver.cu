/**
Author: Chris Henn (https://github.com/chrishenn)
**/

#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_bin_kern.h"
#include "frnn_kern.h"
#include "scan.h"


// define for error checking
// #define CUDA_ERROR_CHECK

// function prototypes
double get_nanos();

void find_edges(
        torch::Tensor lin_radius_at_each_scabin, torch::Tensor scabin_offsets,

        torch::Tensor bin_ptids, torch::Tensor bin_wcounts, torch::Tensor neighbors, torch::Tensor neb_counts, torch::Tensor bin_counts, torch::Tensor bin_offsets,
        torch::Tensor bid2ptid, torch::Tensor write_count, torch::Tensor glob_count, torch::Tensor edges,

        frnn_bipart_kern* FRNN, scan* SCAN, frnn_bin_kern* NEB,

        torch::Tensor pts, torch::Tensor imgid,
        torch::Tensor xmin,

        float lin_radius, float scale_radius, float scalemin,

        int num_binstotal,
        int num_bins_perimg,
        int num_imgs,
        int num_scabin,

        int neighb_size1,

        int pt_size0,
        int pt_size1
);

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    do{
        cudaError err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }

        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while(0);
#endif

    return;
}

// timing helper
double get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_nsec;
}

/**
cpu entry point for frnn python extension.
**/
std::vector<torch::Tensor> coll_nebs_cuda_call(
    torch::Tensor tex,
    torch::Tensor pts,
    torch::Tensor imgid,

    float lin_radius,
    float scale_radius,

    int threshold,
    int coll_iters,
    float dampen_fact,

    int batch_size,
    int img_size,

    int device_id
)
{
    // New code; setup


    auto pt_size0 = pts.size(0);
    auto pt_size1 = pts.size(1);

    auto num_imgs = batch_size;
    auto n_objs = imgid.size(0);
    const int neighb_size1 = 20;

    int intermed = int(lin_radius * 2 + 1);
    const int edges_size0 = (intermed * intermed) * 2 * n_objs;
    const int edges_size1 = 2;

    // allocate device data structures
    torch::Tensor bin_ptids = torch::empty({pt_size0, 1}, options);
    torch::Tensor bid2ptid = torch::empty({pt_size0, 2}, options);
    torch::Tensor write_count = torch::zeros({1}, options);

    torch::Tensor glob_count = torch::zeros({1}, options);
    torch::Tensor edges = torch::empty({edges_size0, edges_size1}, options);

    // Allocate host data
    frnn_bipart_kern* FRNN = new frnn_bipart_kern();
    scan* SCAN = new scan();
    frnn_bin_kern* NEB = new frnn_bin_kern();

    // Make a copy of pts
    torch::Tensor pts_coll = pts.clone();
    torch::Tensor locs;
    torch::Tensor locs_lf;
    torch::Tensor locs_rt;

    torch::Tensor counts;
    torch::Tensor counts_lf;
    torch::Tensor counts_rt;

    torch::Tensor vec_lfrt;
    torch::Tensor vec_rtlf;

    torch::Tensor weight;

    torch::Tensor edges_tmp;
    torch::Tensor edges_0;
    torch::Tensor edges_1;

    for (int i = 0; i < coll_iters; i++)
    {
        auto xs = pts_coll.index({Slice(), 1});
        auto xmin = xs.min();
        auto xmax = xs.max();

        auto scales = pts_coll.index({Slice(), 4}).log();

        auto scalemin = scales.min().item<float>();
        torch::Tensor scalemax = scales.max() + 0.0000001;

        int num_scabin = scalemax.sub(scalemin).div(scale_radius).add(1).item<int32_t>();

        torch::Tensor scabin_cutoffs = torch::linspace(scalemin + scale_radius, scalemax.item<float>(), num_scabin, f_options);
        torch::Tensor lin_radius_at_each_scabin = torch::exp(scabin_cutoffs).mul_(lin_radius);
        torch::Tensor num_xbins_each_scale = (xmax - xmin).div(lin_radius_at_each_scabin).to(torch::kInt32).add(1);

        torch::Tensor scabin_offsets = torch::cat({torch::zeros({1}, options), num_xbins_each_scale});
        scabin_offsets = scabin_offsets.cumsum(0).to(torch::kInt32);

        auto num_bins_perimg = num_xbins_each_scale.index({-1}).item<int32_t>();
        num_scabin = scabin_offsets.size(0) - 1;

        const int num_binstotal = num_bins_perimg * num_imgs;

        torch::Tensor bin_wcounts = torch::zeros({num_binstotal, 1}, options);

        torch::Tensor neighbors = torch::empty({num_binstotal, neighb_size1}, options);
        torch::Tensor neb_counts = torch::empty({num_binstotal, 1}, options);

        torch::Tensor bin_counts = torch::empty({num_binstotal, 1}, options);
        torch::Tensor bin_offsets = torch::empty({num_binstotal, 1}, options);


        find_edges(
            lin_radius_at_each_scabin, scabin_offsets,

            bin_ptids, bin_wcounts,  neighbors,  neb_counts,  bin_counts,  bin_offsets,
            bid2ptid,  write_count,  glob_count,  edges,

            FRNN, SCAN, NEB,

            pts_coll, imgid, xmin,

            lin_radius, scale_radius, scalemin,

            num_binstotal,
            num_bins_perimg,
            num_imgs,
            num_scabin,

            neighb_size1,

            pt_size0,
            pt_size1
        );

        locs = pts_coll.index({Slice(), Slice(None, 2)});

        edges_tmp = edges.index( {Slice(None, glob_count.item<int32_t>()/2), Slice()} );

        counts = edges_tmp.flatten().bincount().add(1).unsqueeze(1);
        edges_0 = edges_tmp.index({Slice(), 0}).to(torch::kInt64);
        edges_1 = edges_tmp.index({Slice(), 1}).to(torch::kInt64);

        counts_lf = counts.index_select(0, edges_0);
        counts_rt = counts.index_select(0, edges_1);

        locs_lf = locs.index_select(0, edges_0);
        locs_rt = locs.index_select(0, edges_1);

        vec_lfrt = locs_lf.sub(locs_rt);
        weight = vec_lfrt.norm(2, {1}).add(1).reciprocal().unsqueeze(1);

        vec_lfrt = vec_lfrt * weight;

        float damp = ((float(i)+1) * dampen_fact);
        locs.index_add_(0, edges_1, vec_lfrt.div(counts_lf.mul(damp)));
        locs.index_add_(0, edges_0, -vec_lfrt.div(counts_rt.mul(damp)) );
    }

    // Make reps
    locs = pts_coll.index({Slice(), Slice(None,2)});
    locs = locs.round().to(torch::kInt64);

    torch::Tensor locs_y = locs.index({Slice(), 0, None}).clamp(0, img_size - 1);
    torch::Tensor locs_x = locs.index({Slice(), 1, None}).clamp(0, img_size - 1);

    torch::Tensor ang_batch = torch::zeros({batch_size, 2, img_size, img_size}, f_options);
    torch::Tensor channels = torch::arange(2, l_options).index({None});

    torch::Tensor angles = pts.index({Slice(), 3});
    ang_batch.index_put_({imgid.unsqueeze(1), channels, locs_y, locs_x}, torch::stack({angles.sin(), angles.cos()}, 1), true);

    torch::Tensor tex_batch = torch::zeros({batch_size, tex.size(1), img_size, img_size}, f_options);
    channels = torch::arange(tex.size(1), l_options).index({None});
    tex_batch.index_put_({imgid.unsqueeze(1), channels, locs_y, locs_x}, tex, true);

    counts = torch::ones({batch_size, 1, img_size, img_size}, f_options);
    counts.index_put_({imgid.unsqueeze(1), torch::tensor(0), locs_y, locs_x}, torch::ones_like(locs_x).to(torch::kFloat32), true);

    ang_batch = torch::atan2( ang_batch.index({Slice(), 0,"..."}), ang_batch.index({Slice(), 1,"..."}) );

    torch::Tensor nonzero_mask = counts.gt(threshold).squeeze();
    torch::Tensor nonzero_ids = nonzero_mask.nonzero();

    torch::Tensor tex_act = tex_batch.permute({0, 2, 3, 1}).index({nonzero_mask});
    torch::Tensor ang_act = ang_batch.index({nonzero_mask}).unsqueeze(1);

    torch::Tensor imgid_act = nonzero_ids.index({Slice(), 0});
    torch::Tensor locs_act_y = nonzero_ids.index({Slice(), 1, None}).to(torch::kFloat32);
    torch::Tensor locs_act_x = nonzero_ids.index({Slice(), 2, None}).to(torch::kFloat32);

    torch::Tensor pts_act = torch::cat( {locs_act_y, locs_act_x, torch::zeros_like(locs_act_x), ang_act, torch::ones_like(locs_act_x), torch::arange(ang_act.size(0), f_options).unsqueeze(1) },  1);

    free(FRNN);
    free(SCAN);
    free(NEB);

    return {tex_act, pts_act, imgid_act};
}


void find_edges(
        torch::Tensor lin_radius_at_each_scabin, torch::Tensor scabin_offsets,

        torch::Tensor bin_ptids, torch::Tensor bin_wcounts, torch::Tensor neighbors, torch::Tensor neb_counts, torch::Tensor bin_counts, torch::Tensor bin_offsets,
        torch::Tensor bid2ptid, torch::Tensor write_count, torch::Tensor glob_count, torch::Tensor edges,

        frnn_bipart_kern* FRNN, scan* SCAN, frnn_bin_kern* NEB,

        torch::Tensor pts, torch::Tensor imgid,
        torch::Tensor xmin,

        float lin_radius, float scale_radius, float scalemin,

        int num_binstotal,
        int num_bins_perimg,
        int num_imgs,
        int num_scabin,

        int neighb_size1,

        int pt_size0,
        int pt_size1
)
{
    glob_count.index_put_({0}, 0);
    write_count.index_put_({0}, 0);

    const dim3 blocks(256);
    const dim3 threads(256);
    const dim3 threads_frnnkern(16, 16);

    // Calculate neighbor bin_ids (neighbors); count writes to neighbor rows in neb_counts.
    // Central bins are written to column 0 of neighbors.
    // Locations in neighbors without a bin_id are set to -1.
    NEB->frnn_neighbors_launch(
        lin_radius_at_each_scabin.data<float>(),
        scabin_offsets.data<int>(),
        neighbors.data<int>(),
        neb_counts.data<int>(),

        num_binstotal,
        num_bins_perimg,
        num_imgs,
        num_scabin,
        neighb_size1,

        blocks,
        threads
    );
    CudaCheckError();

    // prune redundant bin pairs by row in neighbors; shuffle valid neighbors to lower columns; adjust neb_counts
	NEB->frnn_neighbors_prune_launch(
		neighbors.data<int>(),
		neb_counts.data<int>(),

		num_binstotal,
		neighb_size1,
		blocks,
		threads
	);
    CudaCheckError();

    // count number of pts in each bin, write to bin_counts; build bid2ptid (matching a bin_id to a pt_id by row)
    NEB->frnn_bin_counts_launch(
        pts.data<float>(),
        imgid.data<long>(),
        lin_radius_at_each_scabin.data<float>(),
        scabin_offsets.data<int>(),
        bin_counts.data<int>(),
        bid2ptid.data<int>(),
        write_count.data<int>(),

        scale_radius,
        xmin.item<float>(),
        scalemin,
        pt_size0,
        pt_size1,
        num_bins_perimg,
        num_binstotal,

        blocks,
        threads
    );
    CudaCheckError();

    // exclusive-scan bin_counts; write to bin_offsets
    SCAN->scan_launch(  bin_counts.data<int>(),
                        bin_offsets.data<int>(),
                        num_binstotal);

    CudaCheckError();

    // build bin_ptids from bid2ptid, bin_offsets. bin_ptids give contiguous indexes into pts
    // for a given bin, where a bin's pt indexes start at bin_offsets[bin_id].
    NEB->frnn_ptids_launch(
        bin_counts.data<int>(),
        bin_offsets.data<int>(),
        bid2ptid.data<int>(),
        bin_ptids.data<int>(),
        bin_wcounts.data<int>(),

        pt_size0,
        blocks,
        threads
    );
    CudaCheckError();

    // frnn_bipart_kern a block for at least each valid bin_id in neighbors
	const int nebs_width = neb_counts.max().item<int32_t>();
    const dim3 blocks_frnnkern(nebs_width, num_binstotal);

    // each block stores 5 float32's per point, up to max_binsize points per bin, and 2 bins in shared memory
    const int bin_stride = 5;
    const int max_binsize = bin_counts.max().item<int32_t>();
    const size_t shared = (2 * bin_stride * max_binsize) * sizeof(float);

    if (shared > 49152){
        fprintf (stderr, "ERROR coll_nebs_driver.cu: FRNN_MAIN_KERNEL ATTEMPTED TO ALLOCATE TOO MUCH SHARED MEMORY FOR YOUR DEVICE ARCH; DECREASE BATCH SIZE");
        exit(EXIT_FAILURE);
    }

    // compare bins of points on frnn criteria, write indexes of pts with an edge between them into 'edges'.
    // Count number of integer writes in glob_counts (glob_counts= 2xnumber_edges_written).
    FRNN->frnn_kern_launch(
        neighbors.data<int>(),
        bin_counts.data<int>(),
        bin_offsets.data<int>(),
        bin_ptids.data<int>(),
        pts.data<float>(),

        glob_count.data<int>(),
        edges.data<int>(),

        lin_radius,
        scale_radius,

        max_binsize,
        bin_stride,

        neighb_size1,
        pt_size1,

        blocks_frnnkern,
        threads_frnnkern,
        shared
    );
    CudaCheckError();

}
