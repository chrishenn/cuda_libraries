#include <torch/types.h>


// each row of neighbors should contain the 'central' bin at column0, and other bin's bin_ids
// for comparison to the central bin along the same row.
__global__ void frnn_neighbors(
        int*     neighbors,
        int*     neb_counts,

        float*   scabin_linrads,

        int*     n_xbin_eachscale,
        int*     n_xbin_eachscale_excume,

        int*     glob_to_scabin,
        int*     glob_to_im_i,

        const int n_binstotal,
        const int batch_size,

        const int neighb_size1
)
{
    const int thid_glob =       blockIdx.x * blockDim.x + threadIdx.x;
    const int tot_threads =     gridDim.x * blockDim.x;

    for (int i = thid_glob; i < n_binstotal; i += tot_threads)
    {
        int scabin = glob_to_scabin[i];
        int bins_perim = n_binstotal / batch_size;
        int my_im_glob_i = floorf(i / bins_perim) * bins_perim;

        int my_i_in_my_im = glob_to_im_i[i];
        int my_xbin_start_in_im = n_xbin_eachscale_excume[scabin];
        int my_i_in_my_xbin = my_i_in_my_im - my_xbin_start_in_im;

        int n_xbin_in_my_xbin = n_xbin_eachscale[scabin];

        // I am the 'center' bin at 'i'; my bin goes in neighbors[i][0]
        int cols_written = 0;
        neighbors[i * neighb_size1 + cols_written] = i;
        cols_written++;
        // bin x below at my scale
        if (my_i_in_my_xbin > 0) {
            neighbors[i*neighb_size1 + cols_written] = i - 1;
            cols_written++;
        }
        // bin x above at my scale
        if (my_i_in_my_xbin +1 < n_xbin_in_my_xbin){
            neighbors[i*neighb_size1 + cols_written] = i + 1;
            cols_written++;
        }

        // find nebs in the neighboring smaller scale
        float my_linrad = scabin_linrads[scabin];
        float my_x_lf =  my_i_in_my_xbin * (2 * my_linrad);
        float my_x_rt = my_x_lf + (2 * my_linrad);

        int scabin_neb = scabin - 1;
        if (scabin_neb >= 0)
        {
            float neb_linrad = scabin_linrads[scabin_neb];
            int leftmost_neb = floorf(my_x_lf / (2*neb_linrad) - my_linrad/2);
            int rightmost_neb = floorf(my_x_rt / (2*neb_linrad) + my_linrad/2);

            int n_xbin_neb = n_xbin_eachscale[scabin_neb];
            if (rightmost_neb >= n_xbin_neb) rightmost_neb = n_xbin_neb - 1;
            if (leftmost_neb < 0) leftmost_neb = 0;

            int neb_xbin_start_in_im = n_xbin_eachscale_excume[scabin_neb];

            for (int k = leftmost_neb; k <= rightmost_neb; k++) {
                neighbors[i * neighb_size1 + cols_written] = my_im_glob_i + neb_xbin_start_in_im + k;
                cols_written++;
            }
        }

        neb_counts[i] = cols_written;

        // fill remaining in this thread's row with -1
        for (int end = cols_written; end < neighb_size1; end++){
            neighbors[ i * neighb_size1 + end ] = -1;
        }
    }
}

// prune redundant bin_id pairs from rows of neighbors. Update neb_counts for correct block launches in frnn_kern.
__global__ void frnn_neighbors_prune(
    int*  neighbors,
    int*  neb_counts,

    const int num_binstotal,
    const int neighb_size1
){
    for (int phase = 0; phase < 2; phase++) {

        for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_binstotal; row += blockDim.x * gridDim.x) {
            for (int col = 1; col < neighb_size1; col++) {

                int target_row = neighbors[row * neighb_size1 + col];

                if (target_row >= 0 && (target_row+phase)%2 == 0)
                {
                    for (int target_col = 1; target_col < neighb_size1; target_col++) {
                        int old = atomicCAS( &neighbors[target_row*neighb_size1 + target_col], row, -1);

                        if (old == row) { atomicSub( &neb_counts[target_row], 1); }
                    }
                }

            }
        }

    __syncthreads();
    }
}

// shuffle valid bin_ids to lowest open column in neighbors
// each thread reads a row of void bin_ids from neighbors into a local buffer; writes that row;
//   fills remaining in the row with -1.
__global__ void frnn_neighbors_shuffle(
    int*  neighbors,

    const int num_binstotal,
    const int neighb_size1
){
    extern __shared__ int buffer[];

    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_binstotal; row += blockDim.x * gridDim.x) {

        int buffered = 0;

        for (int col = 1; col < neighb_size1; col++) {

            int curr = neighbors[row * neighb_size1 + col];
            if (curr >= 0)
            {
                buffer[threadIdx.x*neighb_size1 + buffered] = curr;
                buffered++;
            }
        }

        int col;
        for (col = 0; col <= buffered; col++) {

            neighbors[row * neighb_size1 + col + 1] = buffer[threadIdx.x*neighb_size1 + col];
        }
        for (int end = col; end < neighb_size1; end++) {
            
            neighbors[row * neighb_size1 + end] = -1;
        }
    }
}



// build frnn_ptids. bin_wcounts give global counters to write to a given bin's space in bin_ptids. 
__global__ void frnn_ptids(
    const   int*      bin_offsets,
    const   int*      bid2ptid,
            int*      bin_ptids,
            int*      bin_wcounts,

    const   int                   pts_size0
)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pts_size0; i += blockDim.x * gridDim.x)
    {
        int bid =  bid2ptid[ i*2 + 0 ];
        int ptid = bid2ptid[ i*2 + 1 ];

        int boffset = bin_offsets[ bid ];

        int bwrites = atomicAdd( &bin_wcounts[bid], 1 );
        bin_ptids[boffset + bwrites] = ptid;
    }
}

// This kernel must be defined in this file, where the AT_DISPATCH macro is used to call it.
// count the number of points in each bin.
// write_count gives a global counter for threads to write to bid2ptid
// linear radii bins are 2*lin_radius wide; scale bins are 1*scale_radius wide.
template <typename scalar_t>
__global__ void frnn_bin_counts(
        const   scalar_t*    pts,
        const   long*     imgid,
        const   float*    radius_at_each_scabin,
        const   int*      scabin_offsets,
                int*      bin_counts,
                int*      bid2ptid,
                int*      write_count,

        const   float     scale_radius,
        const   float     xmin,
        const   float     scalemin,
        const   int       pts_size0,
        const   int       pts_size1,
        const   int       num_bins_perimg,
        const   int       num_binstotal,
        const   int       pts_x_col,
        const   int       pts_scale_col
){
    int thid_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int tot_threads = gridDim.x * blockDim.x;

    for (int i = thid_grid; i < pts_size0; i += tot_threads)
    {
        float x =         float( pts[i * pts_size1 + pts_x_col] );
        float lin_scale = float( pts[i * pts_size1 + pts_scale_col] );

        int scabin_id =         floorf( (log(lin_scale) - scalemin) / scale_radius);
        float my_scabin_linrad =   radius_at_each_scabin[scabin_id];
        int xbin_id =           floorf( (x - xmin) / (2*my_scabin_linrad) );

        int bin_id = imgid[i] * num_bins_perimg + scabin_offsets[scabin_id] + xbin_id;

        atomicAdd( &bin_counts[bin_id], 1 );

        int write_i = atomicAdd( &write_count[0], 2 );
        bid2ptid[write_i + 0] = bin_id;
        bid2ptid[write_i + 1] = i;
    }
}

__host__ void frnn_bin_counts_launch(
    torch::Tensor   pts,
    long*     imgid,
    float*   radius_at_each_scabin,
    int*     scabin_offsets,
    int*     bin_counts,
    int*     bid2ptid,
    int*     write_count,

    float   scale_radius,
    float   xmin,
    float   scalemin,
    int     pts_size0,
    int     pts_size1,
    int     num_bins_perimg,
    int     num_binstotal,
    int pts_x_col,
    int pts_scale_col,

    dim3 blocks,
    dim3 threads
){
    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, pts.scalar_type(), "frnn_bin_counts", ([&] {
        frnn_bin_counts<<<blocks, threads>>>(
                pts.data_ptr<scalar_t>(),
                imgid,
                radius_at_each_scabin,
                scabin_offsets,
                bin_counts,
                bid2ptid,
                write_count,

                scale_radius,
                xmin,
                scalemin,
                pts_size0,
                pts_size1,
                num_bins_perimg,
                num_binstotal,

                pts_x_col,
                pts_scale_col
        );
    }));
}

