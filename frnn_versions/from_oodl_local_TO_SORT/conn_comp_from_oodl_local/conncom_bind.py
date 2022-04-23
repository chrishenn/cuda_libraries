import conn_comp.conncom_kernels as conncom_kern

import numpy as np
import numba.cuda as cuda

import torch as t


def get_conn_comp(edges, imgid):
    '''
    TODO: this is where we would switch on gpu vs. cpu impl, depending on device availability
    '''

    return conncom_forward(edges, imgid.size(0))



def conncom_forward( d_edges, num_n ):
    num_e = d_edges.size(0)

    # Cuda set device; init cuda stream
    torch_device = d_edges.device
    t.cuda.synchronize(torch_device)
    cuda.select_device(torch_device.index)

    ## allocate
    d_an = cuda.device_array(num_n, dtype=np.int32)
    d_an_writeonce = cuda.device_array(num_n, dtype=np.int32)
    d_mask = cuda.device_array(num_n, dtype=np.int32)
    d_mark = cuda.device_array(num_e, dtype=np.int32)

    threads = 512
    grid_n = (num_n - 1) // threads +1
    grid_e = (num_e - 1) // threads +1

    ## init
    conncom_kern.init_zero[grid_e, threads](d_mark, num_e)
    conncom_kern.init_zero[grid_n, threads](d_an_writeonce, num_n)
    conncom_kern.init_sequential[grid_n, threads](d_an, num_n)
    conncom_kern.select_winner_init[grid_e, threads](d_an, d_an_writeonce, d_edges, num_e)

    while True:
        flag = np.array([0])
        d_flag = cuda.to_device(flag)

        conncom_kern.pointer_jump[grid_n, threads](num_n, d_an, d_flag)

        d_flag.copy_to_host(flag)
        if flag[0] == 0: break

    # Main loop
    conncom_kern.update_mask[grid_n, threads](d_mask, num_n, d_an)

    lpc = 0
    while True:
        flag = np.array([0])
        d_flag = cuda.to_device(flag)

        if lpc % 4 == 0:
            conncom_kern.connect_low2hi[grid_e, threads](d_an, d_edges, num_e, d_flag, d_mark)
        else:
            conncom_kern.connect_hi2low[grid_e, threads](d_an, d_edges, num_e, d_flag, d_mark)
        lpc += 1

        d_flag.copy_to_host(flag)
        if flag[0] == 0: break

        while True:
            inner_flag = np.array([0])
            d_inner_flag = cuda.to_device(inner_flag)

            conncom_kern.pointer_jump_masked[grid_n, threads](num_n, d_an, d_inner_flag, d_mask)

            d_inner_flag.copy_to_host(inner_flag)
            if inner_flag[0] == 0: break

        conncom_kern.pointer_jump_unmasked[grid_n, threads](num_n, d_an, d_mask)
        conncom_kern.update_mask[grid_n, threads](d_mask, num_n, d_an)

    return t.tensor(d_an, device=torch_device)


