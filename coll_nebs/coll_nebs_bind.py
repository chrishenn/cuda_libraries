import torch
import torch as t
import torch.utils.data
import torchvision
from torchvision import transforms

from frnn import frnn_bind
import coll_nebs_cuda

import oomodel.oovis as oovis


def get_coll_nebs(data, lin_radius, scale_radius, batch_size, coll_iters, img_size, thresh=1, dampen_fact=.125):

    tex, pts, imgid = data[0],data[1],data[2]
    device = pts.device
    if not isinstance(device, int): device = device.index

    tex_act, pts_act, imgid_act = coll_nebs_cuda.coll_nebs_kernel(
                                tex,
                                pts,
                                imgid,

                                lin_radius,
                                scale_radius,

                                thresh,
                                coll_iters,
                                dampen_fact,

                                batch_size,
                                img_size,

                                device
                                )
    return tex_act, pts_act, imgid_act

def collapse_nebs(data, radius, iters=8):
    pts_temp, imgid = data[1].clone(), data[2]

    for i in range(iters):

        edges = frnn_bind.get_frnn((None, pts_temp, imgid), lin_radius=radius, scale_radius=1.).long()
        locs = pts_temp[:, :2]

        counts = edges.flatten().bincount().add(1)[:,None]
        counts_lf = counts.index_select(0, edges[:, 0])
        counts_rt = counts.index_select(0, edges[:, 1])

        locs_lf = locs.index_select(0, edges[:, 0])
        locs_rt = locs.index_select(0, edges[:, 1])

        vec_lfrt = locs_lf.sub(locs_rt)
        weight = vec_lfrt.norm(p=2, dim=1).add(1).reciprocal()[:, None]

        vec_lfrt = (vec_lfrt * weight)

        damp = ( (i+1) / 12)
        locs.index_add_(0, edges[:, 1], vec_lfrt.div(counts_lf.mul(damp)))
        locs.index_add_(0, edges[:, 0], -vec_lfrt.div(counts_rt.mul(damp)))

    return pts_temp

def collapse_nebs_old(data, radius, iters=20):
    pts_temp, imgid = data[1].clone(), data[2]

    for i in range(iters):
        edges = frnn_bind.get_frnn((None, pts_temp, imgid), lin_radius=radius, scale_radius=1.).long()
        locs = pts_temp[:, :2]

        counts = edges.flatten().bincount().add(1)[:,None]
        counts_lf = counts.index_select(0, edges[:, 0])
        counts_rt = counts.index_select(0, edges[:, 1])

        locs_lf = locs.index_select(0, edges[:, 0])
        locs_rt = locs.index_select(0, edges[:, 1])

        vec_lfrt = locs_lf.sub(locs_rt)
        weight = vec_lfrt.norm(p=2, dim=1).add(0.5).reciprocal()[:, None]

        vec_lfrt = (vec_lfrt * weight)

        locs.index_add_(0, edges[:, 1], vec_lfrt.div(counts_lf))
        locs.index_add_(0, edges[:, 0], -vec_lfrt.div(counts_rt))

    return pts_temp

def make_reps(data, pts_collapsed, img_size, batch_size, thresh=1):
    tex, imgid = data[0], data[2]
    device = tex.device

    locs_coll = pts_collapsed[:,:2].round().long()
    angles = pts_collapsed[:, 3]

    locs_y = locs_coll[:, 0][:, None].clamp(0, img_size - 1)
    locs_x = locs_coll[:, 1][:, None].clamp(0, img_size - 1)

    ## compute neighborhood reps; sum tex and average angles
    ang_batch = t.zeros([batch_size, 2, img_size, img_size], dtype=t.float, device=device)
    channels = t.arange(2, dtype=t.long, device=device).unsqueeze(0)

    ang_batch.index_put_([imgid[:, None], channels, locs_y, locs_x], t.stack([angles.sin(), angles.cos()], 1), accumulate=True)

    tex_batch = t.zeros([batch_size, tex.size(1), img_size, img_size], dtype=t.float, device=device)
    channels = t.arange(tex.size(1), dtype=t.long, device=device).unsqueeze(0)
    tex_batch.index_put_([imgid[:, None], channels, locs_y, locs_x], tex, accumulate=True)

    counts = t.ones([batch_size, 1, img_size, img_size], dtype=t.float, device=device)
    counts.index_put_([imgid.unsqueeze(1), t.tensor(0), locs_y, locs_x], t.ones_like(locs_x).float(), accumulate=True)

    ang_batch = t.atan2(ang_batch[:, 0, ...], ang_batch[:, 1, ...])

    nonzero_mask = counts.gt(thresh).squeeze()
    nonzero_ids = nonzero_mask.nonzero(as_tuple=True)

    tex_act = tex_batch.permute(0, 2, 3, 1)[nonzero_mask]
    ang_act = ang_batch[nonzero_mask][:, None]
    imgid_act = nonzero_ids[0]
    locs_act_y = nonzero_ids[1][:, None].float()
    locs_act_x = nonzero_ids[2][:, None].float()

    pts_act = t.cat([locs_act_y, locs_act_x, t.zeros_like(locs_act_x), ang_act, t.ones_like(locs_act_x), t.arange(ang_act.size(0), dtype=t.float, device=device)[:, None]], 1)

    return (tex_act, pts_act, imgid_act)


###################################################################################################################
###################################################################################################################
#### Debugging
def gen_objects(data, device):
    D1 = t.arange(data.size(2), dtype=t.int)
    D2 = t.arange(data.size(3), dtype=t.int)

    gridy, gridx = t.meshgrid(D1,D2)
    gridy = gridy.unsqueeze(0).unsqueeze(0).repeat(data.size(0),1,1,1).float().to(device)
    gridx = gridx.unsqueeze(0).unsqueeze(0).repeat(data.size(0),1,1,1).float().to(device)
    zeros = t.zeros_like(gridx)
    ones =  t.ones_like(gridx)

    angles = t.zeros_like(gridx)
    pt_ids = t.arange(data.size(0)*data.size(2)*data.size(3), dtype=t.float, device=device).unsqueeze(1)
    geom = t.cat([gridy, gridx, zeros, angles, ones], 1)

    img_ids = t.arange(data.size(0), device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,data.size(2), data.size(3))

    texture =   data.permute([0,2,3,1])
    geom =      geom.permute([0,2,3,1])
    img_ids =   img_ids.permute([0,2,3,1])

    texture =   texture.reshape(-1,texture.size(3))
    geom =      geom.reshape(-1,geom.size(3))
    img_ids =   img_ids.reshape(-1,img_ids.size(3))
    img_ids =   img_ids.squeeze(1)

    geom = t.cat([geom, pt_ids], 1).float()

    return texture.to(device), geom.to(device), img_ids.to(device)

def get_test_loader(batch_size=40, img_size=32, datasize=1000):
    workers = 6

    tran_list = list()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tran_list.append(transforms.Resize([img_size, img_size]))
    tran_list.append(transforms.ToTensor())
    tran_list.append(normalize)
    transform = transforms.Compose(tran_list)

    dataset = torchvision.datasets.CIFAR10(root='~/datasets/brainkit/', train=True, download=True, transform=transform)

    if datasize is not None:
        dataset = torch.utils.data.random_split(dataset, [datasize, len(dataset) - datasize])[0]

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=False)
    return loader

def choose_and_drop(drop_fn):

    def choose_and_drop_fn(data, batch_size, locality_size, img_size, frac, edges, actid):

        actid, active_mask = drop_fn(data, batch_size, locality_size, img_size, frac, edges, actid)
        texture, pts, imgid = data[0], data[1], data[2]

        pts = pts.index_select(0, actid)
        pts[:, 5] = t.arange(pts.size(0), dtype=t.float, device=pts.device)
        texture = texture.index_select(0, actid)
        imgid = imgid.index_select(0, actid)

        return (texture,pts,imgid)

    return choose_and_drop_fn

def dropout_rand(data, batch_size, locality_size, img_size, keep_frac, edges, actid):
    imgid = data[2]

    actid = t.arange(imgid.size(0), dtype=t.long, device=imgid.device)
    scores = t.rand_like(imgid.float())

    active_mask = scores.le(keep_frac)
    actid = actid[active_mask]

    return actid, active_mask

if __name__ == "__main__":
    batch_size=40
    loader = get_test_loader(batch_size, img_size=32)

    drop_fn = choose_and_drop(dropout_rand)

    for data, labels in loader:
        data = gen_objects(data, 1)

        data = drop_fn(data, None, None, None, 0.8, None, None)

        # tex, pts, imgid = data
        # device = pts.device
        # if not isinstance(device, int): device = device.index
        #
        # lin_radius=1.45
        # scale_radius=1.0
        #
        # tex_act, pts_act, imgid_act = coll_nebs_cuda.coll_nebs_kernel(
        #     tex,
        #     pts,
        #     imgid,
        #
        #     device,
        #     lin_radius,
        #     scale_radius,
        #     32,
        #     1
        # )

        pts_coll = collapse_nebs(data, radius=1.45, iters=6)

        oovis.vis_objects(0, pts_coll, data[2], max=32)

        tex_act, pts_act, imgid_act = make_reps(data, pts_coll, 32, batch_size, thresh=4)

        oovis.vis_objects(0, pts_act, imgid_act, max=32)

        break


