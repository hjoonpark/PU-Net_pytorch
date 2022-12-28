import matplotlib.pyplot as plt

import argparse
import os, shutil, json, time, glob

import torch.nn.functional as F
from torch_scatter import scatter_mean

from utils_mine import Logger, Plotter, GPUStat, make_output_folders
from utils_mine.log import get_loss_string
from utils_mine.obj_writer import write_obj_triangle
parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--log_dir', default='logs/test', help='Log dir [default: logs/test_log]')
parser.add_argument('--npoint', type=int, default=1024,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=1.0) # for repulsion loss
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--use_decay', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.71)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[30, 60])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
from auction_match import auction_match

from dataset import PUNET_Dataset
import numpy as np
import importlib

def stat(x):
    return "{} min/max=({:.4f}, {:.4f}), mean={:.4f}, std={:.4f}".format(x.shape, x.min(), x.max(), x.mean(), x.std())

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps

    # def knn(self, x, k, return_dist=False):
    #     print("--------------- knn")
    #     print(x.shape, x.is_contiguous())
    #     B, dim, n = x.shape
    #     # xT = x.transpose(2, 1)
    #     # print("xT:", xT.shape, x.shape)
    #     # inner = -2*torch.matmul(xT, x)
    #     # print(inner.shape)
    #     # xx = torch.sum(x**2, dim=1, keepdim=True)
    #     # print(xx.shape)
    #     # pairwise_distance = -xx - inner - xx.view(-1, n, dim)
        
    #     pairwise_distance = torch.cdist(x, x) # (B, dim, N)
    #     print(">>>> pairwise_distance:", pairwise_distance.shape, pairwise_distance.device)
    #     # print("pairwise_distance:", pairwise_distance.shape, pairwise_distance.topk(k=k, dim=-1)[1].clone().shape)
    #     # idx = pairwise_distance.topk(k=k, dim=-1)[1].clone() # (batch_size, num_points, k)
    #     idx = torch.argsort(pairwise_distance, dim=1)[:k] # (batch_size, num_points, k)
    #     print(">>>> idx:", idx.shape)
    #     if return_dist:
    #         return idx, -pairwise_distance
    #     else:
    #         return idx, None

    def get_emd_loss(self, pred, gt, pcd_radius):
        idx, _ = auction_match(pred, gt)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
        dist2 /= pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred):
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        # torch.Size([6, 40816, 4])
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        idx = idx.contiguous() # B, N, nn

        pred = pred.transpose(1, 2).contiguous() # B, 3, N
        grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        return uniform_loss

    def forward(self, pred, gt, pcd_radius):
        emd_loss = self.get_emd_loss(pred, gt, pcd_radius) * 100
        rep_loss = self.get_repulsion_loss(pred)
        return emd_loss, self.alpha * rep_loss

def get_optimizer():
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    else:
        raise NotImplementedError
    
    if args.use_decay:
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in args.decay_step_list:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * args.lr_decay
            return max(cur_decay, args.lr_clip / args.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
        return optimizer, lr_scheduler
    else:
        return optimizer, None

def as_np(x):
    return x.detach().cpu().numpy()

def stitch(patches, trg_shape, index):
    if patches.device != index.device:
        patches = patches.cuda()
        index = index.cuda()
    out1 = torch.zeros(trg_shape)[:,0].to(patches.device)
    out2 = torch.zeros(trg_shape)[:,1].to(patches.device)
    out3 = torch.zeros(trg_shape)[:,2].to(patches.device)
    idx = index.reshape(-1)
    src1 = patches[:, :, 0].reshape(-1)
    src2 = patches[:, :, 1].reshape(-1)
    src3 = patches[:, :, 2].reshape(-1)
    out1 = scatter_mean(src1, idx, out=out1)[:, None]
    out2 = scatter_mean(src2, idx, out=out2)[:, None]
    out3 = scatter_mean(src3, idx, out=out3)[:, None]
    out = torch.cat((out1, out2, out3), dim=1)
    return out.cpu()

params = {
  "batch_size": 1,
  "n_epochs": 100000000,
  "print_freq": 1,
  "val_freq": 5,
  "chkpt_freq": 5,
  "starting_epoch": 0,
  "lr": 0.0001,
  "drop_out": 0.2,
  "knn": 5,
  "pos_feat_dim": 32,
  "emb_feat_dims": [35, 64, 128],
  "w_up_feat_dim": 128,
  "decoder_dim": 256,
  "n_interpolate_samples": 10,
  "n_upscale_layers": 8,
  "w_face_normal": 0.1,
  "w_lap_smooth": -1,
  "z_reg": 0.01
}
if __name__ == '__main__':
    set_all_seeds(0)

    root_dir = "/nobackup/joon/1_Projects/PU-Net_pytorch"
    output_dir = os.path.join(root_dir, "output_nvidia")

    # ========================================================================
    out_dirs = make_output_folders(output_dir, ['log', 'model', 'trobj', 'val'], makedirs=True)
    log_path = os.path.join(out_dirs['log'], 'loss_log.txt')
    plot_path = os.path.join(out_dirs['log'], "loss_plot.jpg")
    logger = Logger(os.path.join(out_dirs["log"], "log.txt"))
    plotter = Plotter()
    gpu_stat = GPUStat()
    
    args.batch_size = params["batch_size"]

    logger.print("========================================")
    logger.print("{}".format(args))
    logger.print("========================================")

    try:
        ml_path = os.path.join(out_dirs["model"], "model_losses.json")
        model_losses = json.load(open(ml_path, "r"))
    except:
        model_losses = {} # {folder_name: {'val': }}


    losses_per_epoch = {}
    val_losses_per_epoch = {"mean": []}
    val_epochs = []

    from dataset_nvidia.nvidia_face_punet_4 import DatasetGroupedPatch
    validation_frames = {
        "fear": [130, 99],
        "anger": [9, 112]
    }
    trobj_frames = {
        "pain": [123]
    }
    dataset_val = DatasetGroupedPatch(frames_to_load=validation_frames)
    dataset_trobj = DatasetGroupedPatch(frames_to_load=trobj_frames)
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)
    loader_trobj = DataLoader(dataset_trobj, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)
    logger.print("dataset_trobj: {}, dataset_val: {}".format(len(dataset_trobj), len(dataset_val)))
    # ========================================================================

    load_opt = ["metadata"]
    from dataset_nvidia.nvidia_face_punet_4 import DatasetVertxDx
    dataset_tr = DatasetVertxDx(is_train=True, load_opt=load_opt, logger=logger)
    sd = dataset_tr.static_data
    # save normalization data
    save_path = os.path.join(out_dirs["log"], "mu_sigma.json")
    out = {
        "hmu": sd["hmu"], "hsigma": sd["hsigma"]
    }
    json.dump(out, open(save_path, "w+"), indent=4)

    logger.print("dataset_tr:", len(dataset_tr))
    train_loader = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
                        
    print("train_loader:", len(train_loader))
    logger.print('models.' + args.model)
    MODEL = importlib.import_module('models.' + args.model + "_nvidia")
    model = MODEL.get_model(npoint=sd["n_lx"], up_ratio=args.up_ratio, 
                use_normal=False, use_bn=args.use_bn, use_res=args.use_res)
    # model.cuda()
    model = model.cuda()
    
    optimizer, lr_scheduler = get_optimizer()
    loss_func = UpsampleLoss(alpha=args.alpha)

    model.train()
    logger.print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.print("n_trainable_params={:,}".format(n_trainable_params))

    radius_data0 = torch.tensor([sd["hsigma"]]).float().cuda()[None, :]
    epoch0 = params['starting_epoch']

    lres_clusters = sd["lres_clusters"]
    hres_clusters = sd["hres_clusters"]
    lshape = sd['lrestshape'].shape
    hshape = sd['hrestshape'].shape
    sigma = sd['hsigma']
    torch.cuda.empty_cache()
    for epoch in range(args.max_epoch):
        model.train()
        loss_curr = {}
        save_plots = ((epoch-epoch0) == 0 or epoch % params['val_freq'] == 0)

        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            lx_pos = batch["lx_pos"].cuda()
            hx_pos = batch["hx_pos"].cuda()
            radius_data = radius_data0.repeat(lx_pos.shape[0], 1)

            preds = model(lx_pos)
            emd_loss, rep_loss = loss_func(preds, hx_pos, radius_data)
            loss = emd_loss + rep_loss

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            emd_loss_list.append(emd_loss.item())
            rep_loss_list.append(rep_loss.item())

            # --------------------------------------------- #
            losses = {
                "emd": emd_loss.item(), "rep": rep_loss.item()
            }
            for k, v in losses.items():
                if k not in loss_curr:
                    loss_curr[k] = 0
                loss_curr[k] += v / len(train_loader)
            # --------------------------------------------- #

            batch_idx += 1

        print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, repulsion loss {:.4f}, lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        if (epoch + 1) % 20 == 0:
            state = {'epoch': epoch, 'model_state': model.state_dict()}
            save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
            torch.save(state, save_path)

        # batch iteration ends
        for k, v in loss_curr.items():
            if k not in losses_per_epoch:
                losses_per_epoch[k] = []
            losses_per_epoch[k].append(v)

        if epoch == 0:
            logger.print(gpu_stat.get_stat_str())

        if save_plots:
            t0 = time.time()

            # delete objs before saving new
            files = glob.glob(os.path.join(out_dirs["trobj"], "*.obj"))
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))

            with torch.no_grad():
                patches = {} # seq_name_seq_frame_patch_idx
                for data_tr in loader_trobj:
                    frame = data_tr["frame"][0]
                    seq_name = data_tr["seq_name"][0]
                    seq_frame = data_tr["seq_frame"][0]
                    patch_idx = int(data_tr["patch_idx"][0])
                    key = f"{frame}_{seq_name}_{seq_frame}"
                    lx_pos = data_tr["lx_pos"].cuda()
                    hx_pred = model(lx_pos)
                    hx_pos = data_tr["hx_pos"]
                    if key not in patches:
                        patches[key] = {}

                    if patch_idx not in patches[key]:
                        patches[key][patch_idx] = {}
                    patches[key][patch_idx]['lx'] = lx_pos
                    patches[key][patch_idx]['hx_pred'] = hx_pred
                    patches[key][patch_idx]['hx_true'] = hx_pos
                
                # stitch
                for key in patches.keys():
                    d = patches[key]
                    lxs = []
                    hx_trues = []
                    hx_preds = []
                    for patch_idx in range(len(lres_clusters)):
                        lxs.append(d[patch_idx]["lx"])
                        hx_preds.append(d[patch_idx]["hx_pred"])
                        hx_trues.append(d[patch_idx]["hx_true"])
                
                    lxs = torch.cat(lxs, dim=0)
                    hx_trues = torch.cat(hx_trues, dim=0)
                    hx_preds = torch.cat(hx_preds, dim=0)
                    lxs = stitch(lxs, lshape, lres_clusters)
                    hx_trues = stitch(hx_trues, hshape, hres_clusters)
                    hx_preds = stitch(hx_preds, hshape, hres_clusters)
                    write_obj_triangle(os.path.join(out_dirs["trobj"], "{}_{}_low.obj".format(epoch, key)), as_np(lxs), sd["lfaces"])
                    write_obj_triangle(os.path.join(out_dirs["trobj"], "{}_{}_htrue.obj".format(epoch, key)), as_np(hx_trues), sd["hfaces"])
                    write_obj_triangle(os.path.join(out_dirs["trobj"], "{}_{}_hpred.obj".format(epoch, key)), as_np(hx_preds), sd["hfaces"])
                
        # ===============
        # validation
        # ===============
        if ((epoch - epoch0) > 0 and (epoch - epoch0) % params['chkpt_freq'] == 0):
            plotter.plot_current_losses(plot_path, epoch0, epoch, losses_per_epoch)
            logger.print("- {}".format(plot_path))

            logger.print(gpu_stat.get_stat_str())

            # validate
            if epoch > 0:
                with torch.no_grad():
                    losses_val_keyframes = {}
                    losses_val = []

                    model.eval()
                    patches = {} # seq_name_seq_frame_patch_idx
                    for data_te in loader_val:
                        frame = data_te["frame"][0]
                        seq_name = data_te["seq_name"][0]
                        seq_frame = data_te["seq_frame"][0]
                        patch_idx = int(data_te["patch_idx"][0])
                        key = f"{frame}_{seq_name}_{seq_frame}"
                        lx_pos = data_te["lx_pos"].cuda()
                        hx_pred = model(lx_pos)
                        hx_pos = data_te["hx_pos"]
                        if key not in patches:
                            patches[key] = {}

                        if patch_idx not in patches[key]:
                            patches[key][patch_idx] = {}
                        patches[key][patch_idx]['lx'] = lx_pos
                        patches[key][patch_idx]['hx_pred'] = hx_pred
                        patches[key][patch_idx]['hx_true'] = hx_pos
                    
                    # stitch
                    for key in patches.keys():
                        d = patches[key]
                        lxs = []
                        hx_trues = []
                        hx_preds = []
                        for patch_idx in range(len(hres_clusters)):
                            lxs.append(d[patch_idx]["lx"])
                            hx_preds.append(d[patch_idx]["hx_pred"])
                            hx_trues.append(d[patch_idx]["hx_true"])
                        lxs = torch.cat(lxs, dim=0)
                        hx_trues = torch.cat(hx_trues, dim=0)
                        hx_preds = torch.cat(hx_preds, dim=0)

                        lxs = stitch(lxs, lshape, lres_clusters)
                        hx_trues = stitch(hx_trues, hshape, hres_clusters)
                        hx_preds = stitch(hx_preds, hshape, hres_clusters)
                        write_obj_triangle(os.path.join(out_dirs["val"], "{}_{}_low.obj".format(epoch, key)), as_np(lxs), sd["lfaces"])
                        write_obj_triangle(os.path.join(out_dirs["val"], "{}_{}_htrue.obj".format(epoch, key)), as_np(hx_trues), sd["hfaces"])
                        write_obj_triangle(os.path.join(out_dirs["val"], "{}_{}_hpred.obj".format(epoch, key)), as_np(hx_preds), sd["hfaces"])

                        loss = F.l1_loss(hx_preds, hx_trues).detach().item()*sigma
                        losses_val.append(loss)

                    model.train()
                losses_val = np.float32(losses_val)

                # save model validation losses
                model_name_curr = "{:07d}".format(epoch)
                model_losses[model_name_curr] = {"val": {"min": float(losses_val.min()), "max": float(losses_val.max()), "mean": float(losses_val.mean()), "std": float(losses_val.std())}}
                save_path = os.path.join(out_dirs["model"], "model_losses.json")
                json.dump(model_losses, open(save_path, "w+"), indent=4)
                logger.print(f"- Model losses saved: {save_path}")
    print("### DONE")