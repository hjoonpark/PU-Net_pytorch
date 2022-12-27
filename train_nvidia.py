import matplotlib.pyplot as plt

import argparse
import os

from utils_mine import Logger, Plotter, GPUStat
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
        print(">> get_emd_loss")
        return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred):
        print("1: get_repulsion_loss")
        print("pred:", pred.shape, pred.dtype, pred.device, pred.is_contiguous(), pred.min().item(), pred.max().item(), pred.mean().item())
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        # torch.Size([6, 40816, 4])
        print(">>",idx.shape)
        print("1: get_repulsion_loss")
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        print("1: get_repulsion_loss")
        idx = idx.contiguous() # B, N, nn

        print("3: get_repulsion_loss")
        pred = pred.transpose(1, 2).contiguous() # B, 3, N
        print("3: get_repulsion_loss")
        print(pred.shape)
        print(idx.shape)
        grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)
        print("3: get_repulsion_loss")

        grouped_points = grouped_points - pred.unsqueeze(-1)
        print("1: get_repulsion_loss")
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)
        print("2: get_repulsion_loss")

        uniform_loss = torch.mean((self.radius - dist) * weight)
        print("3: get_repulsion_loss")
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        print(">> get_repulsion_loss")
        return uniform_loss

    def forward(self, pred, gt, pcd_radius):
        print("loss:", pred.shape, gt.shape, pcd_radius.shape)
        emd_loss = self.get_emd_loss(pred, gt, pcd_radius) * 100
        print("emd_loss:", emd_loss.shape)
        rep_loss = self.get_repulsion_loss(pred)
        print("rep_loss:", rep_loss.shape)
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


params = {
  "batch_size": 1,
  "n_epochs": 100000000,
  "print_freq": 1,
  "val_freq": 50,
  "chkpt_freq": 50,
  "starting_epoch": 2900,
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
    out_dir = args.log_dir
    args.batch_size = params["batch_size"]

    gpu_stat = GPUStat()
    logger = Logger(os.path.join(out_dir, "log.txt"))
    logger.print("========================================")
    logger.print("{}".format(args))
    logger.print("========================================")
    
    load_opt = ["metadata", "face_normals"]
    from dataset_nvidia.nvidia_face_dataset_v4 import DatasetVertxDx
    dataset_tr = DatasetVertxDx(is_train=True, load_opt=load_opt, logger=logger)
    sd = dataset_tr.static_data

    logger.print("dataset_tr:", len(dataset_tr))
    train_loader = DataLoader(dataset_tr, batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True, num_workers=args.workers)
                        
    print("train_loader:", len(train_loader))
    logger.print('models.' + args.model)
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=sd["n_ldx"], up_ratio=args.up_ratio, 
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
    
    torch.cuda.empty_cache()
    for epoch in range(args.max_epoch):

        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            lx_pos = batch["lx_pos"].cuda()
            hx_pos = batch["hx_pos"].cuda()
            radius_data = radius_data0.repeat(lx_pos.shape[0], 1)
            print("lx_pos:", stat(lx_pos.cpu().numpy()))
            print("hx_pos:", stat(hx_pos.cpu().numpy()))
            print("radius_data:", stat(radius_data.cpu().numpy()))

            # save_path = os.path.join(out_dir, "{}_input.jpg".format(batch_idx))
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # x = input_data[0, :, 0:3].cpu().numpy()
            # ax.scatter(x[:,0], x[:,1], x[:,2])
            # plt.title("{}".format(x.shape))
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # plt.savefig(save_path, dpi=150)
            # plt.close()
            # logger.print(save_path)

            # save_path = os.path.join(out_dir, "{}_true.jpg".format(batch_idx))
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # x = gt_data[0, :, 0:3].cpu().numpy()
            # ax.scatter(x[:,0], x[:,1], x[:,2])
            # plt.title("{}".format(x.shape))
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # plt.savefig(save_path, dpi=150)
            # plt.close()

            preds = model(lx_pos)
            print("preds:", preds.shape, preds.device)
            emd_loss, rep_loss = loss_func(preds, hx_pos, radius_data)
            loss = emd_loss + rep_loss

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            emd_loss_list.append(emd_loss.item())
            rep_loss_list.append(rep_loss.item())
            break
        if epoch == 0:
            logger.print(gpu_stat.get_stat_str())
        assert 0
        print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, repulsion loss {:.4f}, lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if (epoch + 1) % 20 == 0:
            state = {'epoch': epoch, 'model_state': model.state_dict()}
            save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
            torch.save(state, save_path)