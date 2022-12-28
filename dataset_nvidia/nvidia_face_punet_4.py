import random
import numpy as np
import os
import torch
import torchvision
import glob
import json

def read_obj(path):
    faces = []
    hv = []
    hvn = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            ls = line.split(" ")
            if ls[0].lower() == "f":
                f1 = int(ls[1].split("//")[0])
                f2 = int(ls[2].split("//")[0])
                f3 = int(ls[3].split("//")[0])
                faces.append([f1, f2, f3])
            elif ls[0].lower() == "v":
                v1 = float(ls[1])
                v2 = float(ls[2])
                v3 = float(ls[3])
                hv.append([v1, v2, v3])
            elif ls[0].lower() == "vn":
                vn1 = float(ls[1])
                vn2 = float(ls[2])
                vn3 = float(ls[3])
                hvn.append([vn1, vn2, vn3])
                
    hv = np.array(hv)
    hvn = np.array(hvn)
    faces = np.array(faces)-1
    return hv, hvn, faces
class DatasetVertxDx(torch.utils.data.Dataset):
    def __init__(self, is_train, load_opt, logger):
        in_dir = "datas/nvidia_punet_v4"
        
        self.frames = []
        self.seq_frames = []
        self.seq_names = []
        self.patch_indices = []
        self.lx_pos = []
        self.hx_pos = []
        self.static_data = {}
        self.framewise_data_names = ["frames", "seq_frames", "seq_names", "lx_pos", "hx_pos"]

        # patch info
        lres_clusters = torch.from_numpy(np.load(os.path.join(in_dir, "lres_clusters.npy"))).long()
        hres_clusters = torch.from_numpy(np.load(os.path.join(in_dir, "hres_clusters.npy"))).long()

        # load restshapes
        lrestshape, _, lfaces = read_obj(os.path.join(in_dir, "lrestshape.obj"))
        hrestshape, _, hfaces = read_obj(os.path.join(in_dir, "hrestshape.obj"))

        # load paths
        if is_train:
            hpaths = sorted(list(glob.glob(os.path.join(in_dir, "high", "x", "*amazement*.npy"))) + list(glob.glob(os.path.join(in_dir, "high", "x", "*pain*.npy"))))
        else:
            # test
            hpaths = sorted(list(glob.glob(os.path.join(in_dir, "high", "x", "*fear*.npy"))) + list(glob.glob(os.path.join(in_dir, "high", "x", "*anger*.npy"))))

        # load both high & low res
        if is_train:
            N = 3
        else:
            N = -1

        for i, hpath in enumerate(hpaths):
            if i == N-1:
                break
            basename = os.path.basename(hpath)
            frame, seq_name, seq_frame, patch_idx = basename.split(".")[0].split("_")
            lpath = os.path.join(in_dir, "low", "x", basename)

            # load x
            lx = np.load(lpath)
            hx = np.load(hpath)

            self.frames.append(frame)
            self.seq_names.append(seq_name)
            self.seq_frames.append(seq_frame)
            self.patch_indices.append(patch_idx)

            self.lx_pos.append(lx)
            self.hx_pos.append(hx)
        
        self.lx_pos = torch.tensor(np.array(self.lx_pos)).float()
        self.hx_pos = torch.tensor(np.array(self.hx_pos)).float()

        # normalize/standardize here
        with open(os.path.join(in_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
            x_stat = meta["lrestshape"]
            lx_stat = meta["lxs"]

            lmax = meta["lrestshape"]["max"]
            lmin = meta["lrestshape"]["min"]
            hmu = 0.5*(lmax+lmin)
            hsigma = 0.5*(lmax-lmin)

            lrestshape = (lrestshape-hmu)/hsigma
            hrestshape = (hrestshape-hmu)/hsigma
            self.lx_pos = (self.lx_pos-hmu)/hsigma
            self.hx_pos = (self.hx_pos-hmu)/hsigma

        if N > 0:
            for name in self.framewise_data_names:
                data = getattr(self, name)
                data = data[:N]
                setattr(self, name, data)

        if logger is not None:
            logger.print("Restshapes")
            logger.print("    lrestshape: {}, min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f})".format(lrestshape.shape, lrestshape.min(), lrestshape.max(), lrestshape.mean(), lrestshape.std()))
            logger.print("    hrestshape: {}, min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f})".format(hrestshape.shape, hrestshape.min(), hrestshape.max(), hrestshape.mean(), hrestshape.std()))
            logger.print("Dataset info")
            logger.print("    lx_pos       {}: min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f})".format(self.lx_pos.shape, self.lx_pos.min(), self.lx_pos.max(), self.lx_pos.mean(), self.lx_pos.std()))
            logger.print("    hx_pos       {}: min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f})".format(self.hx_pos.shape, self.hx_pos.min(), self.hx_pos.max(), self.hx_pos.mean(), self.hx_pos.std()))

        self.static_data["lres_clusters"] = lres_clusters
        self.static_data["hres_clusters"] = hres_clusters
        self.static_data["n_lx"] = self.lx_pos.shape[1]
        self.static_data["n_hx"] = self.hx_pos.shape[1]
        self.static_data["lrestshape"] = lrestshape
        self.static_data["hrestshape"] = hrestshape
        self.static_data["hfaces"] = hfaces
        self.static_data["lfaces"] = lfaces
        self.static_data["hmu"] = hmu
        self.static_data["hsigma"] = hsigma

    def __getitem__(self, index):
        frame = self.frames[index]
        seq_frame = self.seq_frames[index]
        seq_name = self.seq_names[index]
        patch_idx = self.patch_indices[index]
        lx_pos = self.lx_pos[index]
        hx_pos = self.hx_pos[index]

        out = {"frame": frame, "seq_frame": seq_frame, "seq_name": seq_name, "patch_idx": patch_idx, "lx_pos": lx_pos, "hx_pos": hx_pos}
            
        return out

    def __len__(self):
        return len(self.frames)

class DatasetGroupedPatch(torch.utils.data.Dataset):
    def __init__(self, frames_to_load):
        in_dir = "datas/nvidia_punet_v4"
        
        self.frames = []
        self.seq_frames = []
        self.seq_names = []
        self.patch_indices = []
        self.lx_pos = []
        self.hx_pos = []

        # load paths
        hpaths = {}
        lpaths = {}
        for seq_name, seq_frames in frames_to_load.items():
            if seq_name not in hpaths:
                hpaths[seq_name] = []
                lpaths[seq_name] = []
            for seq_frame in seq_frames:
                hps = sorted(glob.glob(os.path.join(in_dir, "high", "x", f"*_{seq_name}_{seq_frame:03d}_*.npy")))
                hpaths[seq_name].extend(hps)
                lps = sorted(glob.glob(os.path.join(in_dir, "low", "x", f"*_{seq_name}_{seq_frame:03d}_*.npy")))
                lpaths[seq_name].extend(lps)

        for seq_name in hpaths.keys():
            hpatch_paths = hpaths[seq_name]
            lpatch_paths = lpaths[seq_name]

            for i in range(len(hpatch_paths)):
                hpp = hpatch_paths[i]
                lpp = lpatch_paths[i]
                basename = os.path.basename(hpp)
                frame, seq_name, seq_frame, patch_idx = basename.split(".")[0].split("_")

                # load x
                lx = np.load(lpp)
                hx = np.load(hpp)

                self.frames.append(frame)
                self.seq_names.append(seq_name)
                self.seq_frames.append(seq_frame)
                self.patch_indices.append(patch_idx)

                self.lx_pos.append(lx)
                self.hx_pos.append(hx)
            
        self.lx_pos = torch.tensor(np.array(self.lx_pos)).float()
        self.hx_pos = torch.tensor(np.array(self.hx_pos)).float()

        # normalize/standardize here
        with open(os.path.join(in_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
            x_stat = meta["lrestshape"]
            lx_stat = meta["lxs"]

            lmax = meta["lrestshape"]["max"]
            lmin = meta["lrestshape"]["min"]
            hmu = 0.5*(lmax+lmin)
            hsigma = 0.5*(lmax-lmin)

            self.lx_pos = (self.lx_pos-hmu)/hsigma
            self.hx_pos = (self.hx_pos-hmu)/hsigma

        print("Dataset info")
        print("    lx_pos       {}: min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f})".format(self.lx_pos.shape, self.lx_pos.min(), self.lx_pos.max(), self.lx_pos.mean(), self.lx_pos.std()))
        print("    hx_pos       {}: min/max=({:.2f}, {:.2f}), mean={:.2f}, std={:.2f})".format(self.hx_pos.shape, self.hx_pos.min(), self.hx_pos.max(), self.hx_pos.mean(), self.hx_pos.std()))

    def __getitem__(self, index):
        frame = self.frames[index]
        seq_frame = self.seq_frames[index]
        seq_name = self.seq_names[index]
        patch_idx = self.patch_indices[index]
        lx_pos = self.lx_pos[index]
        hx_pos = self.hx_pos[index]

        out = {"frame": frame, "seq_frame": seq_frame, "seq_name": seq_name, "patch_idx": patch_idx, "lx_pos": lx_pos, "hx_pos": hx_pos}
            
        return out

    def __len__(self):
        return len(self.frames)