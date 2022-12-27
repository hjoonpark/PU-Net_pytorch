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
        load_metadata = "metadata" in load_opt
        load_face_normals = "face_normals" in load_opt
        load_emb_info = "emb_info" in load_opt

        in_dir = "/nobackup/joon/1_Projects/220608_3dSuperRes/scripts/data/nvidia_face_4"
        
        self.frames = []
        self.seq_frames = []
        self.seq_names = []
        # self.ldxs, self.hdxs = [], []
        # self.fns = []
        self.lx_pos = []
        self.hx_pos = []
        self.static_data = {}
        self.framewise_data_names = ["frames", "seq_frames", "seq_names", "lx_pos"]

        # load restshapes
        lrestshape, _, lfaces = read_obj(os.path.join(in_dir, "lrestshape.obj"))
        lrestshape = np.array(lrestshape).astype(np.float32)[:1024, :]
        hrestshape, _, hfaces = read_obj(os.path.join(in_dir, "hrestshape.obj"))
        hrestshape = np.array(hrestshape).astype(np.float32)[:4096, :]

        # low-res attention cluster
        # attn_clusters = torch.tensor(np.load(os.path.join(in_dir, "attn_clusters.npy"))).long()
        # if logger is not None:
        #     logger.print(f"attn_clusters: {attn_clusters.shape}, n_clusters={attn_clusters[:,1].max()+1}")

        # load paths
        if is_train:
            hpaths = sorted(list(glob.glob(os.path.join(in_dir, "high", "dx", "*amazement*.npy"))) + list(glob.glob(os.path.join(in_dir, "high", "dx", "*pain*.npy"))))
        else:
            # test
            hpaths = sorted(list(glob.glob(os.path.join(in_dir, "high", "dx", "*fear*.npy"))) + list(glob.glob(os.path.join(in_dir, "high", "dx", "*anger*.npy"))))

        # load both high & low res
        if is_train:
            N = -1
        else:
            N = -1

        for i, hpath in enumerate(hpaths):
            if i == N-1:
                break
            basename = os.path.basename(hpath)
            frame, seq_name, seq_frame = basename.split(".")[0].split("_")
            lpath = os.path.join(in_dir, "low", "dx", basename)
            # fn_path = os.path.join(in_dir, "high", "face_normals", basename)

            # load dx
            ldx = np.load(lpath)[:1024, :]
            hdx = np.load(hpath)[:4096, :]

            # load face normals
            # fn = np.load(fn_path)

            self.frames.append(frame)
            self.seq_names.append(seq_name)
            self.seq_frames.append(seq_frame)

            # self.hdxs.append(hdx)
            # self.ldxs.append(ldx)
            # self.fns.append(fn)

            lx_pos = ldx + lrestshape
            hx_pos = hdx + hrestshape
            self.lx_pos.append(lx_pos)
            self.hx_pos.append(hx_pos)
        
        # self.ldxs = torch.tensor(np.array(self.ldxs)).float()
        # self.hdxs = torch.tensor(np.array(self.hdxs)).float()
        self.lx_pos = torch.tensor(np.array(self.lx_pos)).float()
        self.hx_pos = torch.tensor(np.array(self.hx_pos)).float()
        # self.fns = torch.from_numpy(np.array(self.fns)).float()

        # normalize/standardize here
        with open(os.path.join(in_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
            x_stat = meta["lrestshape"]
            dx_stat = meta["ldxs"]

            # hmu = x_stat["mean"]
            # hsigma = x_stat["std"]
            # mu = dx_stat["mean"]
            # sigma = dx_stat["std"]

            # restshapes: [-1, 1], dxs: [-1, 1]
            # mu = 0.5*(dx_stat["max"]+dx_stat["min"])
            # sigma = 0.5*(dx_stat["max"]-dx_stat["min"])
            # hmu = mu #0.5*(x_stat["max"]+x_stat["min"])
            # hsigma = sigma #0.5*(x_stat["max"]-x_stat["min"])
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

        # load distance mtx: only from non-embedded surface points to low-res points
        # in_path = os.path.join(in_dir, "geo_dists_surf2tet.npy")
        # dist_mtx = torch.from_numpy(np.load(in_path)).float()
        # dist_mtx = dist_mtx/hsigma
        # if logger is not None:
        #     logger.print(f"Distance mtx: {in_path}\n    {dist_mtx.shape}, min/max=({dist_mtx.min():.2f}, {dist_mtx.max():.2f}), mean={dist_mtx.mean():.2f}, std={dist_mtx.std():.2f}")

        # # load geodesic mtx for EdgeConv
        # in_path1 = os.path.join(in_dir, "geo_dists_tet2tet_top100_idx.npy")
        # tet2tet_dist_mtx_idx = torch.from_numpy(np.load(in_path1)).long()
        # in_path2 = os.path.join(in_dir, "geo_dists_tet2tet_top100.npy")
        # tet2tet_dist_mtx = torch.from_numpy(np.load(in_path2)).float()
        # tet2tet_dist_mtx /= hsigma
        # if logger is not None:
        #     logger.print(f"Tet2Tet Distance mtx sorted        : {tet2tet_dist_mtx.shape}, min/max=({tet2tet_dist_mtx.min():.2f}, {tet2tet_dist_mtx.max():.2f}), mean={tet2tet_dist_mtx.mean():.2f}, std={tet2tet_dist_mtx.std():.2f}")
        #     logger.print(f"Tet2Tet Distance mtx sorted indices: {tet2tet_dist_mtx_idx.shape}, min/max=({tet2tet_dist_mtx_idx.min():.2f}, {tet2tet_dist_mtx_idx.max():.2f})")

        # # load geodesic mtx for upscaling
        # in_path1 = os.path.join(in_dir, "geo_dist_surf2tet_all_top100_idx.npy")
        # dist_mtx_idx = torch.from_numpy(np.load(in_path1)).long()
        # in_path2 = os.path.join(in_dir, "geo_dist_surf2tet_all_top100.npy")
        # dist_mtx = torch.from_numpy(np.load(in_path2)).float()
        # dist_mtx /= hsigma
        # if logger is not None:
        #     logger.print(f"Distance mtx sorted        : {dist_mtx.shape}, min/max=({dist_mtx.min():.2f}, {dist_mtx.max():.2f}), mean={dist_mtx.mean():.2f}, std={dist_mtx.std():.2f}")
        #     logger.print(f"Distance mtx sorted indices: {dist_mtx_idx.shape}, min/max=({dist_mtx_idx.min():.2f}, {dist_mtx_idx.max():.2f})")

        # # load load_emb_info
        # if load_emb_info:
        #     in_path = os.path.join(in_dir, "embeddings.json")
        #     j = json.load(open(in_path, "r"))
        #     elements = np.array(j["elements"]).astype(int)
        #     embeddings = np.array(j["embeddings"]).astype(int)
        #     emb_surf = np.array(j["embedded_surface"]).astype(int)
        #     not_emb_surf = np.array(j["nonembedded_surface"]).astype(int)

        #     emb_surf = torch.from_numpy(emb_surf).long()
        #     not_emb_surf = torch.from_numpy(not_emb_surf).long()

        #     emb_tets = torch.from_numpy(elements[embeddings]).long()
        #     emb_weights = torch.from_numpy(np.array(j["weights"])).float()[None, :, :, None]
        #     self.static_data["emb_tets"] = emb_tets
        #     self.static_data["emb_weights"] = emb_weights
        #     self.static_data["emb_surf"] = emb_surf
        #     self.static_data["not_emb_surf"] = not_emb_surf

        self.static_data["n_ldx"] = self.lx_pos.shape[1]
        self.static_data["n_hdx"] = self.hx_pos.shape[1]
        self.static_data["lrestshape"] = lrestshape
        self.static_data["hrestshape"] = hrestshape
        self.static_data["hsigma"] = hsigma
        self.static_data["hfaces"] = hfaces
        self.static_data["lfaces"] = lfaces
        self.static_data["hmu"] = hmu
        self.static_data["hsigma"] = hsigma

    def __getitem__(self, index):
        frame = self.frames[index]
        seq_frame = self.seq_frames[index]
        seq_name = self.seq_names[index]
        lx_pos = self.lx_pos[index]
        hx_pos = self.hx_pos[index]

        out = {"frame": frame, "seq_frame": seq_frame, "seq_name": seq_name, "lx_pos": lx_pos, "hx_pos": hx_pos}
            
        return out

    def __len__(self):
        return len(self.frames)