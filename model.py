import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
import pointnet2.pytorch_utils as pt_utils

class PUNet(nn.Module):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        npoints = [
            npoint, 
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        ## for 4 downsample layers
        in_ch = 0 if not use_normal else 3
        self.SA_modules = nn.ModuleList()
        for k in range(len(npoints)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k],
                    use_xyz=True,
                    bn=use_bn))
            in_ch = mlps[k][-1]

        ## upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[k + 1][-1], 64], 
                    bn=use_bn))
        
        ## feature Expansion
        in_ch = len(npoints) * 64 + 3 # 4 layers + input xyz
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_bn))

        ## coordinate reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)) 


    def forward(self, points):
        ## points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() \
            if self.use_normal else None

        ## downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        ## upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[k](xyz, l_xyz[k + 2], None, l_feats[k + 2])
            up_feats.append(upk_feats)

        ## aggregation
        # [xyz, l0, l1, l2, l3]
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1) # bs, mid_ch, N, 1

        ## expansion
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1

        ## reconstruction
        output = self.pcd_layer(r_feats) # bs, 3, r * N, 1
        return output.squeeze(-1).transpose(1, 2).contiguous() # bs, 3, r * N


if __name__ == '__main__':
    model = PUNet(up_ratio=2, use_normal=True).cuda()
    points = torch.randn([1, 1024, 6]).float().cuda()
    output = model(points)
    print(output.shape)