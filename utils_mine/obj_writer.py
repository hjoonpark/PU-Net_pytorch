import os, glob
import numpy as np
import time
import argparse
# def tet2face(elements):
#     fs = []
#     dup_check = set()
#     face_start_idx_count = {}
#     for e in elements:
#         f0 = e[0]
#         if f0 not in face_start_idx_count:
#             face_start_idx_count[f0] = 1
#         else:
#             face_start_idx_count[f0] += 1
        
#         if face_start_idx_count[f0] == 1:
#             tri1 = [e[0], e[2], e[1]]
#         elif face_start_idx_count[f0] == 2:
#             tri1 = [e[0], e[1], e[3]]
#         elif face_start_idx_count[f0] == 3:
#             tri1 = [e[0], e[1], e[3]]
#         elif face_start_idx_count[f0] == 4:
#             tri1 = [e[0], e[3], e[2]]
#         elif face_start_idx_count[f0] == 5:
#             tri1 = [e[0], e[3], e[2]]
#         elif face_start_idx_count[f0] == 6:
#             tri1 = [e[0], e[2], e[1]]
#         else:
#             assert False
        
#         tri2 = [e[1], e[2], e[3]]
        
#         s1 = sorted(tri1)
#         s2 = sorted(tri2)
#         k1 = "{}_{}_{}".format(s1[0],s1[1],s1[2])
#         k2 = "{}_{}_{}".format(s2[0],s2[1],s2[2])
#         if k1 not in dup_check:
#             dup_check.add(k1)
#             fs.append(tri1)
        
#         if k2 not in dup_check:
#             dup_check.add(k2)
#             fs.append(tri2)
            
#     return fs

def tet2face(elements):
    fs = []
    for e in elements:
        tri1 = [e[0], e[1], e[3]]
        tri2 = [e[1], e[2], e[3]]
        fs.append(tri1)
        fs.append(tri2)
    return fs

def write_obj(save_path, x, elements):
    """
    x: (w*h*d, 3)
    elements: (N, 4)
    """
    fs = tet2face(elements)
    # fs = list of 3 vertex indices forming a triangle
    with open(save_path, "w+") as file:
        for i, xi in enumerate(x):
            file.write("v {} {} {}\n".format(xi[0], xi[1], xi[2]))

        for fidx, f in enumerate(fs):
            # compute normals
            v1 = x[f[0]]
            v2 = x[f[1]]
            v3 = x[f[2]]
            vn = np.cross(v2-v1, v3-v1)
            vn_norm = np.linalg.norm(vn)
            if vn_norm < 1e-4:
                vn = np.cross(v3-v2, v1-v2)
                vn_norm = np.linalg.norm(vn)
                if vn_norm < 1e-4:
                    vn = np.cross(v1-v3, v2-v3)
                    vn_norm = np.linalg.norm(vn)
            vn /= (vn_norm+1e-12)

            file.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))

        for i, f in enumerate(fs):
            # i = face index
            # f: 3d vector
            file.write("f {}//{} {}//{} {}//{}\n".format(f[0]+1, i, f[1]+1, i , f[2]+1, i))

def write_obj_triangle(save_path, x, faces):
    if x.shape[-1] != 3:
        x = x.transpose(1, 0)
        
    with open(save_path, "w+") as file:
        for xi in x:
            file.write("v {} {} {}\n".format(xi[0], xi[1], xi[2]))

        V = x[np.int32(faces).flatten()].reshape(-1, 3, 3)
        v0 = V[:, 0, :]
        v1 = V[:, 1, :]
        v2 = V[:, 2, :]
        n1 = v1-v0
        n2 = v2-v0
        vns = np.cross(n1, n2, axis=1)
        vn_norm = np.linalg.norm(vns, axis=1)
        normals = vns / vn_norm[:, None]
        for vn in normals:
            file.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))

        for i, f in enumerate(faces):
            # i = face index
            # f: 3d vector
            file.write("f {}//{} {}//{} {}//{}\n".format(f[0]+1, i+1, f[1]+1, i+1 , f[2]+1, i+1))

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