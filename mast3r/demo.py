#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import torch
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl


class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

import numpy as np

def rotmat_to_unitquat_custom(R: np.ndarray, wxyz: bool = False) -> np.ndarray:
    """
    Convert rotation matrix or batch of rotation matrices to unit quaternion(s).
    Adapted from SciPy/roma logic, but using NumPy only.
    
    Args:
        R (np.ndarray): shape (3,3) or (B,3,3).
        wxyz (bool): if True, return quaternions in (w, x, y, z) order 
                     instead of default (x, y, z, w).
    
    Returns:
        quaternions (np.ndarray): shape (B,4) if input is (B,3,3),
                                  shape (4,) if input is (3,3).
                                  By default: (x, y, z, w) order, or 
                                  (w, x, y, z) if wxyz=True.
    """
    # 1) reshape input to (B, 3, 3)
    if R.ndim == 2 and R.shape == (3, 3):
        # single rotation
        R = R[np.newaxis, ...]  # => (1,3,3)
    elif R.ndim == 3 and R.shape[-2:] == (3, 3):
        # multiple rotations (B,3,3)
        pass
    else:
        raise ValueError(f"rotmat_to_unitquat_custom: invalid shape {R.shape}, expected (3,3) or (B,3,3)")
    
    B = R.shape[0]  # batch size

    # diagonal elements
    d0 = R[:, 0, 0]
    d1 = R[:, 1, 1]
    d2 = R[:, 2, 2]
    d3 = d0 + d1 + d2  # trace

    # pick max among d0, d1, d2, d3
    decision_matrix = np.stack([d0, d1, d2, d3], axis=1)  # shape (B,4)
    choices = np.argmax(decision_matrix, axis=1)         # shape (B,)

    quat = np.empty((B, 4), dtype=R.dtype)

    # case 1: trace is max
    trace_mask = (choices == 3)
    idx_trace = np.nonzero(trace_mask)[0]
    # x = (R[2,1] - R[1,2]) / (4w) etc., but adapted from the direct approach
    # see original code or SciPy approach
    # Here we do direct assignment:
    # x = R[2,1] - R[1,2]
    # y = R[0,2] - R[2,0]
    # z = R[1,0] - R[0,1]
    # w = 1 + trace
    quat[idx_trace, 0] = R[idx_trace, 2, 1] - R[idx_trace, 1, 2]  # x
    quat[idx_trace, 1] = R[idx_trace, 0, 2] - R[idx_trace, 2, 0]  # y
    quat[idx_trace, 2] = R[idx_trace, 1, 0] - R[idx_trace, 0, 1]  # z
    quat[idx_trace, 3] = 1.0 + d3[idx_trace]                      # w

    # case 2: one of the diagonals R[i,i] is max
    idx_not_trace = np.nonzero(choices != 3)[0]
    if idx_not_trace.size > 0:
        i = choices[idx_not_trace]
        j = (i + 1) % 3
        k = (j + 1) % 3

        # fill: x,y,z,w => using formula from SciPy/roma
        # quat[idx, i] = 1 - trace + 2 * R[i,i]
        # quat[idx, j] = R[j,i] + R[i,j]
        # quat[idx, k] = R[k,i] + R[i,k]
        # quat[idx, 3] = R[k,j] - R[j,k]
        # but we do index-based approach
        quat[idx_not_trace, i] = 1.0 - d3[idx_not_trace] + 2.0 * R[idx_not_trace, i, i]
        quat[idx_not_trace, j] = R[idx_not_trace, j, i] + R[idx_not_trace, i, j]
        quat[idx_not_trace, k] = R[idx_not_trace, k, i] + R[idx_not_trace, i, k]
        quat[idx_not_trace, 3] = R[idx_not_trace, k, j] - R[idx_not_trace, j, k]

    # normalize each quaternion
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    quat /= norms

    # if wxyz => rearrange to (w,x,y,z)
    if wxyz:
        # currently (x,y,z,w)
        # reorder => (w,x,y,z)
        x = quat[:, 0].copy()
        y = quat[:, 1].copy()
        z = quat[:, 2].copy()
        w = quat[:, 3].copy()
        quat[:, 0] = w
        quat[:, 1] = x
        quat[:, 2] = y
        quat[:, 3] = z

    # if single rotation => return shape (4,)
    if B == 1:
        return quat[0]
    return quat


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true', default=True)
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.add_argument('--data_path', default="/workspace/data/jeonghonoh/dataset/dynamic/dual_arm/seq_01", type=str)
    
    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser

def align_convert_dual_scene_to_ply(scene_state, outdir: str, seq: str, name: str, clean_depth=True, min_conf_thr=0.2):
    if scene_state is None:
        return None
    export_dir = os.path.join(outdir, seq)
    os.makedirs(export_dir, exist_ok=True)
    export_dir = os.path.join(export_dir, 'pcd')
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"{name}.ply")
    scene = scene_state.sparse_ga
    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    
    export_scene = trimesh.Scene()
    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, msk)]).reshape(-1, 3)
    col = np.concatenate([p[m] for p, m in zip(scene.imgs, msk)]).reshape(-1, 3)
    valid_msk = np.isfinite(pts.sum(axis=1))
    pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
    export_scene.add_geometry(pct)
    geom = export_scene.geometry["geometry_0"]
    geom.export(export_path)
    print('(exporting 3D scene to', export_path, ')')
    
    masking_option = True
    #export each geometry
    srt = int(name.split('_')[1])
    end = int(name.split('_')[2])
    idx = 0
    for i in range(srt, end+1):
        name = f"align_{i}"
        export_path = os.path.join(export_dir, f"{name}.ply")
        if masking_option:
            pts_l = pts3d[idx][msk[idx].ravel()].reshape(-1, 3)
            pts_r = pts3d[idx+1][msk[idx+1].ravel()].reshape(-1, 3)
            col_l = scene.imgs[idx][msk[idx]].reshape(-1, 3)
            col_r = scene.imgs[idx+1][msk[idx+1]].reshape(-1, 3)
            valid_msk_l = np.isfinite(pts_l.sum(axis=1))
            valid_msk_r = np.isfinite(pts_r.sum(axis=1))
            pts = np.concatenate([pts_l[valid_msk_l], pts_r[valid_msk_r]])
            col = np.concatenate([col_l[valid_msk_l], col_r[valid_msk_r]])
            valid_msk = np.concatenate([valid_msk_l, valid_msk_r])
            pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        else:
            pts_l = pts3d[idx].reshape(-1, 3)
            pts_r = pts3d[idx+1].reshape(-1, 3)
            col_l = scene.imgs[idx].reshape(-1, 3)
            col_r = scene.imgs[idx+1].reshape(-1, 3)
            valid_msk_l = np.isfinite(pts_l.sum(axis=1))
            valid_msk_r = np.isfinite(pts_r.sum(axis=1))
            pts = np.concatenate([pts_l[valid_msk_l], pts_r[valid_msk_r]])
            col = np.concatenate([col_l[valid_msk_l], col_r[valid_msk_r]])
            valid_msk = np.concatenate([valid_msk_l, valid_msk_r])
            pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        export_scene = trimesh.Scene()
        export_scene.add_geometry(pct)
        geom = export_scene.geometry["geometry_0"]
        geom.export(export_path)
        print('(exporting 3D scene to', export_path, ')')
        idx += 2

    return

def convert_dual_scene_to_ply(scene_state, outdir: str, seq: str, name: str, clean_depth=True, min_conf_thr=0.2):
    if scene_state is None:
        return None
    export_dir = os.path.join(outdir, seq)
    os.makedirs(export_dir, exist_ok=True)
    export_dir = os.path.join(export_dir, 'pcd')
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f"{name}.ply")
    export_obj_path = os.path.join(export_dir, f"{name}.obj")
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    
    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    
    return _convert_dual_scene_to_ply(imgs = rgbimg, pts3d = pts3d, mask = msk, focals = focals, cams2world = cams2world, export_path = export_path, export_obj_path = export_obj_path)

def _convert_dual_scene_to_ply(imgs, pts3d, mask, focals, cams2world, export_path: str, export_obj_path: str, clean_depth=True, min_conf_thr=0.2):
    # assert len(pts3d) == 2
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
    valid_msk = np.isfinite(pts.sum(axis=1))
    # breakpoint()
    pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
    scene.add_geometry(pct)
    
    # add each camera: 생략
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    # scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # mesh = scene.geometry # OrderedDict([('geometry_0', <trimesh.PointCloud(vertices.shape=(272846, 3))>)])
    geom = scene.geometry["geometry_0"]
    geom.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    geom.export(export_path)
    print('(exporting 3D scene to', export_path, ')')
    # mesh['geometry_0'].export(export_path)
    # scene.export(export_path, file_type='ply')
    # scene.export(export_obj_path, file_type='obj')
    # print('(exporting 3D scene to', export_obj_path, ')')
    # breakpoint()
    
    return export_path

def convert_multiple_dual_scene_to_ply(scene_state_lst: list, outdir: str, seq: str, clean_depth=True, min_conf_thr=0.2):
    export_dir = os.path.join(outdir, seq, 'pcd')
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, 'multi.ply')
    
    all_pts = []  # 모든 포인트 (각각 (N,3))
    all_cols = []  # 모든 컬러 (각각 (N,3))
    cam_to_worlds = []  # 모든 카메라 위치 (각각 (4,4))
    scene_out = trimesh.Scene()
    
    for scene_state in scene_state_lst:
        scene = scene_state.sparse_ga
        rgbimg = scene.imgs
        cam_to_worlds.append(scene.get_im_poses().cpu())
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
        mask = to_numpy([c > min_conf_thr for c in confs])
        # breakpoint()
        # pts3d와 rgbimg는 두 요소짜리 리스트라고 가정 (예: [pts_left, pts_right], [img_left, img_right])
        pts_combined_list = []
        col_combined_list = []
        for p, m, im in zip(pts3d, mask, rgbimg):
            # m은 2D boolean mask (이미지 해상도와 동일)
            # p[m.ravel()]는 p의 1D 인덱스로 valid 점들을 선택
            pts_valid = p[m.ravel()]
            # im[m]는 im의 해당 mask 위치의 컬러 값을 선택 (im은 (H, W, 3))
            col_valid = im[m]
            pts_combined_list.append(pts_valid.reshape(-1, 3))
            col_combined_list.append(col_valid.reshape(-1, 3))
        
        pts_merged = np.concatenate(pts_combined_list, axis=0)  # (N_left + N_right, 3)
        col_merged = np.concatenate(col_combined_list, axis=0)  # (N_left + N_right, 3)

        all_pts.append(pts_merged)
        all_cols.append(col_merged)
    
    every_combine = True
    if every_combine:    
        pts_combined = np.concatenate(all_pts, axis=0)
        cols_combined = np.concatenate(all_cols, axis=0)
        valid_msk = np.isfinite(pts_combined.sum(axis=1))    
        pct = trimesh.PointCloud(pts_combined[valid_msk], colors=cols_combined[valid_msk])
        pct.export(export_path)
        print('(exporting combined point cloud to', export_path, ')')
    else:
        for idx, (pts, cols) in enumerate(zip(all_pts, all_cols)):
            valid_msk = np.isfinite(pts.sum(axis=1))
            pct = trimesh.PointCloud(pts[valid_msk], colors=cols[valid_msk])
            #여기서 pct 여러개를 한 파일로 저장
            pct.apply_transform(np.linalg.inv(cam_to_worlds[idx][0] @ OPENGL @ np.eye(4)))
            goem_name = f'geometry_{idx}'
            scene_out.add_geometry(pct, geom_name=goem_name)
        scene_out.export(export_path)
        print('(exporting multiple point clouds to', export_path, ')')
        breakpoint()
    return export_path


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    # breakpoint()
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
                            filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    # breakpoint()
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    # breakpoint()
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    # breakpoint()
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    # breakpoint()
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    return scene_state, outfile


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    show_win_controls = scenegraph_type in ["swin", "logwin"]
    show_winsize = scenegraph_type in ["swin", "logwin"]
    show_cyclic = scenegraph_type in ["swin", "logwin"]
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                            minimum=min_winsize, maximum=max_winsize, step=1, visible=show_winsize)
    win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=show_cyclic)
    win_col = gradio.Column(visible=show_win_controls)
    refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=scenegraph_type == 'oneref')
    return win_col, winsize, win_cyclic, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False,
              share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model, device,
                                  silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Number(value=500, precision=0, minimum=0, maximum=10_000,
                                               label="num_iterations", info="For coarse alignment!")
                        lr2 = gradio.Slider(label="Fine LR", value=0.014, minimum=0.005, maximum=0.05, step=0.001)
                        niter2 = gradio.Number(value=200, precision=0, minimum=0, maximum=100_000,
                                               label="num_iterations", info="For refinement!")
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine+depth', label="OptLevel",
                                                      info="Optimization level")
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=5.,
                                                          minimum=0., maximum=30., step=0.1,
                                                          info="Before Fallback to Regr3D!")
                        shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                            info="Only optimize one set of intrinsics for all views")
                        scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                           ("swin: sliding window", "swin"),
                                                           ("logwin: sliding window with long range", "logwin"),
                                                           ("oneref: match one image with all", "oneref")],
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as win_col:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                        refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                              minimum=0, maximum=0, step=1, visible=False)
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
                TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[win_col, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
                          outputs=[scene, outmodel])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                    outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)

import json
import yaml
import PIL.Image
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose
from dust3r.utils.image import _resize_pil_image

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_stereo_data(folder_path, srt_frame, end_frame, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_path, str):
        if verbose:
            print(f'>> Loading images from {folder_path}')
        # root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
        # left_root, right_root, left_content, right_content = \
        #     os.path.join(folder_or_list, "left"), os.path.join(folder_or_list, "right"), sorted(os.listdir(os.path.join(folder_or_list, "left"))), sorted(os.listdir(os.path.join(folder_or_list, "right")))
        left_calib_path, right_calib_path = os.path.join(folder_path, "calibration_03.yaml"), os.path.join(folder_path, "calibration_02.yaml")
        metadata_path = os.path.join(folder_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
    else:
        raise ValueError(f'bad {folder_path=} ({type(folder_path)})')

    #load calibration data, calibration_03.yaml
    with open(left_calib_path, 'r') as f:
        left_calib_data = yaml.safe_load(f)
    left_intrinsic = np.array([[left_calib_data['fx'], left_calib_data['skew'], left_calib_data['cx']],
                               [0, left_calib_data['fy'], left_calib_data['cy']],
                               [0, 0, 1]])
    with open(right_calib_path, 'r') as f:
        right_calib_data = yaml.safe_load(f)
    right_intrinsic = np.array([[right_calib_data['fx'], right_calib_data['skew'], right_calib_data['cx']],
                                [0, right_calib_data['fy'], right_calib_data['cy']],
                                [0, 0, 1]])
    if verbose:
        print(f'Loaded calibration data from {left_calib_path} and {right_calib_path}')
        print(f'Left intrinsic matrix: {left_intrinsic}')
        print(f'Right intrinsic matrix: {right_intrinsic}')
    
    samples = []
    cnt = 0
    debug_mode = False
    for entry in metadata:
        if debug_mode and cnt == 3:
            break

        if entry.get("sample_number", None) is not None:
            #srt_frame ~ end_frame
            if entry.get("sample_number", None) < srt_frame or entry.get("sample_number", None) > end_frame:
                continue
            else:
                cnt += 1
        # else:
        #     print(f"entry.get('sample_number', None): {entry.get('sample_number', None)}")

        # Extract paths from metadata
        left_img_rel = entry["image_left"]
        right_img_rel = entry["image_right"]
        left_depth_rel = entry["depth_left"]
        right_depth_rel = entry["depth_right"]
        left_pose_rel = entry["pose_left"]
        right_pose_rel = entry["pose_right"]
    
        # Build full paths
        left_img_path = os.path.join(folder_path, left_img_rel)
        right_img_path = os.path.join(folder_path, right_img_rel)
        left_depth_path = os.path.join(folder_path, left_depth_rel)
        right_depth_path = os.path.join(folder_path, right_depth_rel)
        left_pose_path = os.path.join(folder_path, left_pose_rel)
        right_pose_path = os.path.join(folder_path, right_pose_rel)
        
        # Load left/right images with PIL
        left_img = PIL.Image.open(left_img_path).convert('RGB')
        right_img = PIL.Image.open(right_img_path).convert('RGB')
        assert left_img.size == right_img.size, f'original: left and right images must have the same size'
        
        W_origin, H_origin = left_img.size
        
        if size == 512:
            # resize long side to 512
            left_img = _resize_pil_image(left_img, size)
            right_img = _resize_pil_image(right_img, size)
        else:
            print(f"Unsupported size: {size}")
            return None
        assert left_img.size == right_img.size, f'modified: left and right images must have the same size'
        
        W_resize, H_resize = left_img.size
        cx, cy = W_resize//2, H_resize//2
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        left_img = left_img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        right_img = right_img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        assert left_img.size == right_img.size, f'cropped: left and right images must have the same size'
        
        W, H = left_img.size
        # if verbose and entry.get("sample_number", None) % 100 == 1:
        if verbose:
            print(f' - adding {left_img_rel} and {right_img_rel} with resolution {W_origin}x{H_origin} --> {W_resize}x{H_resize} --> {W}x{H}')
        
        #intrinsic matrix for modified images
        #original -> resize
        scale_w = W_resize / W_origin
        scale_h = H_resize / H_origin
        left_intrinsic_modified = left_intrinsic.copy()
        right_intrinsic_modified = right_intrinsic.copy()
        left_intrinsic_modified[0, 0] *= scale_w
        left_intrinsic_modified[0, 2] *= scale_w
        left_intrinsic_modified[1, 1] *= scale_h
        left_intrinsic_modified[1, 2] *= scale_h
        left_intrinsic_modified[0, 1] *= scale_w #skew
        right_intrinsic_modified[0, 0] *= scale_w
        right_intrinsic_modified[0, 2] *= scale_w
        right_intrinsic_modified[1, 1] *= scale_h
        right_intrinsic_modified[1, 2] *= scale_h
        right_intrinsic_modified[0, 1] *= scale_w
        #resize -> crop
        crop_w = cx - halfw
        crop_h = cy - halfh
        left_intrinsic_modified[0, 2] -= crop_w
        left_intrinsic_modified[1, 2] -= crop_h
        right_intrinsic_modified[0, 2] -= crop_w
        right_intrinsic_modified[1, 2] -= crop_h
        
        # Load depth images (assuming they are also image files)
        # If the depth is stored in 16-bit PNG or similar, you may need to handle that differently
        left_depth_img = PIL.Image.open(left_depth_path)
        right_depth_img = PIL.Image.open(right_depth_path)
        
        # Convert depth to np.array for easier handling
        left_depth = np.array(left_depth_img)
        right_depth = np.array(right_depth_img)
        
        # Load pose file and convert its content to a numpy array
        with open(left_pose_path, "r") as f:
            # Read all lines from the file
            lines = f.readlines()
            # Convert each line into a list of floats and build a 2D numpy array
            left_pose = np.array([list(map(float, line.split())) for line in lines])

        with open(right_pose_path, "r") as f:
            lines = f.readlines()
            right_pose = np.array([list(map(float, line.split())) for line in lines])
            
        # Collect all data in a dictionary
        sample = {
            "sample_number": entry.get("sample_number", None),
            "timestamp": entry.get("timestamp", None),
            "left_img_path": left_img_path, # str
            "right_img_path": right_img_path, # str
            "left_img": left_img,           # PIL Image
            "right_img": right_img,         # PIL Image
            "left_depth": left_depth,   # numpy array
            "right_depth": right_depth, # numpy array
            "left_pose": left_pose,    # numpy array
            "right_pose": right_pose,  # numpy array
            "left_intrinsic": left_intrinsic_modified, # numpy array
            "right_intrinsic": right_intrinsic_modified, # numpy array
            "image_left_timestamp": entry.get("image_left_timestamp", None),
            "image_right_timestamp": entry.get("image_right_timestamp", None),
            "depth_left_timestamp": entry.get("depth_left_timestamp", None),
            "depth_right_timestamp": entry.get("depth_right_timestamp", None),
            "pose_left_timestamp": entry.get("pose_left_timestamp", None),
            "pose_right_timestamp": entry.get("pose_right_timestamp", None),
            "max_sync_error": entry.get("max_sync_error", None)
        }
        
        samples.append(sample)

        # if verbose and entry.get("sample_number", None) % 100 == 1:
        if verbose:
            print(f"Loaded sample ~{entry.get('sample_number', '?')}")
            print(f"modified left intrinsic matrix: {left_intrinsic_modified}")
            print(f"modified right intrinsic matrix: {right_intrinsic_modified}")

    if verbose:
        print(f"Total samples loaded: {len(samples)}")
    return samples 
    

def make_stereo_pairs(samples, symmetrize=True):
    filelist_total = []
    pairs_total = []

    
    for sample in samples:
        filelist = []
        imgs = []
        # {'img': tensor형식, 'true_shape': array([[288, 512]], dtype=int32), 'idx': 1, 'instance': '1'}
        #imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
                # [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        pairs = [] #tuple의 리스트
        left_img = sample["left_img"]
        right_img = sample["right_img"]
        left_dict = dict(img=ImgNorm(left_img)[None], true_shape=np.int32([left_img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)))
        imgs.append(left_dict)
        right_dict = dict(img=ImgNorm(right_img)[None], true_shape=np.int32([right_img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)))
        imgs.append(right_dict)

        filelist.append(sample["left_img_path"])
        filelist.append(sample["right_img_path"])
        
        pairs.append((left_dict, right_dict))
        
        if symmetrize:
            pairs.append((right_dict, left_dict))

        filelist_total.append(filelist)
        pairs_total.append(pairs)

    return filelist_total, pairs_total

def align_mode_stereo_pairs(samples, symmetrize=True):
    ret_filelist = []
    ret_pairs = []
    instance_idx = 0
    for sample in samples:
        ret_filelist.append(sample["left_img_path"])
        ret_filelist.append(sample["right_img_path"])
        left_img = sample["left_img"]
        right_img = sample["right_img"]
        left_dict = dict(img=ImgNorm(left_img)[None], true_shape=np.int32([left_img.size[::-1]]), idx=instance_idx, instance=str(instance_idx))
        instance_idx += 1
        right_dict = dict(img=ImgNorm(right_img)[None], true_shape=np.int32([right_img.size[::-1]]), idx=instance_idx, instance=str(instance_idx))
        instance_idx += 1
        ret_pairs.append((left_dict, right_dict))
        ret_pairs.append((right_dict, left_dict))
    
    prev_left = None
    for i in range(0, len(ret_pairs), 2):
        left, _ = ret_pairs[i]
        if prev_left is not None:
            ret_pairs.append((prev_left, left))
            ret_pairs.append((left, prev_left))
        prev_left = left
    
    return ret_filelist, ret_pairs


#raft_ws  
import cv2
from tqdm import tqdm
from third_party.raft import load_RAFT
from dust3r.utils.geom_opt import OccMask, DepthBasedWarping
from third_party.RAFT.core.utils.flow_viz import flow_to_image
def make_flow_image_lists(samples):
    """
    Given a list of samples (from load_stereo_data), extract left and right images
    as numpy arrays suitable for optical flow computation.
    
    Each sample is expected to have the following keys:
      - "left_img": PIL Image for left view.
      - "right_img": PIL Image for right view.
    
    Returns:
      left_imgs: list of numpy arrays, each of shape (H, W, 3)
      right_imgs: list of numpy arrays, each of shape (H, W, 3)
    """
    left_imgs = []
    right_imgs = []
    
    for sample in samples:
        # Convert PIL images to numpy arrays
        left_img_np = np.array(sample["left_img"])
        right_img_np = np.array(sample["right_img"])
        left_imgs.append(left_img_np)
        right_imgs.append(right_img_np)
    
    return left_imgs, right_imgs

def get_flow(left_imgs: list, right_imgs: list, model_path="third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"):
    """
    Compute optical flow for two sequences of images:
      - left_imgs: list of left image numpy arrays, shape (H, W, C)
      - right_imgs: list of right image numpy arrays, shape (H, W, C)
    
    For each sequence, optical flow is computed between consecutive frames
    (both forward: frame i -> frame i+1, and backward: frame i+1 -> frame i).
    
    Returns:
      A tuple: (left_flow_forward, left_flow_backward, left_valid_mask),
               (right_flow_forward, right_flow_backward, right_valid_mask)
    """
    print('Precomputing flow...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_valid_flow_mask = OccMask(th=3.0)
    
    # Convert lists of images (numpy arrays) into tensors with shape (N, H, W, C)
    left_tensor = torch.tensor(np.stack(left_imgs)).float().to(device)
    right_tensor = torch.tensor(np.stack(right_imgs)).float().to(device)
    
    # Load RAFT model using the given model_path
    flow_net = load_RAFT(model_path)
    flow_net = flow_net.to(device)
    flow_net.eval()
    
    def compute_flow_seq(img_seq : torch.Tensor): # (N, H, W, C)
        """ Compute forward and backward flow between consecutive frames in a sequence. """
        flows_forward = []
        flows_backward = []
        num = img_seq.shape[0]
        chunk_size = 12
        for i in tqdm(range(0, num - 1, chunk_size), desc="Processing sequence"):
            end_idx = min(i + chunk_size, num - 1)
            # Prepare source (frame i) and target (frame i+1) frames
            src = img_seq[i:end_idx].permute(0, 3, 1, 2) * 255  # (B, C, H, W)
            tgt = img_seq[i+1:end_idx+1].permute(0, 3, 1, 2) * 255
            # Compute forward flow: src -> tgt
            flow_forward = flow_net(src, tgt, iters=20, test_mode=True)[1]
            # Compute backward flow: tgt -> src
            flow_backward = flow_net(tgt, src, iters=20, test_mode=True)[1]
            flows_forward.append(flow_forward)
            flows_backward.append(flow_backward)
        if flows_forward:
            flows_forward = torch.cat(flows_forward, dim=0)
            flows_backward = torch.cat(flows_backward, dim=0)
        else:
            flows_forward = torch.empty(0)
            flows_backward = torch.empty(0)
        return flows_forward, flows_backward

    with torch.no_grad():
        # Compute flows for left and right sequences separately
        left_flow_forward, left_flow_backward = compute_flow_seq(left_tensor)
        right_flow_forward, right_flow_backward = compute_flow_seq(right_tensor)
        
        # Compute valid masks for each sequence using OccMask
        left_valid_mask = get_valid_flow_mask(left_flow_forward, left_flow_backward)
        right_valid_mask = get_valid_flow_mask(right_flow_forward, right_flow_backward)
    
    print('Flow precomputed.')
    if flow_net is not None:
        del flow_net
    return (left_flow_forward, left_flow_backward, left_valid_mask), (right_flow_forward, right_flow_backward, right_valid_mask)


def apply_transform_pts3d(pts3d: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    pts3d: (H, W, 3) 형태 [view coordinate]
    transform: (4, 4) 형태의 SE3 변환 행렬
    return: (H, W, 3) 형태 [world coordinate]
    """
    device = pts3d.device
    dtype = pts3d.dtype

    H, W, _ = pts3d.shape
    # (H, W, 3)을 (N, 3)으로 펼치기
    coords = pts3d.view(-1, 3)

    # 좌표를 homogeneous (N, 4)로 만들어 transform 적용
    ones = torch.ones((coords.shape[0], 1), device=device, dtype=dtype)
    coords_hom = torch.cat([coords, ones], dim=-1)  # (N, 4)

    # transform^T 곱 (행렬 곱)
    coords_world_hom = coords_hom @ transform.T  # (N, 4)

    # w 분할 (透視投影 등 고려, 여기서는 보통 w=1로 유지)
    w = coords_world_hom[..., 3].clamp_min(1e-10)
    coords_world = coords_world_hom[..., :3] / w.unsqueeze(-1)  # (N, 3)

    # 다시 (H, W, 3)로 reshape
    coords_world = coords_world.view(H, W, 3)
    return coords_world



def find_dynamic_mask_using_pts3d(
    pts3d_1: torch.Tensor, 
    pts3d_2: torch.Tensor, 
    flow_12: torch.Tensor, 
    flow_21: torch.Tensor, 
    valid: torch.Tensor, 
    transform_1: torch.Tensor,
    transform_2: torch.Tensor,
    thres=0.1,
    normalize=True,
    normalize_thres = 1.5):
        """
        find the dynamic points in the 3d space using the flow as dense correspondence
        
        두 이미지(view1, view2)에 대한 3D 포인트 맵(pts3d_1, pts3d_2)과
        view1 -> view2 방향 흐름(flow_12), view2 -> view1 방향 흐름(flow_21)을 이용해
        일정값 이상 차이가 나는 지점을 dynamic으로 간주하여 마스크를 반환한다.

        Args:
            pts3d_1 (torch.Tensor): (H, W, 3) 형태, 첫 번째 뷰의 3D 포인트 맵
            pts3d_2 (torch.Tensor): (H, W, 3) 형태, 두 번째 뷰의 3D 포인트 맵
            flow_12 (torch.Tensor): (2, H, W) 형태, view1 -> view2의 흐름
            flow_21 (torch.Tensor): (2, H, W) 형태, view2 -> view1의 흐름
            valid (torch.Tensor): (1, H, W) 형태, occmask에 의해 판단된 valid mask
            transform_1: (4,4) SE3 행렬. pts3d_1에 적용할 transform
            transform_2: (4,4) SE3 행렬. pts3d_2에 적용할 transform
            thres (float): 동적 포인트 판단 임계값
            normalize (bool): True일 경우, 평균과 표준편차를 이용해 동적 포인트 비율을 출력

        Returns:
            dynamic_mask_1 (torch.BoolTensor): (H, W) 형태, view1에서 동적으로 판단된 픽셀 마스크
            dynamic_mask_2 (torch.BoolTensor): (H, W) 형태, view2에서 동적으로 판단된 픽셀 마스크
        """
        assert pts3d_1.shape == pts3d_2.shape, 'shape mismatch(pts3d_1, pts3d_2)'
        assert flow_12.shape == flow_21.shape, 'shape mismatch(flow_12, flow_21)'
        H, W, _ = pts3d_1.shape
        device = flow_12.device
        
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        
        if transform_1 is not None:
            transform_1 = np.linalg.inv(to_numpy(transform_1) @ OPENGL @ rot)
            transform_1 = torch.tensor(transform_1, device=device, dtype=torch.float32)
            pts3d_1 = apply_transform_pts3d(pts3d_1.to(device), transform_1)
        if transform_2 is not None:
            transform_2 = np.linalg.inv(to_numpy(transform_2) @ OPENGL @ rot)
            transform_2 = torch.tensor(transform_2, device=device, dtype=torch.float32)
            pts3d_2 = apply_transform_pts3d(pts3d_2.to(device), transform_2.to(device))
        
        dynamic_mask_1 = torch.zeros((H, W), dtype=torch.bool, device=device)
        dynamic_mask_2 = torch.zeros((H, W), dtype=torch.bool, device=device)
        
        coords_1 = torch.stack(torch.meshgrid(torch.arange(H, device = device), torch.arange(W, device = device)), dim=-1).float()
        moved_coords_1 = torch.round(coords_1 + flow_12.permute(1, 2, 0)).long()
        # breakpoint()
        # only consider the points that are within the image
        valid_mask = ((moved_coords_1[..., 0] >= 0) & (moved_coords_1[..., 0] < H) & (moved_coords_1[..., 1] >= 0) & (moved_coords_1[..., 1] < W))
        # valid_mask가 True인 좌표의 (y, x) 인덱스
        # shape: (N, 2)
        valid_idxs = valid_mask.nonzero(as_tuple=False)
        # moved_coords_1_int에서 valid 영역만 뽑은 것
        valid_moved_coords = moved_coords_1[valid_mask]  # (N, 2)
        # breakpoint()
        # 유효한 좌표들에 대해 3D 포인트를 한 번에 조회
        # pts3d_1_valid: view1에서 valid 픽셀의 3D 포인트
        # pts3d_2_valid: view2에서 valid 픽셀의 3D 포인트(흐름에 따라 이동한 좌표)
        pts3d_1_valid = pts3d_1[valid_idxs[:, 0], valid_idxs[:, 1]]  # shape: (N, 3)
        pts3d_2_valid = pts3d_2[valid_moved_coords[:, 0], valid_moved_coords[:, 1]]  # shape: (N, 3)
        dist = torch.norm(pts3d_2_valid - pts3d_1_valid, dim=-1)
        # breakpoint()
        if normalize:
            mean = dist.mean()
            std = dist.std()
            too_big_mask = (dist > mean + normalize_thres * std)
            print(f'mean: {mean}, std: {std}, dynamic points: {too_big_mask.sum()/dist.size(0) * 100:.2f}%')
        else:
            too_big_mask = (dist > thres)
        
        dynamic_idxs_1 = valid_idxs[too_big_mask]
        dynamic_idxs_2 = valid_moved_coords[too_big_mask]
        
        dynamic_mask_1[dynamic_idxs_1[:, 0], dynamic_idxs_1[:, 1]] = True
        dynamic_mask_2[dynamic_idxs_2[:, 0], dynamic_idxs_2[:, 1]] = True
        
        # tmp_data = []
        # cnt = 0
        # for i in range(H):
        #     for j in range(W):
        #         if moved_coords_1[i, j, 0] >= 0 and moved_coords_1[i, j, 0] < H and moved_coords_1[i, j, 1] >= 0 and moved_coords_1[i, j, 1] < W:
        #             # if torch.norm(pts3d_2[int(moved_coords_1[i, j, 0]), int(moved_coords_1[i, j, 1])] - pts3d_1[i, j]) > thres:
        #             #     dynamic_mask_1[i, j] = True
        #             #     dynamic_mask_2[int(moved_coords_1[i, j, 0]), int(moved_coords_1[i, j, 1])] = True
        #             tmp_data.append(torch.norm(pts3d_2[int(moved_coords_1[i, j, 0]), int(moved_coords_1[i, j, 1])] - pts3d_1[i, j]))
        #         else:
        #             cnt += 1
        
        # breakpoint()
        
        return dynamic_mask_1, dynamic_mask_2

# def rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
#     # Compute the error rotation matrix: R_error = R_pred * R_gt^T
#     R_error = np.dot(R_pred, R_gt.T)
#     # Compute the trace of R_error
#     trace_val = np.trace(R_error)
#     # Ensure numerical stability by clipping the value between -1 and 1
#     cos_angle = np.clip((trace_val - 1) / 2.0, -1.0, 1.0)
#     # Calculate rotation error in radians
#     return np.arccos(cos_angle)

# def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
#     # Compute the Euclidean distance between predicted and ground truth translation vectors
#     return np.linalg.norm(t_pred - t_gt)

def compare_translation_scale(t_pred: np.ndarray, t_gt: np.ndarray) -> bool:
    # Compute the Euclidean norm (scale) of predicted and ground truth translation vectors
    scale_pred = np.linalg.norm(t_pred)
    scale_gt = np.linalg.norm(t_gt)
    
    if np.isclose(scale_pred, scale_gt, atol=0.01): # 1cm
        print("Pred and Ground Truth are similar in scale.")
        return True
    elif scale_pred > scale_gt:
        print("Predicted scale is larger than Ground Truth.")
        return False
    else:
        print("Predicted scale is smaller than Ground Truth.")
        return False

def evaluate_relative_pose_error(relative_pose_pred: np.ndarray, relative_pose_gt: np.ndarray) -> tuple:
    error_transform = np.dot(np.linalg.inv(relative_pose_gt), relative_pose_pred)
    
    # Extract the rotation error and translation error from the error transformation
    R_err = error_transform[:3, :3]
    t_err = error_transform[:3, 3]
    
    # Calculate rotation error using the trace of the rotation error matrix
    trace_val = np.trace(R_err)
    cos_angle = np.clip((trace_val - 1) / 2.0, -1.0, 1.0)
    rot_err = np.arccos(cos_angle)
    
    # Calculate translation error as the norm of the translation error vector
    trans_err = np.linalg.norm(t_err)
    
    compare_translation_scale(relative_pose_gt[:3, 3], relative_pose_pred[:3, 3])
    
    return rot_err, trans_err


# flowformer_ws
from third_party.FlowFormerPlusPlus.visualize_flow import forward_flowformer, build_model
def flowformer_pairs(samples: list):
    """
    Given a list of samples (from load_stereo_data), extract left and right images
    as numpy arrays suitable for optical flow computation.
    
    Each sample is expected to have the following keys:
      - "left_img": PIL Image for left view.
      - "right_img": PIL Image for right view.
    
    Returns:
      left_img_pairs: list of tuple, (img1, img2)
      right_img_pairs: list of tuple, (img1, img2)
    """
    left_img_lst = []
    right_img_lst = []
    left_file_name_lst = []
    right_file_name_lst = []

    for sample in samples:
        left_img_lst.append(sample["left_img"])
        right_img_lst.append(sample["right_img"])
        #sample["left_img_path"]: /workspace/data/jeonghonoh/dataset/dynamic/dual_arm/seq_01/images/left/0070_left.png
        #마지막 파일명만 리스트에 넣기
        left_file_name_lst.append(sample["left_img_path"].split('/')[-1])
        right_file_name_lst.append(sample["right_img_path"].split('/')[-1])
        # breakpoint()

    left_img_pairs = []
    right_img_pairs = []
    left_file_name = []
    right_file_name = []
    # tuple
    for i in range(len(left_img_lst) - 1):
        left_img_pairs.append((left_img_lst[i], left_img_lst[i+1]))
        right_img_pairs.append((right_img_lst[i], right_img_lst[i+1]))
        left_file_name.append((left_file_name_lst[i], left_file_name_lst[i+1]))
        right_file_name.append((right_file_name_lst[i], right_file_name_lst[i+1]))

    return (left_file_name, right_file_name), (left_img_pairs, right_img_pairs)



def get_3d_distance_using_flow(pts3d_1: torch.Tensor, pts3d_2: torch.Tensor, flow: torch.Tensor, forward = True) -> torch.Tensor:
    """
    Compute the 3D distance between two 3D point clouds using the flow as dense correspondence.
    pts3d_1: (H, W, 3) shape, 3D point cloud of view 1
    pts3d_2: (H, W, 3) shape, 3D point cloud of view 2
    flow: (H, W, 2) shape,
        if forward=True : flow from view1 to view2 (1->2)
        if forward=False: flow from view2 to view1 (2->1)
    
    return: (H, W) shape, 3D distance between the two point clouds, coordinate is from view 1
    
    Explanation:
    1) forward=True
       dist[y1,x1] = norm( pts3d_1[y1,x1] - pts3d_2[y2,x2] ),
         where (y2,x2) = round( (y1,x1) + flow[y1,x1] )
       -> dist가 "image1 픽셀" 위치에 기록.

    2) forward=False
       dist[y1,x1] = norm( pts3d_1[y1,x1] - pts3d_2[y2,x2] ),
         where (y2,x2) is the pixel in image2, 
               and flow[y2,x2] tells (y2,x2) -> (y1,x1).
       -> dist가 "image1 픽셀" 위치에 기록.
    """
    H, W, _ = pts3d_1.shape
    device = pts3d_1.device
    dist = torch.zeros(H, W, device=device)
    
    if forward:
        coords_1 = torch.stack(torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device)), dim=-1)
        coords_1_fwd = coords_1 + flow
        coords_1_rounded = coords_1_fwd.round().long() # image1+flow의 좌표 (image2 pixel 좌표)
        
        # in-bound check
        valid_mask = (coords_1_rounded[...,0]>=0)&(coords_1_rounded[...,0]<H)&\
                    (coords_1_rounded[...,1]>=0)&(coords_1_rounded[...,1]<W) #image1의 pixel 좌표가 flow를 따라 이동한 뒤 image 2의 범위 내에 있는지 mask
        valid_idx = valid_mask.nonzero(as_tuple=False)                      #image1의 pixel 좌표가 image 2의 범위 내에 있는 pixel 좌표
        pcd_1_valid = pts3d_1[valid_idx[:,0], valid_idx[:,1]]
        coords2 = coords_1_rounded[valid_idx[:,0], valid_idx[:,1]] #image 2의 대응점 좌표
        pcd_2_valid = pts3d_2[coords2[:,0], coords2[:,1]]
        dist_valid = torch.norm(pcd_2_valid - pcd_1_valid, dim=1)
        
        dist[valid_idx[:,0], valid_idx[:,1]] = dist_valid
    else:
        #flow: flow 2 -> 1
        coords_2 = torch.stack(torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device)), dim=-1)
        coords_2_fwd = coords_2 + flow  # (H,W,2) ; (y2,x2)+(flow2_1) => (y1,x1)
        coords_2_rounded = coords_2_fwd.round().long()
        valid_mask = (
            (coords_2_rounded[...,0] >= 0) & (coords_2_rounded[...,0] < H) &
            (coords_2_rounded[...,1] >= 0) & (coords_2_rounded[...,1] < W)
        )
        valid_idx = valid_mask.nonzero(as_tuple=False)
        # x2 = pts3d_2[y2,x2]
        pcd_2_valid = pts3d_2[valid_idx[:,0], valid_idx[:,1]]
        # coords_1 = (y1,x1)
        coords_1 = coords_2_rounded[valid_idx[:,0], valid_idx[:,1]]  # shape(N,2)
        # x1 = pts3d_1[y1,x1]
        pcd_1_valid = pts3d_1[coords_1[:,0], coords_1[:,1]]

        dist_valid = torch.norm(pcd_1_valid - pcd_2_valid, dim=1)

        # dist를 image1 기준으로 저장 => dist[y1,x1] = distance
        dist[coords_1[:,0], coords_1[:,1]] = dist_valid
    
    cv2.imwrite('dist.png', (dist/dist.max()*255).cpu().numpy().astype(np.uint8))
    breakpoint()    

    return dist



# def warp_to_ego_flow(src_R: torch.Tensor, src_t: torch.Tensor, tgt_R: torch.Tensor, tgt_t: torch.Tensor, src_disp: torch.Tensor, K: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
#     """
#     Args:
#             src_R (FloatTensor): 1x3x3
#             src_t (FloatTensor): 1x3x1
#             tgt_R (FloatTensor): Nx3x3
#             tgt_t (FloatTensor): Nx3x1
#             src_disp (FloatTensor): Nx1XHxW
#             src_K (FloatTensor): 1x3x3
#     """
#     _, _, H, W = src_disp.shape
#     B = tgt_R.shape[0]
#     device = src_disp.device
#     if not hasattr(self, "coord"):
#         self.generate_grid(H, W, device=device)
#     else:
#         if self.coord.shape[-1] != H * W:
#             del self.coord
#             self.generate_grid(H, W, device=device)
#     # if self.jitted_warp_by_disp is None:
#     # self.jitted_warp_by_disp = torch.jit.trace(
#     #     warp_by_disp, (src_R.detach(), src_t.detach(), tgt_R.detach(), tgt_t.detach(), K, src_disp.detach(), self.coord, inv_K))

#     return warp_by_disp(src_R, src_t, tgt_R, tgt_t, K, src_disp, self.coord, inv_K, debug_mode, use_depth)


def decouple_adjacent_pcds(left_pcds_1: torch.Tensor, right_pcds_1: torch.Tensor, #torch.Size([147456, 3])
                           left_conf_1: torch.Tensor, right_conf_1: torch.Tensor, #torch.Size([288, 512])
                           left_pcds_2: torch.Tensor, right_pcds_2: torch.Tensor,
                           left_conf_2: torch.Tensor, right_conf_2: torch.Tensor,
                           left_fwd_flow: torch.Tensor, right_fwd_flow: torch.Tensor,     #torch.Size([288, 512, 2])
                           left_bwd_flow: torch.Tensor, right_bwd_flow: torch.Tensor,
                           left_pose_1: torch.Tensor, right_pose_1: torch.Tensor,
                           left_pose_2: torch.Tensor, right_pose_2: torch.Tensor,
                           left_depth_1: torch.Tensor, right_depth_1: torch.Tensor,
                           left_depth_2: torch.Tensor, right_depth_2: torch.Tensor,
                           left_focal: int, right_focal: int,
                           alpha = 0.5, threshold = 0.5, space_3d = True) -> tuple:
    '''
    output: dynamic_pcds, static_pcds, frame 1 기준 pcds decoupling
    '''
    device = left_pcds_1.device
    left_pose_1 = left_pose_1.to(device)
    right_pose_1 = right_pose_1.to(device)
    left_pose_2 = left_pose_2.to(device)
    right_pose_2 = right_pose_2.to(device)
    left_K = torch.tensor([[left_focal, 0.0,        256.0],
                           [0.0,        left_focal, 144.0],
                           [0.0,        0.0,        1.0]], device=device)
    right_K = torch.tensor([[right_focal, 0.0,        256.0],
                            [0.0,        right_focal, 144.0],
                            [0.0,        0.0,        1.0]], device=device)
    H, W = left_conf_1.shape
    left_pcds_1 = left_pcds_1.view(H, W, 3)
    right_pcds_1 = right_pcds_1.view(H, W, 3)
    left_pcds_2 = left_pcds_2.view(H, W, 3)
    right_pcds_2 = right_pcds_2.view(H, W, 3)
    left_dynamic_pixels_score = torch.zeros(H, W, dtype=torch.float32, device=left_pcds_1.device)
    right_dynamic_pixels_score = torch.zeros(H, W, dtype=torch.float32, device=right_pcds_1.device)
    
    # algorithm for decoupling dynamic and static pixels
    if space_3d: 
        # step 1: left_1의 (u, v)에서 left_2의 (u+fwd, v_fwd)로 forward flow를 이용해서 3d 점들 사이의 거리를 계산 (dist_1: distance between (x1, y1, z1) and (x2, y2, z2))
        left_dist_1 = get_3d_distance_using_flow(left_pcds_1, left_pcds_2, left_fwd_flow, forward=True)
        right_dist_1 = get_3d_distance_using_flow(right_pcds_1, right_pcds_2, right_fwd_flow, forward=True)

        # step 2: left_2의 (u+fwd, v_fwd)와 left_1의 (u+fwd+bwd, v_fwd+bwd)의 거리를 계산 (dist_2: distance between (x2, y2, z2) and (x3, y3, z3))
        left_dist_2 = get_3d_distance_using_flow(left_pcds_1, left_pcds_2, left_bwd_flow, forward=False)
        right_dist_2 = get_3d_distance_using_flow(right_pcds_1, right_pcds_2, right_bwd_flow, forward=False)
        
        # step 3: (x1, y1, z1)과 (x3, y3, z3)의 거리를 계산 (dist_error: distance between (x1, y1, z1) and (x3, y3, z3))
        
        # step 4: dist_1, dist_2, dist_error를 이용해서 distance score를 계산 (0~1, 클수록 dynamic에 가까움)
        # normalize dist_1, dist_2
        all_dist = torch.cat([left_dist_1.view(-1), left_dist_2.view(-1)], dim=0)
        d_min = all_dist.min()
        d_max = all_dist.max()
        dist_range = d_max - d_min + 1e-6
        left_dist_1 = (left_dist_1 - d_min) / dist_range
        left_dist_2 = (left_dist_2 - d_min) / dist_range
        all_dist = torch.cat([right_dist_1.view(-1), right_dist_2.view(-1)], dim=0)
        d_min = all_dist.min()
        d_max = all_dist.max()
        dist_range = d_max - d_min + 1e-6
        right_dist_1 = (right_dist_1 - d_min) / dist_range
        right_dist_2 = (right_dist_2 - d_min) / dist_range
    else:
        # step 1: ego_flow와 optical flow를 이용해서 dynamic pixel score를 계산
        depth_wrapper = DepthBasedWarping()
        #torch.Size([1, 3, 288, 512])
        ego_flow_1_2, _ = depth_wrapper(src_R=left_pose_1[:3, :3].view(1, 3, 3), src_t=left_pose_1[:3, 3].view(1, 3, 1),
                                        tgt_R=left_pose_2[:3, :3].view(1, 3, 3), tgt_t=left_pose_2[:3, 3].view(1, 3, 1),
                                        src_disp=1/(left_depth_1 + 1e-6).view(1, 1, H, W), K=left_K.view(1, 3, 3), inv_K=torch.linalg.inv(left_K).view(1, 3, 3))
        ego_flow_2_1, _ = depth_wrapper(src_R=left_pose_2[:3, :3].view(1, 3, 3), src_t=left_pose_2[:3, 3].view(1, 3, 1),
                                        tgt_R=left_pose_1[:3, :3].view(1, 3, 3), tgt_t=left_pose_1[:3, 3].view(1, 3, 1),
                                        src_disp=1/(left_depth_2 + 1e-6).view(1, 1, H, W), K=left_K.view(1, 3, 3), inv_K=torch.linalg.inv(left_K).view(1, 3, 3))

        left_dist_1 = torch.norm(ego_flow_1_2[:, :2, :, :] - left_fwd_flow.reshape(1, 2, H, W), dim=1)
        right_dist_1 = torch.norm(ego_flow_2_1[:, :2, :, :] - left_bwd_flow.reshape(1, 2, H, W), dim=1)
        
        ego_flow_1_2, _ = depth_wrapper(src_R=right_pose_1[:3, :3].view(1, 3, 3), src_t=right_pose_1[:3, 3].view(1, 3, 1),
                                        tgt_R=right_pose_2[:3, :3].view(1, 3, 3), tgt_t=right_pose_2[:3, 3].view(1, 3, 1),
                                        src_disp=1/(right_depth_1 + 1e-6).view(1, 1, H, W), K=right_K.view(1, 3, 3), inv_K=torch.linalg.inv(right_K).view(1, 3, 3))
        ego_flow_2_1, _ = depth_wrapper(src_R=right_pose_2[:3, :3].view(1, 3, 3), src_t=right_pose_2[:3, 3].view(1, 3, 1),
                                        tgt_R=right_pose_1[:3, :3].view(1, 3, 3), tgt_t=right_pose_1[:3, 3].view(1, 3, 1),
                                        src_disp=1/(right_depth_2 + 1e-6).view(1, 1, H, W), K=right_K.view(1, 3, 3), inv_K=torch.linalg.inv(right_K).view(1, 3, 3))
        # breakpoint()
        left_dist_2 = torch.norm(ego_flow_1_2[:, :2, :, :] - right_fwd_flow.reshape(1, 2, H, W), dim=1)
        right_dist_2 = torch.norm(ego_flow_2_1[:, :2, :, :] - right_bwd_flow.reshape(1, 2, H, W), dim=1)
        
        #normalize
        left_dist_1 = (left_dist_1 - left_dist_1.amin(dim=(1, 2), keepdim=True)) / (left_dist_1.amax(dim=(1, 2), keepdim=True) - left_dist_1.amin(dim=(1, 2), keepdim=True) + 1e-6)
        right_dist_1 = (right_dist_1 - right_dist_1.amin(dim=(1, 2), keepdim=True)) / (right_dist_1.amax(dim=(1, 2), keepdim=True) - right_dist_1.amin(dim=(1, 2), keepdim=True) + 1e-6)
        left_dist_2 = (left_dist_2 - left_dist_2.amin(dim=(1, 2), keepdim=True)) / (left_dist_2.amax(dim=(1, 2), keepdim=True) - left_dist_2.amin(dim=(1, 2), keepdim=True) + 1e-6)
        right_dist_2 = (right_dist_2 - right_dist_2.amin(dim=(1, 2), keepdim=True)) / (right_dist_2.amax(dim=(1, 2), keepdim=True) - right_dist_2.amin(dim=(1, 2), keepdim=True) + 1e-6)
        cv2.imwrite('left_dist_1.png', (left_dist_1[0] * 255).cpu().numpy().astype(np.uint8))
        cv2.imwrite('right_dist_1.png', (right_dist_1[0] * 255).cpu().numpy().astype(np.uint8))
        cv2.imwrite('left_dist_2.png', (left_dist_2[0] * 255).cpu().numpy().astype(np.uint8))
        cv2.imwrite('right_dist_2.png', (right_dist_2[0] * 255).cpu().numpy().astype(np.uint8))
    # breakpoint()
    # step 5: mast3r의 confidence score는 0~1로 normalize되어있는데 클수록 신뢰도가 높고, static에 가까움
    
    # step 6: confidence score와 distance score를 이용해서 dynamic pixel score를 계산
    
    
    
    # dynamic_pcds, static_pcds
    # left_pcds_1 에서 left_dynamic_pixels > threshold 인 픽셀들만 dynamic_pcds에 추가
    l_dynamic_pcds = left_pcds_1[left_dynamic_pixels_score > threshold].view(-1, 3)
    r_dynamic_pcds = right_pcds_1[right_dynamic_pixels_score > threshold].view(-1, 3)
    dynamic_pcds = torch.cat([l_dynamic_pcds, r_dynamic_pcds], dim=0)
    l_static_pcds = left_pcds_1[left_dynamic_pixels_score <= threshold].view(-1, 3)
    r_static_pcds = right_pcds_1[right_dynamic_pixels_score <= threshold].view(-1, 3)
    static_pcds = torch.cat([l_static_pcds, r_static_pcds], dim=0)
    
    return dynamic_pcds, static_pcds, left_dynamic_pixels_score, right_dynamic_pixels_score #torch.Size([d, 3]), torch.Size([s, 3]), torch.Size([H, W]), torch.Size([H, W])

def decouple_pcds(scene, scene_state, flow_info: dict):
    '''
    flow_into.keys: left_forward_flows, left_backward_flows, right_forward_flows, right_backward_flows, left_valid_masks, right_valid_masks
    
    output: dynamic_pcds: list of pcds, static_pcds: list of pcds
    '''
    
    left_forward_flows = flow_info['left_forward_flows']    # list of torch.Size([288, 512, 2]), len: pair_num - 1
    left_backward_flows = flow_info['left_backward_flows']  # list of torch.Size([288, 512, 2]), len: pair_num - 1
    right_forward_flows = flow_info['right_forward_flows']  
    right_backward_flows = flow_info['right_backward_flows']
    left_valid_masks = flow_info['left_valid_masks']        # list of torch.Size([1, 1, 288, 512]), len: pair_num - 1
    right_valid_masks = flow_info['right_valid_masks']
    
    dynamic_pcds = []
    static_pcds = []
    left_dynamic_pixels_score_lst = []
    right_dynamic_pixels_score_lst = []
    focal = scene.get_focals().cpu()
    pose = scene.get_im_poses().cpu()
    sparse_pts3d = scene.get_sparse_pts3d()
    dense_pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth = True)
    #dense_pts3d: list of torch.Size([147456, 3]), len: pair_num * 2
    #conf: list of torch.Size([288, 512]), len: pair_num * 2
    #pose: torch.Size([pair_num * 2, 4, 4])
    
    pair_num = len(dense_pts3d) // 2
    assert pair_num == (len(left_forward_flows) + 1), 'pair_num mismatch'
    
    #dense_pts3d, conf, flow를 이용해서 dynamic, static pcds 구하기
    for i in range(pair_num - 1):
        left_pts3d_1 = dense_pts3d[i * 2]                               # torch.Size([147456, 3])
        right_pts3d_1 = dense_pts3d[i * 2 + 1]
        left_conf_1 = confs[i * 2]                                      # torch.Size([288, 512])
        right_conf_1 = confs[i * 2 + 1]
        left_pts3d_2 = dense_pts3d[i * 2 + 2]
        right_pts3d_2 = dense_pts3d[i * 2 + 3]
        left_conf_2 = confs[i * 2 + 2]
        right_conf_2 = confs[i * 2 + 3]
        
        #fwd: i -> i+1, 기준좌표계: i
        #bwd: i+1 -> i, 기준좌표계: i+1
        left_forward_flow = left_forward_flows[i]                       # torch.Size([288, 512, 2])
        right_forward_flow = right_forward_flows[i]
        left_backward_flow = left_backward_flows[i]
        right_backward_flow = right_backward_flows[i]
        # left_valid_mask = left_valid_masks[i].squeeze(0).squeeze(0)     # torch.Size([288, 512])
        # right_valid_mask = right_valid_masks[i].squeeze(0).squeeze(0)
        
        left_pose_1 = pose[i * 2]
        right_pose_1 = pose[i * 2 + 1]
        left_pose_2 = pose[i * 2 + 2]
        right_pose_2 = pose[i * 2 + 3]
        
        left_depth_1 = depthmaps[i * 2]
        right_depth_1 = depthmaps[i * 2 + 1]
        left_depth_2 = depthmaps[i * 2 + 2]
        right_depth_2 = depthmaps[i * 2 + 3]
        
        dynamic_pcds_i, static_pcds_i, left_dynamic_pixels_score, right_dynamic_pixels_score = \
            decouple_adjacent_pcds(left_pcds_1 = left_pts3d_1, right_pcds_1 = right_pts3d_1,
                                   left_conf_1 = left_conf_1, right_conf_1 = right_conf_1,
                                   left_pcds_2 = left_pts3d_2, right_pcds_2 = right_pts3d_2,
                                   left_conf_2 = left_conf_2, right_conf_2 = right_conf_2,
                                   left_fwd_flow = left_forward_flow, right_fwd_flow = right_forward_flow,
                                   left_bwd_flow = left_backward_flow, right_bwd_flow = right_backward_flow,
                                   left_pose_1 = left_pose_1, right_pose_1 = right_pose_1,
                                   left_pose_2 = left_pose_2, right_pose_2 = right_pose_2,
                                   left_depth_1 = left_depth_1, right_depth_1 = right_depth_1,
                                   left_depth_2 = left_depth_2, right_depth_2 = right_depth_2,
                                   left_focal = focal[i * 2], right_focal = focal[i * 2 + 1],)
        
        dynamic_pcds.append(dynamic_pcds_i)
        static_pcds.append(static_pcds_i)
        left_dynamic_pixels_score_lst.append(left_dynamic_pixels_score)
        right_dynamic_pixels_score_lst.append(right_dynamic_pixels_score)

    return dynamic_pcds, static_pcds, left_dynamic_pixels_score_lst, right_dynamic_pixels_score_lst


def tmp_load_dynamic_mask(mask_path: str, imsize: tuple, seq_num: int):
    mask_files = sorted(os.listdir(mask_path))
    print("mask_files: ", mask_files)
    dynamic_mask = []
    for mask_file in mask_files:
        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == imsize, 'mask shape mismatch'
        assert seq_num * 2 == len(mask_files), 'mask file num mismatch'
        dynamic_mask.append(mask)
    return dynamic_mask

# def forward_two_images(data_path, outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
#                             filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
#                             as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
#                             win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
def forward_two_images(data_path, srt_frame, end_frame, model, device, image_size = 512, silent = False, seq = 'seq_01'):
    # scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
    # as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    # scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics
    cache_dir = os.path.join('/workspace/data/jeonghonoh/mast3r/', 'cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    optim_level = 'coarse'
    lr1 = 0.07
    niter1 = 500
    lr2 = 0.014
    niter2 = 200
    matching_conf_thr = 5.
    shared_intrinsics = False
    if optim_level == 'coarse':
        niter2 = 0
    
    samples = load_stereo_data(data_path, srt_frame, end_frame, size=image_size, verbose=not silent)
    # breakpoint()
    
    align_mode = True
    if align_mode:
        filelist_total, pairs_total = align_mode_stereo_pairs(samples) #(n + (n-1))*2
    else:
        filelist_total, pairs_total = make_stereo_pairs(samples) #filelist: 이미지 경로 리스트 2n개, pairs: 이미지 쌍의 리스트 n개
    
    # get flow (FlowFormer++)
    flowformer_model = build_model()
    #img_pairs: list of tuple, (PIL Image, PIL Image)
    (ff_left_file_name, ff_right_file_name), (ff_left_img_pairs, ff_right_img_pairs) = flowformer_pairs(samples)
    # breakpoint()
    with torch.no_grad():
        debug_mode = True
        # visualize_flow('.', 'viz_results/', flowformer_model, ff_img_pairs, keep_size = True)
        left_forward_flows, left_backward_flows, right_forward_flows, right_backward_flows, left_names, right_names = forward_flowformer(flowformer_model, ff_left_file_name, ff_right_file_name, ff_left_img_pairs, ff_right_img_pairs, keep_size = True, debug_mode = debug_mode, seq = seq)
        # valid using OccMask
        get_valid_flow_mask = OccMask(th=3.0)
        left_valid_masks = []
        right_valid_masks = []
        for left_forward_flow, left_backward_flow in zip(left_forward_flows, left_backward_flows):
            # (H, W, 2) -> (B, 2, H, W)
            left_valid_mask = get_valid_flow_mask(left_forward_flow.unsqueeze(0).permute(0, 3, 1, 2), left_backward_flow.unsqueeze(0).permute(0, 3, 1, 2))
            left_valid_masks.append(left_valid_mask)
        for right_forward_flow, right_backward_flow in zip(right_forward_flows, right_backward_flows):
            right_valid_mask = get_valid_flow_mask(right_forward_flow.unsqueeze(0).permute(0, 3, 1, 2), right_backward_flow.unsqueeze(0).permute(0, 3, 1, 2))
            right_valid_masks.append(right_valid_mask)
    flow_info = {'left_forward_flows': left_forward_flows, 'left_backward_flows': left_backward_flows,
                 'right_forward_flows': right_forward_flows, 'right_backward_flows': right_backward_flows,
                 'left_valid_masks': left_valid_masks, 'right_valid_masks': right_valid_masks}
    # breakpoint() #check: left_valid_mask, right_valid_mask ratio (0.85~0.95)
    # valid mask visualization
    debug_mode = True
    if debug_mode:
        valid_dir = os.path.join('output', seq, 'valid_mask')
        os.makedirs(valid_dir, exist_ok=True)
        valid_left_dir = os.path.join(valid_dir, 'left')
        valid_right_dir = os.path.join(valid_dir, 'right')
        os.makedirs(valid_left_dir, exist_ok=True)
        os.makedirs(valid_right_dir, exist_ok=True)
        for left_valid_mask, right_valid_mask, left_name, right_name in zip(left_valid_masks, right_valid_masks, left_names, right_names):
            left_valid_mask = left_valid_mask.squeeze().cpu().numpy()
            right_valid_mask = right_valid_mask.squeeze().cpu().numpy()
            left_name = left_name.split('/')[-1].split('.')[0]
            right_name = right_name.split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(valid_left_dir, f'{left_name}_valid_mask.png'), left_valid_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(valid_right_dir, f'{right_name}_valid_mask.png'), right_valid_mask.astype(np.uint8) * 255)


    focal_total = []
    pose_total = []
    sparse_pts3d_total = []
    dense_pts3d_total = []
    depthmaps_total = []
    confs_total = []
    scene_state_lst = []
    initialize_pose = False
    initialize_values = {}
    
    if align_mode:
        scene = sparse_global_alignment(imgs = filelist_total, pairs_in = pairs_total, cache_path = cache_dir,
                                        model = model, lr1 = lr1, niter1 = niter1, lr2 = lr2, niter2 = niter2, device = device,
                                        opt_depth = 'depth' in optim_level, shared_intrinsics = shared_intrinsics,
                                        matching_conf_thr = matching_conf_thr, initialize_pose = initialize_pose, initialize_values = initialize_values)
        
        focal = scene.get_focals().cpu()
        pose = scene.get_im_poses().cpu()
        sparse_pts3d = scene.get_sparse_pts3d()
        dense_pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth = True)
        name = f"align_{srt_frame}_{end_frame}"
        scene_state = SparseGAState(scene, False, cache_dir, None)
        
        dynamic_pcds, static_pcds, left_dynamic_pixels_score_lst, right_dynamic_pixels_score_lst = decouple_pcds(scene, scene_state, flow_info)
        
        align_convert_dual_scene_to_ply(scene_state, 'output/', seq, name, clean_depth = True, min_conf_thr = 0.2)
        # breakpoint()
        dynamic_masks = tmp_load_dynamic_mask('input/mask', (288, 512), end_frame - srt_frame + 1)
        #gaussian joint optimization (initialize gaussians as static pcds and initial pose given by scene.get_im_poses().cpu(), use masked smooth photometric loss)
        breakpoint()
        
        return samples, filelist_total, scene, scene_state, dynamic_masks
    else:
        for filelist, pairs in zip(filelist_total, pairs_total):
            scene = sparse_global_alignment(imgs = filelist, pairs_in = pairs, cache_path = cache_dir,
                                            model = model, lr1 = lr1, niter1 = niter1, lr2 = lr2, niter2 = niter2, device = device,
                                            opt_depth = 'depth' in optim_level, shared_intrinsics = shared_intrinsics,
                                            matching_conf_thr = matching_conf_thr, initialize_pose = initialize_pose, initialize_values = initialize_values)
            focal = scene.get_focals().cpu()
            pose = scene.get_im_poses().cpu()
            sparse_pts3d = scene.get_sparse_pts3d()
            dense_pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth = True)
            focal_total.append(focal)
            pose_total.append(pose)
            sparse_pts3d_total.append(sparse_pts3d)
            dense_pts3d_total.append(dense_pts3d)
            depthmaps_total.append(depthmaps)
            confs_total.append(confs)

            
            # initialize_pose = True
            pose_np = pose.numpy()
            R_mat = pose_np[:, :3, :3]
            tran = pose_np[:, :3, 3].astype(np.float32)
            # quaternion (x, y, z, w)
            quat = rotmat_to_unitquat_custom(R_mat, wxyz=False) #xyzw
            # breakpoint()
            initialize_values['quats'] = quat
            initialize_values['trans'] = tran
            
            scene_state = SparseGAState(scene, False, cache_dir, None)
            scene_state_lst.append(scene_state)
            # breakpoint()
            debug_mode = True #save ply
            if debug_mode:
                name = filelist[0].split('/')[-1].split('_')[0]
                export_path = convert_dual_scene_to_ply(scene_state, 'output/', seq, name, clean_depth = True, min_conf_thr = 0.2)
    

        for pose, sample in zip(pose_total, samples):
            (left_c2w, right_c2w) = pose
            relative_pose_l2r = torch.inverse(left_c2w) @ right_c2w
            relative_pose_pred = to_numpy(relative_pose_l2r)
            
            left_gt_pose = sample["left_pose"]
            right_gt_pose = sample["right_pose"]
            relative_pose_gt = np.linalg.inv(left_gt_pose) @ right_gt_pose
            
            # print(f'Ground truth left pose: \n{left_gt_pose}')
            # print(f'Ground truth right pose: \n{right_gt_pose}')
            # print(f'Predicted left pose: \n{left_c2w}')
            # print(f'Predicted right pose: \n{right_c2w}')
            
            # print(f'Ground truth relative pose: \n{relative_pose_gt}')
            # print(f'Predicted relative pose: \n{relative_pose_pred}')

            # print(f'Ground truth relative pose: \n{relative_pose_gt}')
            # print(f'Predicted relative pose: \n{relative_pose_pred}')
            rot_err, trans_err = evaluate_relative_pose_error(relative_pose_pred, relative_pose_gt)
            print(f'Rotation error: {rot_err:.4f} rad, Translation error: {trans_err:.4f} m')

        
        
        
        # if debug_mode:
        #     convert_multiple_dual_scene_to_ply(scene_state_lst, 'output/', seq, clean_depth = True, min_conf_thr = 0.2)    

        assert len(dense_pts3d_total) == (len(left_forward_flows) + 1)

        #dynamic mask
        debug_mode = True
        left_index = 0
        right_index = 1
        for idx, (left_forward_flow, left_backward_flow, left_valid) in enumerate(zip(left_forward_flows, left_backward_flows, left_valid_masks)):
            # dense_pts3d: list of torch.Tensor, 2 of (H * W, 3) -> (H, W, 3)
            # left_forward_flow: torch.Tensor, (H, W, 2) -> (2, H, W)
            # left_backward_flow: torch.Tensor, (H, W, 2) -> (2, H, W)
            # left_valid: torch.Tensor, (1, 1, W, H) -> (1, H, W)
            H, W, _ = left_forward_flow.shape
            
            dense_pts3d_prev = dense_pts3d_total[idx][left_index]
            dense_pts3d_next = dense_pts3d_total[idx + 1][left_index]
            cam2world_prev = pose_total[idx][left_index]
            cam2world_next = pose_total[idx + 1][left_index]
            
            dynamic_mask_1, dynamic_mask_2 = find_dynamic_mask_using_pts3d(pts3d_1 = dense_pts3d_prev.reshape(H, W, 3), 
                                                                        pts3d_2 = dense_pts3d_next.reshape(H, W, 3), 
                                                                        flow_12 = left_forward_flow.permute(2, 0, 1), 
                                                                        flow_21 = left_backward_flow.permute(2, 0, 1), 
                                                                        valid = left_valid.squeeze(0).permute(0, 2, 1), 
                                                                        transform_1 = cam2world_prev,
                                                                        transform_2 = cam2world_next,
                                                                        normalize=True,
                                                                        normalize_thres = 1.5)
            if debug_mode:
                #save dynamic mask
                mask_output_path = os.path.join('/workspace/data/jeonghonoh/mast3r/output', seq, 'dynamic_mask')
                os.makedirs(mask_output_path, exist_ok=True)
                left_mask_path = os.path.join(mask_output_path, 'left')
                os.makedirs(left_mask_path, exist_ok=True)

                # dynamic_mask_1.device: cuda:0
                dynamic_mask_1 = dynamic_mask_1.cpu().numpy().astype(np.uint8) * 255
                dynamic_mask_2 = dynamic_mask_2.cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(left_mask_path, f'{0}_{left_names[idx]}_dynamic_mask.png'), dynamic_mask_1)
                cv2.imwrite(os.path.join(left_mask_path, f'{1}_{left_names[idx]}_dynamic_mask.png'), dynamic_mask_2)

        # breakpoint()

        for idx, (right_forward_flow, right_backward_flow, right_valid) in enumerate(zip(right_forward_flows, right_backward_flows, right_valid_masks)):
            dense_pts3d_prev = dense_pts3d_total[idx][right_index]
            dense_pts3d_next = dense_pts3d_total[idx + 1][right_index]
            cam2world_prev = pose_total[idx][right_index]
            cam2world_next = pose_total[idx + 1][right_index]
            dynamic_mask_1, dynamic_mask_2 = find_dynamic_mask_using_pts3d(pts3d_1 = dense_pts3d_prev.reshape(H, W, 3), 
                                                                        pts3d_2 = dense_pts3d_next.reshape(H, W, 3), 
                                                                        flow_12 = right_forward_flow.permute(2, 0, 1), 
                                                                        flow_21 = right_backward_flow.permute(2, 0, 1), 
                                                                        valid = right_valid.squeeze(0).permute(0, 2, 1), 
                                                                        transform_1 = cam2world_prev,
                                                                        transform_2 = cam2world_next,
                                                                        normalize=True,
                                                                        normalize_thres = 1.5)
            if debug_mode:
                right_mask_path = os.path.join(mask_output_path, 'right')
                os.makedirs(right_mask_path, exist_ok=True)
                dynamic_mask_1 = dynamic_mask_1.cpu().numpy().astype(np.uint8) * 255
                dynamic_mask_2 = dynamic_mask_2.cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(right_mask_path, f'{0}_{right_names[idx]}_dynamic_mask.png'), dynamic_mask_1)
                cv2.imwrite(os.path.join(right_mask_path, f'{1}_{right_names[idx]}_dynamic_mask.png'), dynamic_mask_2)




        # # get flow (RAFT)
        # # list of images, each of shape: (288, 512, 3)
        # left_imgs, right_imgs = make_flow_image_lists(samples)
        # # (num - 1, 2, 288, 512)
        # (left_flow_forward, left_flow_backward, left_valid_mask), (right_flow_forward, right_flow_backward, right_valid_mask) = get_flow(left_imgs, right_imgs)
        # # breakpoint()
        debug_mode = True
        if debug_mode:
            #save every images, flows
            output_save_path = '/workspace/data/jeonghonoh/mast3r/output'
            os.makedirs(output_save_path, exist_ok=True)
            img_path = os.path.join(output_save_path, seq, 'images')
            os.makedirs(img_path, exist_ok=True)
            flow_path = os.path.join(output_save_path, seq, 'flows')
            os.makedirs(flow_path, exist_ok=True)
            # flow_valid_path = os.path.join(output_save_path, seq, 'flow_valid')
            # os.makedirs(flow_valid_path, exist_ok=True)
            img_path_left = os.path.join(img_path, 'left')
            os.makedirs(img_path_left, exist_ok=True)
            img_path_right = os.path.join(img_path, 'right')
            os.makedirs(img_path_right, exist_ok=True)
            # flow_path_left = os.path.join(flow_path, 'left')
            # os.makedirs(flow_path_left, exist_ok=True)
            # flow_path_right = os.path.join(flow_path, 'right')
            # os.makedirs(flow_path_right, exist_ok=True)
            # flow_valid_path_left = os.path.join(flow_valid_path, 'left')
            # os.makedirs(flow_valid_path_left, exist_ok=True)
            # flow_valid_path_right = os.path.join(flow_valid_path, 'right')
            # os.makedirs(flow_valid_path_right, exist_ok=True)
            sample_num_list = [sample["sample_number"] for sample in samples]
            for sample_num, sample in zip(sample_num_list, samples):
                left_img = sample["left_img"]
                right_img = sample["right_img"]
                left_img.save(os.path.join(img_path_left, f'{sample_num}_left.png'))
                right_img.save(os.path.join(img_path_right, f'{sample_num}_right.png'))
    #     for sample_num, (left_flow, right_flow) in zip(sample_num_list, zip(left_flow_forward, right_flow_forward)):
    #         #flow: (2, 288, 512)
    #         left_img = flow_to_image(left_flow.permute(1, 2, 0).cpu().numpy())
    #         right_img = flow_to_image(right_flow.permute(1, 2, 0).cpu().numpy())
    #         cv2.imwrite(os.path.join(flow_path_left, f'{sample_num}_left_flow.png'), left_img)
    #         cv2.imwrite(os.path.join(flow_path_right, f'{sample_num}_right_flow.png'), right_img)
    #     for sample_num, (left_valid, right_valid) in zip(sample_num_list, zip(left_valid_mask, right_valid_mask)):
    #         #valid: (1, 288, 512)
    #         left_img = np.squeeze(left_valid.cpu().numpy(), axis=0).astype(np.uint8) * 255
    #         right_img = np.squeeze(right_valid.cpu().numpy(), axis=0).astype(np.uint8) * 255
    #         cv2.imwrite(os.path.join(flow_valid_path_left, f'{sample_num}_left_valid.png'), left_img)
    #         cv2.imwrite(os.path.join(flow_valid_path_right, f'{sample_num}_right_valid.png'), right_img)



    
    
    # breakpoint()

        


    
    return