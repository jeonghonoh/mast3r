#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
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
    assert len(pts3d) == 2
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
    valid_msk = np.isfinite(pts.sum(axis=1))
    pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
    scene.add_geometry(pct)
    
    # add each camera: 생략
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
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

#raft_ws  
import cv2
import torch
from tqdm import tqdm
from third_party.raft import load_RAFT
from dust3r.utils.geom_opt import OccMask
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

    # breakpoint() #check: left_valid_mask, right_valid_mask ratio (0.85~0.95)


    focal_total = []
    pose_total = []
    sparse_pts3d_total = []
    dense_pts3d_total = []
    depthmaps_total = []
    confs_total = []
    scene_state_lst = []
    for filelist, pairs in zip(filelist_total, pairs_total):
        scene = sparse_global_alignment(imgs = filelist, pairs_in = pairs, cache_path = cache_dir,
                                        model = model, lr1 = lr1, niter1 = niter1, lr2 = lr2, niter2 = niter2, device = device,
                                        opt_depth = 'depth' in optim_level, shared_intrinsics = shared_intrinsics,
                                        matching_conf_thr = matching_conf_thr)
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
        
        scene_state = SparseGAState(scene, False, cache_dir, None)
        scene_state_lst.append(scene_state)
        # breakpoint()
        debug_mode = True #save ply
        if debug_mode:
            name = filelist[0].split('/')[-1].split('_')[0]
            export_path = convert_dual_scene_to_ply(scene_state, 'output/', seq, name, clean_depth = True, min_conf_thr = 0.2)
    
    # if debug_mode:
    #     convert_multiple_dual_scene_to_ply(scene_state_lst, 'output/', seq, clean_depth = True, min_conf_thr = 0.2)    
    # breakpoint()

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
    # debug_mode = True
    # if debug_mode:
    #     #save every images, flows
    #     output_save_path = '/workspace/data/jeonghonoh/mast3r/output'
    #     os.makedirs(output_save_path, exist_ok=True)
    #     img_path = os.path.join(output_save_path, seq, 'images')
    #     os.makedirs(img_path, exist_ok=True)
    #     flow_path = os.path.join(output_save_path, seq, 'flows')
    #     os.makedirs(flow_path, exist_ok=True)
    #     flow_valid_path = os.path.join(output_save_path, seq, 'flow_valid')
    #     os.makedirs(flow_valid_path, exist_ok=True)
    #     img_path_left = os.path.join(img_path, 'left')
    #     os.makedirs(img_path_left, exist_ok=True)
    #     img_path_right = os.path.join(img_path, 'right')
    #     os.makedirs(img_path_right, exist_ok=True)
    #     flow_path_left = os.path.join(flow_path, 'left')
    #     os.makedirs(flow_path_left, exist_ok=True)
    #     flow_path_right = os.path.join(flow_path, 'right')
    #     os.makedirs(flow_path_right, exist_ok=True)
    #     flow_valid_path_left = os.path.join(flow_valid_path, 'left')
    #     os.makedirs(flow_valid_path_left, exist_ok=True)
    #     flow_valid_path_right = os.path.join(flow_valid_path, 'right')
    #     os.makedirs(flow_valid_path_right, exist_ok=True)
    #     sample_num_list = [sample["sample_number"] for sample in samples]
    #     for sample_num, sample in zip(sample_num_list, samples):
    #         left_img = sample["left_img"]
    #         right_img = sample["right_img"]
    #         left_img.save(os.path.join(img_path_left, f'{sample_num}_left.png'))
    #         right_img.save(os.path.join(img_path_right, f'{sample_num}_right.png'))
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