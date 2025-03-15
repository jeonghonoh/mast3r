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

def load_stereo_data(folder_path, srt_frame, end_frame, size, square_ok=False, verbose=True, debug_mode=False):
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
    

import torch
from tqdm import tqdm
from third_party.raft import load_RAFT
from dust3r.utils.geom_opt import OccMask

# 예시 사용법:
# samples = load_stereo_data(data_path, srt_frame, end_frame, size=512, verbose=True)
# left_imgs, right_imgs = make_flow_image_lists(samples)
# flow_left, flow_right, valid_mask_left, valid_mask_right = get_flow(left_imgs, right_imgs)
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
    
    def compute_flow_seq(img_seq):
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



# def get_flow(left_imgs: list, right_imgs: list):
#     print('precomputing flow...')
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     get_valid_flow_mask = OccMask(th=3.0)
#     # pair_imgs : [(num, W, H, 3), (num, W, H, 3)]
#     # (90, 512, 272, 3)
#     # pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]
    

#     flow_net = load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
#     flow_net = flow_net.to(device)
#     flow_net.eval()

#     with torch.no_grad():
#         chunk_size = 12
#         flow_ij = []
#         flow_ji = []
#         num_pairs = len(pair_imgs[0])
#         for i in tqdm(range(0, num_pairs, chunk_size)):
#             end_idx = min(i + chunk_size, num_pairs)
#             imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
#                     torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
#             flow_ij.append(flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
#                                     imgs_ij[1].permute(0, 3, 1, 2) * 255, 
#                                     iters=20, test_mode=True)[1])
#             flow_ji.append(flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
#                                     imgs_ij[0].permute(0, 3, 1, 2) * 255, 
#                                     iters=20, test_mode=True)[1])

#         flow_ij = torch.cat(flow_ij, dim=0)
#         flow_ji = torch.cat(flow_ji, dim=0)
#         valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
#         valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
#     print('flow precomputed')
#     # delete the flow net
#     if flow_net is not None: del flow_net
#     return flow_ij, flow_ji, valid_mask_i, valid_mask_j
    

# def forward_two_images(data_path, outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
#                             filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
#                             as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
#                             win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
def forward_two_images(data_path, srt_frame, end_frame, model, device, image_size = 512, silent = False):
    # scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
    # as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    # scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics
    
    cache_dir = os.path.join('/workspace/data/jeonghonoh/mast3r/', 'cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    optim_level = 'refine+depth'
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
    
    # breakpoint()
    focal_total = []
    pose_total = []
    sparse_pts3d_total = []
    dense_pts3d_total = []
    depthmaps_total = []
    confs_total = []
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

    breakpoint()

    # get flow
    left_imgs, right_imgs = make_flow_image_lists(samples)
    breakpoint()
    (left_flow_forward, left_flow_backward, left_valid_mask), (right_flow_forward, right_flow_backward, right_valid_mask) = get_flow(left_imgs, right_imgs)
    breakpoint()

    
    return