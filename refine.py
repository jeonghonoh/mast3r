# --------------------------------------------------------
# monst3r + instantsplat
# --------------------------------------------------------

import argparse
import math
import os
import torch
import numpy as np
import tempfile
import functools
import copy
# import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
import matplotlib.pyplot as pl

import uuid
from tqdm import tqdm
from random import randint
from scene import Scene, GaussianModel
from pathlib import Path
from time import time
from gaussian_renderer import render, network_gui
from sfm_utils import save_time, save_intrinsics, save_extrinsic, save_points3D, save_images_with_masks, compute_co_vis_masks, project_points
from loss_utils import l1_loss, ssim, l1_loss_mask, masked_ssim
from pose_utils import get_camera_from_tensor
from image_utils import psnr, psnr_with_mask
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser, Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1
ABLATION = False

def get_args_parser_custom():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    # parser.add_argument("--weights", type=str, help="path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    # parser.add_argument("--input_dir", '-n', type=str, required=True, help="Path to input images directory")
    parser.add_argument("--output_dir", '-o', type=str, required=True, help="Path to output directory")
    parser.add_argument("--silent", action='store_true', default=False, help="silence logs")
    # parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument("--use_sam2_gt_mask", action='store_true', default=False, help="Use ground truth mask for SAM2")
    
    parser.add_argument('--not_batchify', action='store_true', default=False, help='Use non batchify mode for global optimization')
    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")
    
    #instantsplat arguments
    parser.add_argument("--conf_aware_ranking", action='store_true', default=True, help="Use confidence-aware ranking")
    parser.add_argument("--test_iter", nargs="+", type=int, default=[])
    parser.add_argument("--save_iter", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iter", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--num_views', type=int, required=True)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    return parser, lp, op, pp

# def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
#                             clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
#     """
#     extract 3D_model (glb file) from a reconstructed scene
#     """
#     if scene is None:
#         return None
#     # post processes
#     if clean_depth:
#         scene = scene.clean_pointcloud()
#     if mask_sky:
#         scene = scene.mask_sky()

#     # get optimized values from scene
#     rgbimg = scene.imgs
#     focals = scene.get_focals().cpu()
#     cams2world = scene.get_im_poses().cpu()
#     # 3D pointcloud from depthmap, poses and intrinsics
#     pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
#     scene.min_conf_thr = min_conf_thr
#     scene.thr_for_init_conf = thr_for_init_conf
#     msk = to_numpy(scene.get_masks())
#     cmap = pl.get_cmap('viridis')
#     cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
#     cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
#     return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
#                                         transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
#                                         cam_color=cam_color)

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

# def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
#                             as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
#                             use_sam2_gt_mask, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
#                             flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, fps, num_frames):
#     """
#     from a list of images, run dust3r inference, global aligner.
#     then run get_3D_model_from_scene
#     """
#     translation_weight = float(translation_weight)
#     if new_model_weights != args.weights:
#         model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
#     model.eval()
#     if use_sam2_gt_mask:
#         dynamic_mask_path = filelist[0].split('images')[0] + 'masks'
#     else:
#         dynamic_mask_path = None
#     print("flie list: ", filelist)
#     imgs, image_files, org_imgs_shape, sam2_gt_masks = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)

#     if len(imgs) == 1:
#         imgs = [imgs[0], copy.deepcopy(imgs[0])]
#         imgs[1]['idx'] = 1
#     if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
#         scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
#     elif scenegraph_type == "oneref":
#         scenegraph_type = scenegraph_type + "-" + str(refid)

#     pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
#     output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

#     use_gt_mask = False
#     if len(imgs) > 2:
#         mode = GlobalAlignerMode.PointCloudOptimizer  
#         scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
#                                flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
#                                num_total_iter=niter, empty_cache= len(filelist) > 72, batchify=not args.not_batchify)
#     else:
#         mode = GlobalAlignerMode.PairViewer
#         scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
#     lr = 0.01

#     if mode == GlobalAlignerMode.PointCloudOptimizer:
#         loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        
    
#     #monst3r global align 끝나고 나서 decouple 해주기: monst3r는 그냥 initialization tool이고 joint optimization 과정에서 refining하는거니까 global align 후, refine 하기 전에 decouple 해줘야함
#     dynamic_pts3d, static_pts3d, dynamic_col, static_col, dynamic_masks = scene.decouple_static_dynamic_pcds(use_sam2_gt_mask, sam2_gt_masks)
#     # (dynamic_num, 3), (static_num, 3)    
    
#     if ABLATION:
#         static_pts3d = np.concatenate([static_pts3d, dynamic_pts3d], axis=0)
#         static_col = np.concatenate([static_col, dynamic_col], axis=0)
    
    
#     save_folder = f'{args.output_dir}/initial'  #default is 'path/to/output/inital'
#     os.makedirs(save_folder, exist_ok=True)
#     outfile = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
#                             clean_depth, transparent_cams, cam_size, show_cam)
#     poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
#     K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
#     depth_maps = scene.save_depth_maps(save_folder)
#     dynamic_masks = scene.save_dynamic_masks(save_folder)
#     conf = scene.save_conf_maps(save_folder)
#     init_conf = scene.save_init_conf_maps(save_folder)
#     rgbs = scene.save_rgb_imgs(save_folder)
#     enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 
#     # also return rgb, depth and confidence imgs
#     # depth is normalized with the max value for all images
#     # we apply the jet colormap on the confidence maps
#     rgbimg = scene.imgs
#     depths = to_numpy(scene.get_depthmaps())
#     confs = to_numpy([c for c in scene.im_conf])
#     init_confs = to_numpy([c for c in scene.init_conf_maps])
#     cmap = pl.get_cmap('jet')
#     depths_max = max([d.max() for d in depths])
#     depths = [cmap(d/depths_max) for d in depths]
#     confs_max = max([d.max() for d in confs])
#     confs = [cmap(d/confs_max) for d in confs]
#     init_confs_max = max([d.max() for d in init_confs])
#     init_confs = [cmap(d/init_confs_max) for d in init_confs]

#     return scene, outfile, imgs, image_files, dynamic_masks, dynamic_pts3d, static_pts3d, dynamic_col, static_col, org_imgs_shape


def prepare_refining(scene, outdir, imgs, image_files, dynamic_masks, depth_thre, num_views, static_pcds, static_col, org_imgs_shape, conf_aware_ranking=False, focal_avg=False, infer_video=False):
    assert num_views == len(imgs), f"Number of images {len(imgs)} does not match the number of views {num_views}"
    extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
    intrinsics = to_numpy(scene.get_intrinsics())
    focals = to_numpy(scene.get_focals())
    imgs = np.array(scene.imgs) # (img_num, H, W, 3)
    pts3d = to_numpy(scene.get_pts3d())
    pts3d = np.array(pts3d) # (img_num, H, W, 3)
    depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
    values = [param.detach().cpu().numpy() for param in scene.im_conf]
    confs = np.array(values)
    
    
    if conf_aware_ranking:
        print(f'>> Confiden-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted average confidence scores:", sorted_conf_avg_conf_scores)
    else:
        sorted_conf_indices = np.arange(num_views)
        print("Sorted indices:", sorted_conf_indices)

    # Calculate the co-visibility mask
    print(f'>> Calculate the co-visibility mask...')
    if depth_thre > 0:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None

    # Save results
    focals = np.repeat(focals[0], num_views)
    sparse_0_path = os.path.join(outdir, f"sparse_{num_views}/0")
    sparse_0_path = Path(sparse_0_path)
    os.makedirs(sparse_0_path, exist_ok=True)
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files)
    save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=False)
    # save static/dynamic points separately
    pts_num = save_points3D(sparse_0_path, static_col, static_pcds, confs, masks=None, use_masks=False, mode='static')
    print(f'>> Number of static points: {pts_num}')
    save_images_with_masks(outdir, num_views, imgs, dynamic_masks, image_files)
    
    if True:
        sparse_1_path = os.path.join(outdir, f"sparse_{num_views}/1")
        sparse_1_path = Path(sparse_1_path)
        os.makedirs(sparse_1_path, exist_ok=True)
        save_extrinsic(sparse_1_path, extrinsics_w2c, image_files)
        save_intrinsics(sparse_1_path, focals, org_imgs_shape, imgs.shape, save_focals=False)
        
    return extrinsics_w2c, intrinsics, focals, imgs, pts3d, confs, overlapping_masks

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % 5000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def prepare_confidence(conf, device='cuda', scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
    
    Args:
        conf: confidence map
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.
    
    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_tensor = torch.from_numpy(conf).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
    
    return lr_modifiers

def save_pose(path, quat_pose, train_cams, llffhold=2):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]
    
    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)
    
    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, colmap_poses)

def refining_pose_static_gaussians(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, 
                                   extrinsics_w2c, intrinsics, focals, imgs, pts3d, confs, overlapping_masks, dynamic_pts3d, static_pts3d):
    '''Initialize static gaussians + refine poses'''
    breakpoint()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # per-point-optimizer
    # confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.num_views}/0", "confidence_dsp.npy")
    confidence_lr = prepare_confidence(confs, device='cuda', scale=(1, 100))
    scene = Scene(dataset, gaussians)

    if opt.pp_optimizer:
        gaussians.training_setup_pp(opt, confidence_lr)                          
    else:
        gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    train_cams_init = scene.getTrainCameras().copy()
    for save_iter in saving_iterations:
        os.makedirs(scene.model_path + f'/pose/ours_{save_iter}', exist_ok=True)
        save_pose(scene.model_path + f'/pose/ours_{save_iter}/pose_org.npy', gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    
    save_pose(args.output_dir + '/pose/before_refining_pose.npy', gaussians.P, train_cams_init)
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
    first_iter += 1
    start = time()
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if opt.optim_pose==False:
            gaussians.P.requires_grad_(False)
        else:
            gaussians.P.requires_grad_(True)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        dynamic_mask = viewpoint_cam.mask.cuda()
        
        
        MASKED_LOSS = not ABLATION
        if MASKED_LOSS:
            Ll1 = l1_loss_mask(image, gt_image, 1-dynamic_mask)
            ssim_value = masked_ssim(image, gt_image, 1-dynamic_mask)
        else:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        iter_end.record()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            # if iteration < opt.densify_until_iter:
                # # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Log and save
            if iteration == opt.iterations:
                end = time()
                train_time_wo_log = end - start
                save_time(scene.model_path, '[2] train_joint_TrainTime', train_time_wo_log)
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + f'/pose/ours_{iteration}/pose_optimized.npy', gaussians.P, train_cams_init)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    end = time()
    train_time = end - start
    save_time(scene.model_path, '[2] train_joint', train_time)
    save_pose(args.output_dir + '/pose/after_refining_pose.npy', gaussians.P, train_cams_init)
    
    return 



# if __name__ == '__main__':
#     parser, lp, op, pp = get_args_parser()
#     args = parser.parse_args()
#     args.save_iter.append(args.iterations)

#     if args.output_dir is not None:
#         os.makedirs(args.output_dir, exist_ok=True)

#     if args.weights is not None and os.path.exists(args.weights):
#         weights_path = args.weights
#     else:
#         weights_path = args.model_name

#     model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)


#     if args.input_dir is not None:
#         scene, outfile, imgs, image_files, dynamic_masks, dynamic_pts3d, static_pts3d, dynamic_col, static_col, org_imgs_shape = get_reconstructed_scene(args, args.output_dir, model, args.device, args.silent, args.image_size, filelist = args.input_dir, \
#                                                                                                                 schedule = 'linear', niter = 300, min_conf_thr = 1.1, as_pointcloud = True, mask_sky = False, clean_depth = True, \
#                                                                                                                 transparent_cams = False, cam_size = 0.05, show_cam = True, scenegraph_type = "swinstride", winsize = 5, refid = 0, \
#                                                                                                                 use_sam2_gt_mask = args.use_sam2_gt_mask, new_model_weights = args.weights, temporal_smoothing_weight = 0.01, translation_weight = 1.0, \
#                                                                                                                 shared_focal = False, flow_loss_weight = 0.01, flow_loss_start_iter = 0.1, flow_loss_threshold = 25, \
#                                                                                                                 fps = args.fps, num_frames = args.num_frames)
        
#         extrinsics_w2c, intrinsics, focals, imgs, pts3d, confs, overlapping_masks = prepare_refining(scene, args.output_dir, imgs, image_files, dynamic_masks, 0.1, args.n_views, static_pts3d, static_col, org_imgs_shape, conf_aware_ranking=args.conf_aware_ranking, focal_avg=False, infer_video=False)
        
#         refining_pose_static_gaussians(lp.extract(args), op.extract(args), pp.extract(args), args.test_iter, args.save_iter, args.checkpoint_iter, args.start_checkpoint, args.debug_from, \
#                                                                                    extrinsics_w2c, intrinsics, focals, imgs, pts3d, confs, overlapping_masks, dynamic_pts3d, static_pts3d)
        

#         #rendering
#         # render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args)
        