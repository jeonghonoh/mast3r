#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo executable
# --------------------------------------------------------
import os
import torch
import tempfile
import numpy as np
from contextlib import nullcontext

from mast3r.demo import get_args_parser, main_demo, forward_two_images

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

from refine import prepare_refining, refining_pose_static_gaussians, get_args_parser_custom

torch.backends.cuda.matmul.allow_tf32 = False  # for gpu >= Ampere and pytorch >= 1.12

def convert_format(scene, samples, dynamic_masks):
    focal = scene.get_focals().cpu()
    pose = scene.get_im_poses().cpu()
    sparse_pts3d = scene.get_sparse_pts3d()
    dense_pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth = True)
    
    device = dense_pts3d[0].device
    
    assert len(dynamic_masks) == len(dense_pts3d)
    breakpoint()
    dynamic_masks = [torch.Tensor(mask).to(device) for mask in dynamic_masks]
    static_masks = [~mask for mask in dynamic_masks]
    static_masks = [mask.float() for mask in static_masks]
    
    static_pts3d = [p[m.bool()] for p, m in zip(dense_pts3d, static_masks)]
    static_pts3d = [p.cpu().numpy() for p in static_pts3d]
    static_pts3d = np.concatenate(static_pts3d, axis=0)
    
    
    return imgs, static_pts3d, static_col


if __name__ == '__main__':
    
    ORIGINAL_CODE = False

    if ORIGINAL_CODE:
        parser = get_args_parser()
        args = parser.parse_args()
        set_print_with_timestamp()
        if args.server_name is not None:
            server_name = args.server_name
        else:
            server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

        if args.weights is not None:
            weights_path = args.weights
        else:
            weights_path = "naver/" + args.model_name

        model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
        chkpt_tag = hash_md5(weights_path)

        def get_context(tmp_dir):
            return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
                else nullcontext(tmp_dir)
        with get_context(args.tmp_dir) as tmpdirname:
            cache_path = os.path.join(tmpdirname, chkpt_tag)
            os.makedirs(cache_path, exist_ok=True)
            main_demo(cache_path, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent,
                    share=args.share, gradio_delete_cache=args.gradio_delete_cache)

    else:
        
        #args: (local_network=False, server_name=None, image_size=512, server_port=None, weights=None, model_name='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', device='cuda', tmp_dir=None, silent=False, share=True, gradio_delete_cache=None, data_path='/workspace/data/jeonghonoh/dataset/dynamic/dual_arm/seq_01')    
        # if args.weights is not None:
        #     weights_path = args.weights
        # else:
        #     weights_path = "naver/" + args.model_name
        parser, lp, op, pp = get_args_parser_custom()
        args = parser.parse_args()
        args.save_iter.append(args.iterations)
        args = parser.parse_args()
        
        model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to('cuda')

        # forward_two_images(data_path = args.data_path, model = model, device = args.device, image_size=args.image_size, silent=False)

        # seq_01 70~78 차 충돌, 120~129 공만 움직임, 300~304 휴지 and 공 멈춤, 306~313 휴지 움직임
        # seq_02 95~105 뷰 겹치는 멈춰있는거, 125~140 로봇팔 움직이는거 찍힌거 197~202 공만 움직임 290~293 차 충돌
        rootpath = "/workspace/data/jeonghonoh/dataset/dynamic/dual_arm/"
        
        seq = "seq_01"
        datapath = os.path.join(rootpath, seq)
        
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        
        srt_frame = 70
        end_frame = 78
        samples, image_files, scene, scene_state, dynamic_masks = forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = 'cuda', image_size=512, silent=False, seq = seq)

        # srt_frame = 120
        # end_frame = 129
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # srt_frame = 300
        # end_frame = 304
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # srt_frame = 306
        # end_frame = 313
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # seq = "seq_02"
        # datapath = os.path.join(rootpath, seq)
        
        # if not os.path.exists(datapath):
        #     os.makedirs(datapath)
        
        # srt_frame = 95
        # end_frame = 105
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # srt_frame = 125
        # end_frame = 140
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # srt_frame = 197
        # end_frame = 202
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # srt_frame = 290
        # end_frame = 293
        # forward_two_images(data_path = datapath, srt_frame = srt_frame, end_frame = end_frame, model = model, device = args.device, image_size=args.image_size, silent=False, seq = seq)
        
        # breakpoint()
        
        #imgs: list of dict, dict_keys(['img', 'true_shape', 'idx', 'instance', 'mask', 'dynamic_mask'])
        #static_pts3d: numpy.ndarray, shape=(n_static_pts, 3), dtype=float32
        #static_col: numpy.ndarray, shape=(n_static_pts, 3), dtype=float32
        # imgs, static_pts3d, static_col = convert_format(scene, samples, dynamic_masks)
        
        # org_imgs_shape = (288, 512)
        # breakpoint()
        
        # extrinsics_w2c, intrinsics, focals, imgs, pts3d, confs, overlapping_masks = prepare_refining(scene, args.output_dir, imgs, image_files, dynamic_masks, 0.1, args.n_views, static_pts3d, static_col, org_imgs_shape, conf_aware_ranking=args.conf_aware_ranking, focal_avg=False, infer_video=False)
        
        # refining_pose_static_gaussians(lp.extract(args), op.extract(args), pp.extract(args), args.test_iter, args.save_iter, args.checkpoint_iter, args.start_checkpoint, args.debug_from, \
        #                                                                            extrinsics_w2c, intrinsics, focals, imgs, pts3d, confs, overlapping_masks, dynamic_pts3d, static_pts3d)
        