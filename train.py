#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation  # IMGS_BUILD_ROTATION_IMPORT_PATCH
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from torch import nn
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        # PYTORCH26_CHECKPOINT_LOAD_PATCH
        try:
            (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        except TypeError:
            (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # IMGS_FIELD_LOAD_PATCH
    implicit_metric_field = None
    metric_stats_history = []
    if os.environ.get("IMPLICIT_METRIC_ENABLED", "0") == "1":
        from implicit_metric_field import load_metric_field
        implicit_metric_field = load_metric_field(
            os.environ["IMPLICIT_METRIC_CKPT"], device="cuda"
        )
        print("IMGS_FIELD_LOADED", os.environ["IMPLICIT_METRIC_CKPT"])

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    test_imgs_dir = os.path.join(args.model_path, "test_imgs/")
    os.makedirs(test_imgs_dir, exist_ok = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg_mask = None
    loss_accum = 0
    pseudo_stack = None
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # IMGS_GEOMETRY_FREEZE_PATCH
        freeze_after = int(os.environ.get("IMGS_FREEZE_GEOMETRY_AFTER", "0"))
        if freeze_after > 0 and iteration > freeze_after:
            for param_group in gaussians.optimizer.param_groups:
                if param_group.get("name") in ("xyz", "scaling", "rotation"):
                    param_group["lr"] = 0.0

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        is_pseudo_view = getattr(viewpoint_cam, "is_pseudo", False)
        pseudo_start = int(os.environ.get("PSEUDO_START", "0"))
        # PSEUDO_DISTILL_LOSS_PATCH
        if is_pseudo_view:
            weight_mode = os.environ.get("PSEUDO_WEIGHT_MODE", "uniform")
            pseudo_weight = viewpoint_cam.pseudo_weight.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0) if weight_mode == "qgeo" else torch.ones_like(image[:1])
            pixel_l1 = torch.abs(image - gt_image).mean(dim=0, keepdim=True)
            weighted_l1 = (pixel_l1 * pseudo_weight).sum() / (pseudo_weight.sum() + 1e-8)

            view_conf = pseudo_weight.mean().detach()
            pseudo_ramp = int(os.environ.get("PSEUDO_DISTILL_RAMP", "500"))
            pseudo_max = float(os.environ.get("PSEUDO_LOSS_MAX", "0.35"))
            pseudo_strength = min(max((iteration - pseudo_start) / float(pseudo_ramp), 0.0), 1.0) if pseudo_ramp > 0 else 1.0
            
            Ll1, ssim_value = weighted_l1, torch.zeros((), device=image.device)
            loss = (pseudo_max * pseudo_strength) * view_conf * weighted_l1
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # IMGS_ALIGNMENT_LOSS_PATCH
        if implicit_metric_field is not None and os.environ.get("IMPLICIT_METRIC_ALIGN", "0") == "1":
            align_start = int(os.environ.get("IMPLICIT_METRIC_ALIGN_START", "1000"))
            align_end = int(os.environ.get("IMPLICIT_METRIC_ALIGN_END", "5000"))
            if iteration >= align_start and iteration <= align_end:
                from implicit_metric_field import metric_strength
                align_strength = metric_strength(iteration, int(os.environ.get("IMPLICIT_METRIC_START", "1000")), int(os.environ.get("IMPLICIT_METRIC_END", "5500")), int(os.environ.get("IMPLICIT_METRIC_RAMP", "750")))
                sample_count = min(int(os.environ.get("IMPLICIT_METRIC_ALIGN_SAMPLES", "8192")), int(gaussians._xyz.shape[0]))
                if sample_count > 0 and align_strength > 0:
                    ids = torch.randint(0, gaussians._xyz.shape[0], (sample_count,), device=gaussians._xyz.device)
                    with torch.no_grad():
                        normal_target, align_trust = implicit_metric_field(gaussians._xyz.detach()[ids])

                    from utils.general_utils import build_rotation
                    rot = build_rotation(gaussians._rotation[ids])
                    scales = gaussians.get_scaling[ids]

                    shortest = torch.argmin(scales, dim=-1)
                    gather_idx = shortest[:, None, None].expand(-1, 3, 1)
                    shortest_axis = torch.gather(rot, 2, gather_idx).squeeze(-1)
                    cos2 = (shortest_axis * normal_target).sum(dim=-1, keepdim=True).pow(2)

                    shape_mode = os.environ.get("IMPLICIT_COVARIANCE_MODE", "legacy_align")

                    if shape_mode == "joint":
                        trust_threshold = float(os.environ.get("SHAPE_TRUST_THRESHOLD", "0.78"))
                        trust_temperature = max(float(os.environ.get("SHAPE_TRUST_TEMPERATURE", "0.08")), 1e-4)
                        q_cal = torch.sigmoid((align_trust - trust_threshold) / trust_temperature)

                        opacity_support = gaussians.get_opacity.detach()[ids]
                        opacity_floor = float(os.environ.get("SHAPE_OPACITY_FLOOR", "0.25"))
                        support = opacity_floor + (1.0 - opacity_floor) * opacity_support
                        shape_w = (q_cal * support).detach()

                        align_loss = (shape_w * (1.0 - cos2)).sum() / (shape_w.sum() + 1e-8)

                        n_local = torch.bmm(rot.transpose(1, 2), normal_target.unsqueeze(-1)).squeeze(-1)
                        scales2 = scales.pow(2)
                        normal_var = (n_local.pow(2) * scales2).sum(dim=-1, keepdim=True)
                        total_var = scales2.sum(dim=-1, keepdim=True)
                        tangent_var = ((total_var - normal_var).clamp_min(1e-10) / 2.0)
                        thickness_ratio = torch.sqrt((normal_var + 1e-10) / (tangent_var + 1e-10))
                        max_ratio = float(os.environ.get("THICKNESS_MAX_RATIO", "0.45"))
                        excess = torch.clamp(thickness_ratio - max_ratio, min=0.0)
                        thickness_loss = (shape_w * excess.pow(2)).sum() / (shape_w.sum() + 1e-8)

                        loss = loss + (float(os.environ.get("IMPLICIT_METRIC_ALIGN_LAMBDA", "0.005")) * float(align_strength) * align_loss)
                        loss = loss + (float(os.environ.get("THICKNESS_LAMBDA", "0.002")) * float(align_strength) * thickness_loss)
                    else:
                        align_loss = (align_trust.detach() * (1.0 - cos2)).sum() / (align_trust.detach().sum() + 1e-8)
                        loss = loss + (float(os.environ.get("IMPLICIT_METRIC_ALIGN_LAMBDA", "0.005")) * float(align_strength) * align_loss)

        loss.backward()

        # IMGS_GRADIENT_METRIC_PATCH
        if implicit_metric_field is not None:
            from implicit_metric_field import metric_strength, transform_xyz_gradient_evidence_inplace
            metric_s = metric_strength(iteration, int(os.environ.get("IMPLICIT_METRIC_START", "1000")), int(os.environ.get("IMPLICIT_METRIC_END", "5500")), int(os.environ.get("IMPLICIT_METRIC_RAMP", "750")))
            metric_stats = transform_xyz_gradient_evidence_inplace(
                gaussians=gaussians, field=implicit_metric_field, strength=metric_s,
                rho_min=float(os.environ.get("IMPLICIT_METRIC_RHO_MIN", "0.05")),
                trust_threshold=float(os.environ.get("EC_TRUST_THRESHOLD", "0.78")),
                trust_temperature=float(os.environ.get("EC_TRUST_TEMPERATURE", "0.08")),
                normal_need_beta=float(os.environ.get("EC_NORMAL_NEED_BETA", "0.35")),
                normal_need_clip=float(os.environ.get("EC_NORMAL_NEED_CLIP", "3.0")),
                chunk_size=int(os.environ.get("IMPLICIT_METRIC_CHUNK", "65536"))
            )
            log_interval = int(os.environ.get("IMPLICIT_METRIC_LOG_INTERVAL", "100"))
            if metric_s > 0 and iteration % max(log_interval, 1) == 0:
                metric_stats_history.append({
                    "iteration": int(iteration),
                    "strength": float(metric_s),
                    "mode": "evidence",
                    **metric_stats,
                })
        iter_end.record()

        with torch.no_grad():
            # STEREOGS_ADAPTIVE_OPACITY_DECAY_TRAIN_PATCH
            # IMGS_OPACITY_DECAY_WINDOW_PATCH
            if os.environ.get("STEREOGS_OPACITY_DECAY_ENABLED", "0") == "1":
                decay_end = int(os.environ.get("STEREOGS_OPACITY_DECAY_END", "0"))
                if decay_end <= 0 or iteration <= decay_end:
                    gaussians.adaptive_opacity_decay(
                        min_decay_rate=float(os.environ.get("STEREOGS_OPACITY_DECAY_FACTOR", "0.99")),
                        sensitivity=float(os.environ.get("STEREOGS_GRAD_SENSITIVITY", "0.5")),
                    )

            # Progress bar
            if iteration > opt.densify_from_iter:
                loss_accum += loss.clone().detach().item()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                # IMGS_METRIC_STATS_SAVE_PATCH
                if implicit_metric_field is not None:
                    stats_path = os.path.join(
                        scene.model_path, "imgs_metric_stats_{}.pt".format(iteration)
                    )
                    torch.save(metric_stats_history, stats_path)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # STEREOGS_DISABLE_OPACITY_RESET_PATCH
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if os.environ.get("STEREOGS_OPACITY_DECAY_ENABLED", "0") != "1":
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            render_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000,])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
