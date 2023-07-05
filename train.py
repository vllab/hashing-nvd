import torch
import torch.optim as optim
import numpy as np

from evaluate import eval
from datetime import datetime
from unwrap_utils import get_tuples, pre_train_mapping, load_input_data, save_video
import sys

import logging
from config_utils import config_load, config_save

from pathlib import Path

from networks import build_network
from losses import *
from unwrap_utils import Timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cfg):
    maximum_number_of_frames = cfg["maximum_number_of_frames"]
    resx = np.int64(cfg["resx"])
    resy = np.int64(cfg["resy"])
    iters_num = cfg["iters_num"]
    loss_cfg = cfg["losses"]

    #batch size:
    samples = cfg["samples_batch"]

    # optionally it is possible to load a checkpoint
    load_checkpoint = cfg["load_checkpoint"] # set to true to continue from a checkpoint
    checkpoint_path = cfg["checkpoint_path"]

    # a data folder that contains folders named "[video_name]/flow", "[video_name]/masks", "[video_name]/video_frames"
    data_folder = Path(cfg["data_folder"])
    results_folder_name = cfg["results_folder_name"] # the folder (under the code's folder where the experiments will be saved.
    folder_suffix = cfg["folder_suffix"] # for each experiment folder (saved inside "results_folder_name") add this string

    pretrain_iter_number = cfg["pretrain_iter_number"]

    # the scale of the texture uv coordinates relative to frame's xy coordinates
    uv_mapping_scales = cfg["uv_mapping_scales"]

    mappings_cfg = cfg["model_mapping"]
    alpha_cfg = cfg["alpha"]

    logger_cfg = cfg["logger"]
    eval_cfg = cfg["evaluation"]

    num_of_maps = len(mappings_cfg)

    timer = Timer()

    vid_name = data_folder.name

    results_folder = Path(
        f'./{results_folder_name}/{vid_name}_{folder_suffix}')

    results_folder.mkdir(parents=True, exist_ok=True)
    config_save(cfg, '%s/config.py'%results_folder)
    logging.basicConfig(
        filename='%s/%s.log' % (results_folder, datetime.utcnow().strftime("%m_%d_%Y__%H_%M_%S_%f")),
        level=logging.INFO,
        format='%(asctime)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.info('Started')
    optical_flows_mask, video_frames, optical_flows_reverse_mask, mask_frames, video_frames_dx, video_frames_dy, optical_flows_reverse, optical_flows = load_input_data(
        resy, resx, maximum_number_of_frames, data_folder, True,  True)
    number_of_frames=video_frames.shape[3]
    # save the video in the working resolution
    save_video(video_frames, results_folder)

    model_F_mappings = torch.nn.ModuleList()
    for mapping_cfg in mappings_cfg:
        model_F_mappings.append(
            build_network(device=device, **mapping_cfg))

    model_alpha = build_network(
        num_of_maps=num_of_maps,
        device=device,
        **alpha_cfg)

    start_iteration = 1

    optimizer_all = list()
    optimizer_all.extend(model_alpha.get_optimizer_list())
    for model_F_mapping in model_F_mappings:
        optimizer_all.extend(model_F_mapping.get_optimizer_list())
    optimizer_all = optim.Adam(optimizer_all)

    rez = np.maximum(resx, resy)

    # get losses
    loss_funcs = dict()
    if loss_cfg.get('rgb'):
        loss_funcs['rgb'] = RGBLoss(loss_cfg['rgb']['weight'])
    if loss_cfg.get('gradient'):
        loss_funcs['gradient'] = GradientLoss(rez, number_of_frames, loss_cfg['gradient']['weight'])
    if loss_cfg.get('sparsity'):
        loss_funcs['sparsity'] = SparsityLoss(loss_cfg['sparsity']['weight'])
    if loss_cfg.get('alpha_bootstrapping'):
        loss_funcs['alpha_bootstrapping'] = AlphaBootstrappingLoss(loss_cfg['alpha_bootstrapping']['weight'])
    if loss_cfg.get('alpha_reg'):
        loss_funcs['alpha_reg'] = AlphaRegLoss(loss_cfg['alpha_reg']['weight'])
    if loss_cfg.get('flow_alpha'):
        loss_funcs['flow_alpha'] = FlowAlphaLoss(rez, number_of_frames, loss_cfg['flow_alpha']['weight'])
    if loss_cfg.get('optical_flow'):
        loss_funcs['optical_flow'] = FlowMappingLoss(rez, number_of_frames, loss_cfg['optical_flow']['weight'])
    if loss_cfg.get('rigidity'):
        loss_funcs['rigidity'] = RigidityLoss(
            rez, number_of_frames,
            loss_cfg['rigidity']['derivative_amount'],
            loss_cfg['rigidity']['weight'])
    if loss_cfg.get('global_rigidity'):
        loss_funcs['global_rigidity'] = list()
        for i in range(len(loss_cfg['global_rigidity']['weight'])):
            loss_funcs['global_rigidity'].append(
                RigidityLoss(
                    rez, number_of_frames,
                    loss_cfg['global_rigidity']['derivative_amount'],
                    loss_cfg['global_rigidity']['weight'][i]))
    if loss_cfg.get('residual_reg'):
        loss_funcs['residual_reg'] = ResidualRegLoss(loss_cfg['residual_reg']['weight'])
    if loss_cfg.get('residual_consistent'):
        loss_funcs['residual_consistent'] = ResidualConsistentLoss(rez, number_of_frames, loss_cfg['residual_consistent']['weight'])

    if not load_checkpoint:
        logging.info('Pre-training loop begins.')
        for i, model_F_mapping in enumerate(model_F_mappings):
            if model_F_mapping.pretrain:
                pre_train_mapping(
                    model_F_mapping, number_of_frames, uv_mapping_scales[i],
                    resx=resx, resy=resy, rez=rez,
                    device=device, pretrain_iters=pretrain_iter_number)
    else:
        init_file = torch.load(checkpoint_path)
        model_F_mappings.load_state_dict(init_file["model_F_mappings_state_dict"])
        model_alpha.load_state_dict(init_file["model_F_alpha_state_dict"])
        optimizer_all.load_state_dict(init_file["optimizer_all_state_dict"])
        start_iteration = init_file["iteration"] + 1

    jif_all = get_tuples(number_of_frames, video_frames)

    # Start training!
    logging.info('Main training loop begins.')
    for iteration in range(start_iteration, iters_num+1):
        timer.start()
        losses = dict()

        # randomly choose indices for the current batch
        inds_foreground = torch.randint(
            jif_all.shape[1], (np.int64(samples * 1.0), 1))

        jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)

        # normalize coordinates to be in [-1, 1]
        xyt_current = torch.cat((
            jif_current[0] / (rez / 2) - 1,
            jif_current[1] / (rez / 2) - 1,
            jif_current[2] / (number_of_frames / 2) - 1
        ), dim=1).to(device)  # size (batch, 3)

        uvs, rgb_residuals, rgb_textures = zip(*[i(xyt_current, True, True) for i in model_F_mappings])

        alpha = model_alpha(xyt_current)

        # reconstruct final colors
        rgb_output = sum([(rgb_textures[i] * rgb_residuals[i]) * alpha[:, [i]] for i in range(len(rgb_textures))])

        # RGB loss
        if loss_funcs.get('rgb'):
            rgb_GT = video_frames[jif_current[1], jif_current[0], :, jif_current[2]].squeeze(1).to(device)
            losses['rgb'] = loss_funcs['rgb'](rgb_GT, rgb_output)

        # gradient loss
        if loss_funcs.get('gradient'):
            losses['gradient'] = loss_funcs['gradient'](
                video_frames_dx, video_frames_dy,
                jif_current, rgb_output,
                device, model_F_mappings, model_alpha)

        # sparsity loss
        if loss_funcs.get('sparsity'):
            losses['sparsity'] = loss_funcs['sparsity'](rgb_textures, alpha)

        # alpha bootstrapping loss
        if loss_funcs.get('alpha_bootstrapping'):
            if iteration <= loss_cfg['alpha_bootstrapping']['stop_iteration']:
                alpha_GT = mask_frames[jif_current[1], jif_current[0], jif_current[2]].squeeze(1).to(device)
                losses['alpha_bootstrapping'] = loss_funcs['alpha_bootstrapping'](alpha_GT, alpha)

        # alpha regularization loss
        if loss_cfg['alpha_reg']['weight'] != 0:
            losses['alpha_reg'] = loss_funcs['alpha_reg'](alpha)

        # optical flow alpha loss
        if loss_funcs.get('flow_alpha'):
            losses['flow_alpha'] = loss_funcs['flow_alpha'](
                optical_flows, optical_flows_mask,
                optical_flows_reverse, optical_flows_reverse_mask,
                jif_current, alpha, device, model_alpha)

        # optical flow loss
        if loss_funcs.get('optical_flow'):
            for i in range(num_of_maps):
                losses['optical_flow_%d'%i] = loss_funcs['optical_flow'](
                    optical_flows, optical_flows_mask,
                    optical_flows_reverse, optical_flows_reverse_mask,
                    jif_current, uvs[i], uv_mapping_scales[i],
                    device, model_F_mappings[i], True, alpha[:, [i]])

        # rigidity loss
        if loss_funcs.get('rigidity'):
            for i in range(num_of_maps):
                losses['rigidity_%d'%i] = loss_funcs['rigidity'](
                    jif_current,
                    uvs[i], uv_mapping_scales[i],
                    device, model_F_mappings[i])

        # global rigidity loss
        if loss_funcs.get('global_rigidity'):
            if iteration <= loss_cfg['global_rigidity']['stop_iteration']:
                for i, funcs in enumerate(loss_funcs['global_rigidity']):
                    losses['global_rigidity_%d'%i] = funcs(
                    jif_current,
                    uvs[i], uv_mapping_scales[i],
                    device, model_F_mappings[i])

        # residual regularization loss
        if loss_funcs.get('residual_reg'):
            for i in range(num_of_maps):
                if model_F_mappings[i].model_residual is not None:
                    losses['residual_reg_%d'%i] = loss_funcs['residual_reg'](rgb_residuals[i])

        # residual consistent loss
        if loss_funcs.get('residual_consistent'):
            for i in range(num_of_maps):
                if model_F_mappings[i].model_residual is not None:
                    losses['residual_consistent_%d'%i] = loss_funcs['residual_consistent'](samples, resx, resy, model_F_mappings[i], device)

        loss = sum(losses.values())

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()

        timer.stop()
        if iteration % logger_cfg['period'] == 0:
            logging.info(f'------------------ {results_folder.name} ------------------')
            logging.info('Iteration: %06d' % iteration)

            if logger_cfg['log_time']:
                logging.info('------ time ------')
                logging.info('overall average: %.6f' % timer.average())
                logging.info('last period: %.6f' % timer.last_period(logger_cfg['period']))
                logging.info('ETA: %s' % timer.ETA(iters_num - iteration, logger_cfg['period']))

            if logger_cfg['log_loss']:
                logging.info('------ loss ------')
                logging.info('overall loss: %.6f' % loss)
                for name, l in losses.items():
                    logging.info('%s: %.6f' % (name, l))

            if logger_cfg['log_alpha']:
                logging.info('----------------------------')
                for layer in range(alpha.shape[-1]):
                    logging.info("alpha_mean_%d>0.5: %f" % (layer, alpha[:, [layer]][alpha[:, [layer]]>0.5].mean().detach()))
                    logging.info("alpha_mean_%d<0.5: %f" % (layer, alpha[:, [layer]][alpha[:, [layer]]<0.5].mean().detach()))

        if iteration % eval_cfg['interval'] == 0:
            eval(
                model_F_mappings, model_alpha,
                jif_all, video_frames, number_of_frames, rez, eval_cfg['samples_batch'],
                num_of_maps, str(results_folder/('%06d'%iteration)),
                iteration, optimizer_all, device
            )
            model_F_mappings.train()
            model_alpha.train()


if __name__ == "__main__":
    main(config_load(sys.argv[1]))
