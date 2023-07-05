import skimage.metrics
import torch

import skimage.measure
import os
from PIL import Image
import numpy as np
import imageio

from tqdm import tqdm

# taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
# sample coordinates x,y from image im.
def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

def eval_data_gen(
    model_F_mappings, model_alpha,
    jif_all, number_of_frames, rez, samples_batch,
    num_of_maps, texture_size, device):
    '''
    Given the whole model settings.
    Return:
        - rec_masks: Maximum alpha value sampled of UV map of each layer.
        - xyts: Normalized spatial and temporal location of each frame.
        - alphas: Alpha value of each layer of each frame.
            `[ndarray, ndarray, ...]`, len = number of frames
        - uvs: Color of UV map of each layer of each frame.
            `[[uv1, uv2, ...], [uv1, uv2, ...], ...]`, len = number of frames
        - residuals: Residual value of each layer of each frame, corresponding to video coordinate.
            `[[residual1, residual2, ...], [residual1, residual2, ...], ...]`, len = number of frames
        - rgbs: Color of each layer of each frame, corresponding to video coordinate.
            `[[rgb1, rgb2, ...], [rgb1, rgb2, ...], ...]`, len = number of frames
    '''
    model_F_mappings.eval()
    model_alpha.eval()
    with torch.no_grad():
        rec_x = jif_all[0, :, None] / (rez / 2) - 1.0
        rec_y = jif_all[1, :, None] / (rez / 2) - 1.0
        rec_t = jif_all[2, :, None] / (number_of_frames / 2) - 1.0
        rec_xyt = torch.cat((rec_x, rec_y, rec_t), dim=1)
        batch_xyt = rec_xyt.split(samples_batch, dim=0)
        # init results
        rec_masks = np.zeros((num_of_maps, *texture_size), dtype=np.uint8)
        xyts = list()
        alphas = list()
        uvs = list()
        residuals = list()
        rgbs = list()

        # run eval: split by batch size
        pbar = tqdm(range(number_of_frames), 'Generating')
        progress = 0
        for idx in range(len(batch_xyt)):
            now_xyt = batch_xyt[idx].to(device)
            progress += len(now_xyt) * number_of_frames
            if pbar.n != int(progress / len(rec_xyt)):
                pbar.update(int(progress / len(rec_xyt)) - pbar.n)

            xyts.append(now_xyt.cpu().numpy())
            rec_alpha = model_alpha(now_xyt)
            alphas.append(rec_alpha.cpu().numpy())
            rec_maps, rec_residuals, rec_rgbs = zip(*[i(now_xyt, True, True) for i in model_F_mappings])
            uvs.append(np.stack([i.cpu().numpy() for i in rec_maps]))
            residuals.append(np.stack([i.cpu().numpy() for i in rec_residuals]))
            rgbs.append(np.stack([i.cpu().numpy() for i in rec_rgbs]))
            rec_idxs = [np.clip(np.floor((i * 0.5 + 0.5).cpu().numpy() * 1000).astype(np.int64), 0, 999) for i in rec_maps]
            for i in range(num_of_maps):
                _idx = np.stack((rec_idxs[i][:, 1], rec_idxs[i][:, 0]))
                for d in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    _idx_now = _idx + np.array(d)[:, None]
                    _idx_now[0] = np.clip(_idx_now[0], 0, texture_size[1]-1)
                    _idx_now[1] = np.clip(_idx_now[1], 0, texture_size[0]-1)
                    mask_now = (alphas[-1][..., i] * 255).astype(np.uint8)
                    mask_now = np.max((mask_now, rec_masks[i][_idx_now[0], _idx_now[1]]), axis=0)
                    rec_masks[i][_idx_now[0], _idx_now[1]] = mask_now
        pbar.close()

    # re-split the data by frame number
    xyts = np.split(np.concatenate(xyts), number_of_frames)
    alphas = np.split(np.concatenate(alphas), number_of_frames)
    uvs = np.split(np.concatenate(uvs, axis=1), number_of_frames, axis=1)
    residuals = np.split(np.concatenate(residuals, axis=1), number_of_frames, axis=1)
    rgbs = np.split(np.concatenate(rgbs, axis=1), number_of_frames, axis=1)
    return rec_masks, xyts, alphas, uvs, residuals, rgbs

def eval(
    model_F_mappings, model_alpha,
    jif_all, video_frames, number_of_frames, rez, samples_batch,
    num_of_maps, save_dir,
    iteration, optimizer_all, device, save_checkpoint=True):
    print('Start evaluation.')
    os.makedirs(save_dir, exist_ok=True)
    texture_size = (1000, 1000)
    with torch.no_grad():
        # init results
        texture_maps = list()
        _m = np.array(Image.open('checkerboard.png'))[..., :3]
        edit_textures = [_m for _ in range(num_of_maps)]
        # generate necessary evaluation components
        rec_masks, xyts, alphas, uvs, residuals, rgbs = eval_data_gen(model_F_mappings, model_alpha, jif_all, number_of_frames, rez, samples_batch, num_of_maps, texture_size, device)

        # write results
        print('Synthesizing: textures')
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, texture_size[1]), torch.linspace(-1, 1, texture_size[0]), indexing='ij')
        grid_xy = torch.hstack((grid_y.reshape(-1, 1), grid_x.reshape(-1, 1))).to(device)
        for i in range(num_of_maps):
            texture_map = model_F_mappings[i].model_texture(grid_xy
                ).detach().cpu().numpy().reshape(*texture_size, 3)
            texture_map = (texture_map * 255).astype(np.uint8)
            texture_maps.append(texture_map)
            Image.fromarray(np.concatenate((texture_map, rec_masks[i][..., None]), axis=-1)).save(os.path.join(save_dir, 'tex%d.png'%i))

        print('Synthesizing: alpha masks')
        _write_alpha(save_dir, alphas, video_frames.shape[:2], num_of_maps)
        print('Synthesizing: residuals')
        _write_residual(save_dir, residuals, alphas, video_frames.shape[:2], num_of_maps)
        print('Synthesizing: reconstruction videos')
        psnr, psnr_no_residual = _write_video(save_dir, rgbs, residuals, alphas, video_frames.numpy(), num_of_maps)
        print('Synthesizing: edited videos')
        _write_edited(save_dir, texture_maps, edit_textures, 0.5, uvs, residuals, alphas, video_frames.shape[:2])
        print('PSNR (w/ residual)  = %.6f'%psnr)
        print('PSNR (w/o residual) = %.6f'%psnr_no_residual)
        with open(os.path.join(save_dir, 'PSNR.txt'), 'w') as f:
            f.write('PSNR (w/ residual)  = %.6f\n'%psnr)
            f.write('PSNR (w/o residual) = %.6f\n'%psnr_no_residual)

        # save current model
        if save_checkpoint:
            print('save checkpoint')
            saved_dict = {
                'iteration': iteration,
                'model_F_mappings_state_dict': model_F_mappings.state_dict(),
                'model_F_alpha_state_dict': model_alpha.state_dict(),
                'optimizer_all_state_dict': optimizer_all.state_dict()
            }
            torch.save(saved_dict, '%s/checkpoint' % (save_dir))

def eval_custom_uv(
    model_F_mappings, model_alpha,
    jif_all, number_of_frames, rez, video_frames,
    num_of_maps, save_dir, device, uv_maps, reconstruct):
    texture_size = (1000, 1000)
    video_size = video_frames.shape[:2]
    rec_masks, xyts, alphas, uvs, residuals, _ = eval_data_gen(model_F_mappings, model_alpha, jif_all, number_of_frames, rez, num_of_maps, texture_size, device)
    writer = imageio.get_writer(os.path.join(save_dir, 'custom_edit.mp4'))

    texture_maps = list()
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, texture_size[1]), torch.linspace(-1, 1, texture_size[0]), indexing='ij')
    grid_xy = torch.hstack((grid_y.reshape(-1, 1), grid_x.reshape(-1, 1))).to(device)
    for i in range(num_of_maps):
        texture_map = model_F_mappings[i].model_texture(grid_xy
            ).detach().cpu().numpy().reshape(*texture_size, 3)
        texture_maps.append(texture_map)

    for t in range(number_of_frames):
        rgbs = list()
        for uv, uv_map, texture_map in zip(uvs[t], uv_maps, texture_maps):
            if not reconstruct:
                map = uv_map - texture_map
            else: map = uv_map
            rgb = bilinear_interpolate_numpy(map, (uv[:, 0]*0.5+0.5)*uv_map.shape[1], (uv[:, 1]*0.5+0.5)*uv_map.shape[0])
            rgbs.append(rgb)
        rgb_all = sum([rgbs[i] * residuals[t][i] * alphas[t][..., [i]] for i in range(num_of_maps)])
        if not reconstruct:
            rgb_all += video_frames[..., t].numpy().reshape(-1, 3)
        writer.append_data(np.clip((rgb_all.reshape(*video_size, 3) * 255), 0, 255).astype(np.uint8))
    writer.close()

def _write_alpha(save_dir, alphas, video_size, num_layers):
    writers = [imageio.get_writer(os.path.join(save_dir, 'alpha%d.mp4'%i), fps=10) for i in range(num_layers)]
    for alpha in alphas:
        alpha = alpha.reshape(*video_size, num_layers)
        for i in range(num_layers):
            writers[i].append_data((alpha[..., [i]] * 255).astype(np.uint8))
    for i in writers: i.close()

def _write_residual(save_dir, residuals, alphas, video_size, num_layers):
    writers = [imageio.get_writer(os.path.join(save_dir, 'residual%d.mp4'%i), fps=10) for i in range(num_layers)]
    for alpha, residual in zip(alphas, residuals):
        for i in range(num_layers):
            writers[i].append_data(np.clip(residual[i]*128, 0, 255).astype(np.uint8).reshape(*video_size, 3))
    for i in writers: i.close()

def _write_video(save_dir, rgbs, residuals, alphas, video_frames, num_layers, write_compare=True):
    writer = imageio.get_writer(os.path.join(save_dir, 'rec.mp4'), fps=10)
    writer_no_residual = imageio.get_writer(os.path.join(save_dir, 'rec_no_residual.mp4'), fps=10)
    if write_compare:
        writer_compare = imageio.get_writer(os.path.join(save_dir, 'comp.mp4'), fps=10)
    psnr = np.zeros((len(rgbs), 1))
    psnr_no_residual = np.zeros((len(rgbs), 1))
    for t, (rgb, residual, alpha) in enumerate(zip(rgbs, residuals, alphas)):
        output_res = sum([(rgb[i] * residual[i]) * alpha[..., [i]] for i in range(num_layers)]).reshape(video_frames.shape[:-1])
        writer.append_data((np.clip(output_res, 0, 1) * 255).astype(np.uint8))
        psnr[t] = skimage.metrics.peak_signal_noise_ratio(
            video_frames[:, :, :, t],
            output_res,
            data_range=1)
        output = sum([rgb[i] * alpha[..., [i]] for i in range(num_layers)]).reshape(video_frames.shape[:-1])
        writer_no_residual.append_data((np.clip(output, 0, 1) * 255).astype(np.uint8))
        psnr_no_residual[t] = skimage.metrics.peak_signal_noise_ratio(
            video_frames[:, :, :, t],
            output,
            data_range=1)
        if write_compare:
            comp = np.empty((output.shape[0]*2, output.shape[1]*2, 3))
            # GT: top-left
            comp[:output.shape[0], :output.shape[1]] = video_frames[..., t]
            # w/ residual: top-right
            comp[:output.shape[0], output.shape[1]:] = output_res
            # w/o residual: bottom-left
            comp[output.shape[0]:, :output.shape[1]] = output
            # residual only: bottom-right
            comp[output.shape[0]:, output.shape[1]:] = sum([residual[i] * alpha[..., [i]] for i in range(num_layers)]).reshape(video_frames.shape[:-1]) * 0.5
            writer_compare.append_data((np.clip(comp, 0, 1) * 255).astype(np.uint8))
    writer.close()
    writer_no_residual.close()
    if write_compare: writer_compare.close()
    return psnr.mean(), psnr_no_residual.mean()

def _write_edited(save_dir, maps1, maps2, ratio, uvs, residuals, alphas, video_size):
    writer_all = imageio.get_writer(os.path.join(save_dir, 'edited_all.mp4'), fps=10)
    writer_all_residual = imageio.get_writer(os.path.join(save_dir, 'edited_all_+residual.mp4'), fps=10)
    rgb_all = np.zeros((len(uvs), *video_size, 3))
    residual_all = np.zeros((len(uvs), *video_size, 3))
    for i in range(len(maps1)):
        im = np.clip(np.where(
            maps2[i], maps1[i] * ratio + maps2[i] * (1-ratio), maps1[i]
        ), 0, 255)
        writer = imageio.get_writer(os.path.join(save_dir, 'edited%d.mp4'%i), fps=10)
        writer_residual = imageio.get_writer(os.path.join(save_dir, 'edited%d+residual.mp4'%i), fps=10)
        for t, (uv, alpha) in enumerate(zip(uvs, alphas)):
            rgb = bilinear_interpolate_numpy(im, (uv[i][:, 0]*0.5+0.5)*im.shape[1], (uv[i][:, 1]*0.5+0.5)*im.shape[0])
            writer.append_data((rgb).reshape(*video_size, 3).astype(np.uint8))
            writer_residual.append_data(np.clip((rgb * residuals[t][i]), 0, 255).reshape(*video_size, 3).astype(np.uint8))
            rgb_all[t] += (rgb*alpha[..., [i]]).reshape(*video_size, 3)
            residual_all[t] += ((rgb * residuals[t][i]) * alpha[..., [i]]).reshape(*video_size, 3)
        writer.close()
        writer_residual.close()
    for t in range(len(uvs)):
        writer_all.append_data(rgb_all[t].astype(np.uint8))
        writer_all_residual.append_data(np.clip(residual_all[t], 0, 255).astype(np.uint8))
    writer_all.close()
    writer_all_residual.close()
