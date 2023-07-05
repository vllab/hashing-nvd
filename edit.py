import torch
import numpy as np
from pathlib import Path
from networks import build_network
from unwrap_utils import load_input_data, get_tuples
from config_utils import config_load
from evaluate import eval_custom_uv
import argparse
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config_filename, checkpoint_filename, custom_uv_paths, reconstruct):
    config = config_load(config_filename)
    maximum_number_of_frames = config["maximum_number_of_frames"]
    resx = np.int64(config["resx"])
    resy = np.int64(config["resy"])
    custom_uvs = [np.array(Image.open(path).convert('RGB')) / 255 for path in custom_uv_paths]

    # optionally it is possible to load a checkpoint

    # a data folder that contains folders named "[video_name]","[video_name]_flow","[video_name]_maskrcnn"
    data_folder = Path(config["data_folder"])
    results_folder_name = config["results_folder_name"] # the folder (under the code's folder where the experiments will be saved.
    folder_suffix = config["folder_suffix"] # for each experiment folder (saved inside "results_folder_name") add this string

    config_mappings = config["model_mapping"]
    config_alpha = config["alpha"]

    vid_name = data_folder.name
    vid_root = data_folder.parent
    results_folder = Path(
        f'./{results_folder_name}/{vid_name}_{folder_suffix}')
    _, video_frames, _, _, _, _, _, _ = load_input_data(
        resy, resx, maximum_number_of_frames, data_folder, True,  True, vid_root, vid_name)
    number_of_frames = video_frames.shape[3]
    jif_all = get_tuples(number_of_frames, video_frames)
    rez = np.maximum(resx, resy)

    num_of_maps = len(config_mappings)

    model_F_mappings = torch.nn.ModuleList()
    for config_mapping in config_mappings:
        model_F_mappings.append(
            build_network(device=device, **config_mapping))

    model_alpha = build_network(
        num_of_maps=num_of_maps,
        device=device,
        **config_alpha)

    init_file = torch.load(checkpoint_filename)
    model_F_mappings.load_state_dict(init_file["model_F_mappings_state_dict"])
    model_alpha.load_state_dict(init_file["model_F_alpha_state_dict"])
    start_iteration = init_file["iteration"]
    save_dir = str(results_folder/('%06d'%start_iteration))
    eval_custom_uv(
        model_F_mappings, model_alpha,
        jif_all, number_of_frames, rez, video_frames,
        num_of_maps, save_dir, device, custom_uvs, reconstruct
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to the config file.', type=str)
    parser.add_argument('checkpoint', help='Path to the checkpoint.', type=str)
    parser.add_argument('custom_uvs', help='List of custom UV map (1000x1000) path.', nargs='+', type=str)
    parser.add_argument('--reconstruct', '-R', help='Editing via reconstruction.', action='store_true')
    args = parser.parse_args()
    main(args.config_file, args.checkpoint, args.custom_uvs, args.reconstruct)
