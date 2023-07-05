import torch
import numpy as np
from pathlib import Path
from networks import build_network
from unwrap_utils import load_input_data, get_tuples
from config_utils import config_load
from evaluate import eval
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg_filename, checkpoint_filename):
    cfg = config_load(cfg_filename)
    maximum_number_of_frames = cfg["maximum_number_of_frames"]
    resx = np.int64(cfg["resx"])
    resy = np.int64(cfg["resy"])

    # optionally it is possible to load a checkpoint

    # a data folder that contains folders named "[video_name]","[video_name]_flow","[video_name]_maskrcnn"
    data_folder = Path(cfg["data_folder"])
    results_folder_name = cfg["results_folder_name"] # the folder (under the code's folder where the experiments will be saved.
    folder_suffix = cfg["folder_suffix"] # for each experiment folder (saved inside "results_folder_name") add this string

    mappings_cfg = cfg["model_mapping"]
    alpha_cfg = cfg["alpha"]

    eval_cfg = cfg["evaluation"]

    vid_name = data_folder.name
    vid_root = data_folder.parent
    results_folder = Path(
        f'./{results_folder_name}/{vid_name}_{folder_suffix}')
    _, video_frames, _, _, _, _, _, _ = load_input_data(
        resy, resx, maximum_number_of_frames, data_folder, True,  True)
    number_of_frames = video_frames.shape[3]
    jif_all = get_tuples(number_of_frames, video_frames)
    rez = np.maximum(resx, resy)

    num_of_maps = len(mappings_cfg)

    model_F_mappings = torch.nn.ModuleList()
    for mapping_cfg in mappings_cfg:
        model_F_mappings.append(
            build_network(device=device, **mapping_cfg))

    model_alpha = build_network(
        num_of_maps=num_of_maps,
        device=device,
        **alpha_cfg)

    init_file = torch.load(checkpoint_filename)
    model_F_mappings.load_state_dict(init_file["model_F_mappings_state_dict"])
    model_alpha.load_state_dict(init_file["model_F_alpha_state_dict"])
    start_iteration = init_file["iteration"]
    save_dir = str(results_folder/('%06d'%start_iteration))
    eval(
        model_F_mappings, model_alpha,
        jif_all, video_frames, number_of_frames, rez, eval_cfg['samples_batch'],
        num_of_maps, save_dir,
        start_iteration, None, torch.device('cuda:0'), False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to the config file.', type=str)
    parser.add_argument('checkpoint', help='Path to the checkpoint.', type=str)
    args = parser.parse_args()
    main(args.config_file, args.checkpoint)
