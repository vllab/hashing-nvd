# Hashing Neural Video Decomposition with Multiplicative Residuals in Space-Time

### [Project Page](https://lightbulb12294.github.io/hashing-nvd/) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chan_Hashing_Neural_Video_Decomposition_with_Multiplicative_Residuals_in_Space-Time_ICCV_2023_paper.pdf)

https://github.com/vllab/hashing-nvd/assets/13299616/c4639cb2-0f60-4171-9b8a-ad2e14a06f3a.mp4

## Installation

Our code is compatible and validate with Python 3.9.16, PyTorch 1.13.1, and CUDA 11.7.

```
conda create -n hashing-nvd python=3.9
conda activate hashing-nvd
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib tensorboard scipy  scikit-image tqdm
pip install opencv-python imageio-ffmpeg gdown
CC=gcc-9 CXX=g++-9 python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install easydict
CC=gcc-9 CXX=g++-9 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Directory structures for datasets

```
data
├── <video_name>
│   ├── video_frames
│   │   └── %05d.jpg or %05d.png ...
│   ├── flows
│   │   └── optical flow npy files ...
│   ├── masks
│       ├── <object_0>
│       │   └── %05d.png ...
│       ├── <object_1>
│       │   └── %05d.png ...
│       ⋮
│       └── <object_n>
│           └── %05d.png ...
⋮
```

## Data preparations

### Video frames

The video frames follows the format of [DAVIS](https://davischallenge.org/) dataset. The file type of images should be all either in png or jpg and named as `00000.jpg`, `00001.jpg`, ...

### Preprocess optical flow

We extract the optical flow using [RAFT](https://arxiv.org/abs/2003.12039). The submodule can be linked by the following command:

```
git submodule update --init
cd thirdparty/RAFT/
./download_models.sh
cd ../..
```

To create optical flow for the video, run:

```
python preprocess_optical_flow.py --data-path data/<video_name> --max_long_edge 768
```

The script will automatically generate the corresponding backward and forward optical flow and store the npy files in the right directory.

### Preprocess object masks

We extract the object masks using [Mask-RCNN](https://arxiv.org/abs/1703.06870) via the following script:

```
python preprocess_mask_rcnn.py --data-path data/<video_name> --class_name <class_name> --object_name <object_name>
```

The `class_name` should be one of the COCO class name. It is also possible to use `--class_name anything` to extract the first instance retrieved by Mask-RCNN.

The mask will be stored in `data/<video_name>/masks/<object_name>`. Our implementation also supports decomposition of multiple objects.

## Training

To decompose a video, run:

```
python train.py config/config.py
```

You need to replace the `data_folder` to the folder of your video.

It is also possible to test a certain checkpoint:

```
python test.py <config_file> <checkpoint_file>
```

The config file and checkpoint file will be stored to the assigned result folder.

## Editing

Once the training is complete, the result of a checkpoint will be stored in `<results_folder_name>/<video_name>_<folder_suffix>/<checkpoint_number>`. You can find checkpoint, reconstruction, PSNR report, and other edit videos for debug propose.

You can edit the `tex%d.png` to edit the video. After that, run:

```
python edit.py <config_file> <checkpoint_file> <list of custom textures>
```

The edited video will be generated in the same folder and named as `custom_edit.mp4`.

## Citation

If you find our work useful in your research, please consider citing:

```
@InProceedings{Chan_2023_ICCV,
    author    = {Chan, Cheng-Hung and Yuan, Cheng-Yang and Sun, Cheng and Chen, Hwann-Tzong},
    title     = {Hashing Neural Video Decomposition with Multiplicative Residuals in Space-Time},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7743-7753}
}
```

## Acknowledgement

We thank [Layered Neural Atlases](https://github.com/ykasten/layered-neural-atlases) for using their code implementation as our code base. We modify the code structures to meet our requirements.

This work was supported in part by NSTC grants 111-2221-E-001-011-MY2 and 112-2221-E-A49-100-MY3 of Taiwan. We are grateful to National Center for High-performance Computing for providing computational resources and facilities.
