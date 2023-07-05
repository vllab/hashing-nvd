results_folder_name = 'results'
maximum_number_of_frames = 100
resx = 768
resy = 432
iters_num = 50000
samples_batch = 10000
data_folder = 'data/bear'
uv_mapping_scales = [0.9, 0.6]
pretrain_iter_number = 100
load_checkpoint = False
checkpoint_path = ''
folder_suffix = 'test'

logger = dict(
    period = 500,
    log_time = True,
    log_loss = True,
    log_alpha = True)

evaluation = dict(
    interval = 5000,
    samples_batch = 10000)

losses = dict(
    rgb = dict(
        weight = 5),
    gradient = dict(
        weight = 1),
    sparsity = dict(
        weight = 1),
    alpha_bootstrapping = dict(
        weight = 2,
        stop_iteration = 10000),
    alpha_reg = dict(
        weight = 0.1),
    flow_alpha = dict(
        weight = 0.05),
    optical_flow = dict(
        weight = 0.01),
    rigidity = dict(
        weight = 0.001,
        derivative_amount = 1),
    global_rigidity = dict(
        weight = [0.005, 0.05],
        stop_iteration = 5000,
        derivative_amount = 100),
    residual_reg = dict(
        weight = 0.5),
    residual_consistent = dict(
        weight = 0.1))

config_xyt = {
    'base_resolution': 16,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'n_levels': 16,
    'otype': 'HashGrid',
    'per_level_scale': 1.25}
config_uv = {
    'base_resolution': 16,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'n_levels': 16,
    'otype': 'HashGrid',
    'per_level_scale': 1.25}
model_mapping = [{
    'model_type': 'EncodingMappingNetwork',
    'pretrain': True,
    'texture': {
        'model_type': 'EncodingTextureNetwork',
        'encoding_config': config_uv},
    'residual': {
        'model_type': 'ResidualEstimator',
        'encoding_config': config_xyt},
    'encoding_config': None,
    'num_layers': 4,
    'num_neurons': 256
}, {
    'model_type': 'EncodingMappingNetwork',
    'pretrain': True,
    'texture': {
        'model_type': 'EncodingTextureNetwork',
        'encoding_config': config_uv},
    'residual': {
        'model_type': 'ResidualEstimator',
        'encoding_config': config_xyt},
    'encoding_config': None,
    'num_layers': 4,
    'num_neurons': 256
}]
alpha = {
    'model_type': 'EncodingAlphaNetwork',
    'encoding_config': config_xyt}
