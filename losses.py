import torch
from torch.nn import functional as F

class RGBLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, GT, pred):
        return(torch.norm(pred - GT, dim=1) ** 2).mean() * self.loss_weight

class GradientLoss():
    def __init__(self, rez, num_of_frames, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight

    def __call__(self,
            video_frames_dx, video_frames_dy,
            jif_current, rgb_output,
            device, model_F_mappings, model_alpha):
        x1yt_current = torch.cat((
            (jif_current[0, :] + 1) / (self.rez / 2) - 1,
            jif_current[1, :] / (self.rez / 2) - 1,
            jif_current[2, :] / (self.num_of_frames / 2) - 1
        ), dim=1).to(device)
        xy1t_current = torch.cat((
            jif_current[0, :] / (self.rez / 2) - 1,
            (jif_current[1, :] + 1) / (self.rez / 2) - 1,
            jif_current[2, :] / (self.num_of_frames / 2) - 1
        ), dim=1).to(device)

        alpha_x1yt = model_alpha(x1yt_current)
        alpha_xy1t = model_alpha(xy1t_current)

        rgb_dx_gt = video_frames_dx[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]].squeeze(1).to(device)
        rgb_dy_gt = video_frames_dy[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]].squeeze(1).to(device)

        uv_x1yts, residual_x1yts, rgb_texture_x1yts = zip(*[i(x1yt_current, True, True) for i in model_F_mappings])
        uv_xy1ts, residual_xy1ts, rgb_texture_xy1ts = zip(*[i(xy1t_current, True, True) for i in model_F_mappings])

        rgb_output_x1yt = sum([(rgb_texture_x1yts[i] * residual_x1yts[i]) * alpha_x1yt[:, [i]] for i in range(len(uv_x1yts))])
        rgb_output_xy1t = sum([(rgb_texture_xy1ts[i] * residual_xy1ts[i]) * alpha_xy1t[:, [i]] for i in range(len(uv_xy1ts))])

        rgb_dx_output = rgb_output_x1yt - rgb_output
        rgb_dy_output = rgb_output_xy1t - rgb_output

        loss = torch.mean((rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2 + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2)
        return loss * self.loss_weight

class SparsityLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, rgb_textures, alpha):
        # map N is the background
        rgb_output_foreground_inv = torch.cat(
            [rgb_textures[i] * (1 - alpha[:, [i]]) for i in range(len(rgb_textures)-1)], dim=1)
        return (torch.norm(rgb_output_foreground_inv, dim=1) ** 2).mean() * self.loss_weight

class AlphaBootstrappingLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, GT, pred):
        return F.binary_cross_entropy(pred, GT) * self.loss_weight

class AlphaRegLoss():
    # regularization: each pixel is more likely to contribute one layer only
    def __init__(self, loss_weight, eps=1e-5):
        self.loss_weight = loss_weight
        self.eps = eps

    def __call__(self, alpha):
        alpha_loss = (-torch.log(alpha.amax(dim=1))).mean()
        return self.loss_weight * alpha_loss

class FlowAlphaLoss():
    def __init__(self, rez, num_of_frames, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight

    def __call__(self,
            of, of_mask, of_rev, of_rev_mask,
            jif_current, alpha,
            device, model_alpha):
        # forward
        xyt_forward_match, indices_forward = get_flow_match(of, of_mask, jif_current, self.rez, self.num_of_frames, True)
        alpha_forward_match = model_alpha(xyt_forward_match.to(device))
        loss_forward = (alpha[indices_forward] - alpha_forward_match).abs().mean()

        # backward
        xyt_backward_match, indices_backward = get_flow_match(of_rev, of_rev_mask, jif_current, self.rez, self.num_of_frames, False)
        alpha_backward_match = model_alpha(xyt_backward_match.to(device))
        loss_backward = (alpha[indices_backward] - alpha_backward_match).abs().mean()

        return (loss_forward + loss_backward) * 0.5 * self.loss_weight

class FlowMappingLoss():
    def __init__(self, rez, num_of_frames, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight

    def __call__(self,
            of, of_mask, of_rev, of_rev_mask,
            jif_current, uv, uv_scale,
            device, model_F_mapping, use_alpha=False, alpha=None):
        # forward
        xyt_forward_match, indices_forward = get_flow_match(of, of_mask, jif_current, self.rez, self.num_of_frames, True)
        uv_forward = uv[indices_forward]
        uv_forward_match = model_F_mapping(xyt_forward_match.to(device))
        loss_forward = (uv_forward_match - uv_forward).norm(dim=1) * self.rez / (2 * uv_scale)

        # backward
        xyt_backward_match, indices_backward = get_flow_match(of_rev, of_rev_mask, jif_current, self.rez, self.num_of_frames, False)
        uv_backward = uv[indices_backward]
        uv_backward_match = model_F_mapping(xyt_backward_match.to(device))
        loss_backward = (uv_backward_match - uv_backward).norm(dim=1) * self.rez / (2 * uv_scale)

        if use_alpha:
            loss_forward = loss_forward * alpha[indices_forward].squeeze()
            loss_backward = loss_backward * alpha[indices_backward].squeeze()

        return (loss_forward.mean() + loss_backward.mean()) * 0.5 * self.loss_weight

class RigidityLoss():
    def __init__(self, rez, num_of_frames, d, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.d = d
        self.loss_weight = loss_weight

    def __call__(self,
            jif_current,
            uv, uv_scale, device, model_F_mapping):
        x_patch = torch.cat((jif_current[0], jif_current[0] - self.d))
        y_patch = torch.cat((jif_current[1] - self.d, jif_current[1]))
        t_patch = torch.cat((jif_current[2], jif_current[2]))
        xyt_p = torch.cat((
            x_patch / (self.rez / 2) - 1,
            y_patch / (self.rez / 2) - 1,
            t_patch / (self.num_of_frames / 2) - 1
        ), dim=1).to(device)

        uv_p = model_F_mapping(xyt_p)
        u_p = uv_p[:, 0].reshape(2, -1) # u(x, y-d, t), u(x-d, y, t)
        v_p = uv_p[:, 1].reshape(2, -1) # v(x, y-d, t), v(x-d, y, t)

        u_p_d = (uv[:, 0].unsqueeze(0) - u_p) * self.rez / 2
        v_p_d = (uv[:, 1].unsqueeze(0) - v_p) * self.rez / 2

        du_dx = u_p_d[1]
        du_dy = u_p_d[0]
        dv_dy = v_p_d[0]
        dv_dx = v_p_d[1]

        J = torch.stack((
            du_dx, du_dy, dv_dx, dv_dy
        ), dim=-1).reshape(-1, 2, 2)
        J = J / uv_scale / self.d
        JtJ = torch.matmul(J.transpose(1, 2), J)

        # 2x2 matrix inverse for faster computation
        A = JtJ[:, 0, 0] + 0.001
        B = JtJ[:, 0, 1]
        C = JtJ[:, 1, 0]
        D = JtJ[:, 1, 1] + 0.001
        JtJinv = torch.stack((
            D, -B, -C, A
        ), dim=-1).reshape(-1, 2, 2) / (A * D - B * C).reshape(-1, 1, 1)

        loss = (JtJ ** 2).sum(dim=[1, 2]).sqrt() + (JtJinv ** 2).sum(dim=[1, 2]).sqrt()

        return loss.mean() * self.loss_weight

class ResidualRegLoss():
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, residual):
        return ((residual - 1) ** 2).mean() * self.loss_weight

class ResidualConsistentLoss():
    # this loss assumes that the camera movement is smooth
    def __init__(self, rez, num_of_frames, loss_weight):
        self.rez = rez
        self.num_of_frames = num_of_frames
        self.loss_weight = loss_weight
        self.t = torch.arange(num_of_frames) / (num_of_frames / 2.0) - 1

    def __call__(self, samples_batch, resx, resy, model_F_mapping, device):
        # patch size = 9, each patch sample 5 different times
        patch_size = 9
        samples_frame = 15
        samples_batch = samples_batch // (patch_size * samples_frame)
        # make sample on edge of the video
        split = torch.randint(samples_batch, (3,)).sort()[0]
        # (0, ?), (resx-1, ?), (?, 0), (?, resy-1)
        x = torch.cat((
            torch.zeros(1).expand(split[0]) + 1,
            torch.zeros(1).expand(split[1]-split[0]) + resx - 2,
            torch.randint(low=1, high=resx-1, size=(samples_batch-split[1],))
        )) / (self.rez / 2) - 1
        y = torch.cat((
            torch.randint(low=1, high=resy, size=(split[1],)),
            torch.zeros(1).expand(split[2]-split[1]) + 1,
            torch.zeros(1).expand(samples_batch-split[2]) + resy - 2
        )) / (self.rez / 2) - 1
        t = torch.randint(self.num_of_frames, (samples_batch,)) / (self.num_of_frames / 2.0) - 1
        xyt = torch.stack((x, y, t), axis=-1).to(device)

        # get the patches
        one = 1 / (self.rez / 2)
        x_patch = torch.stack((
            x-one, x-one, x-one, x, x+one, x+one, x+one, x
        ), axis=-1).ravel()
        y_patch = torch.stack((
            y-one, y, y+one, y+one, y+one, y, y-one, y-one
        ), axis=-1).ravel()
        t_patch = torch.stack((
            t, t, t, t, t, t, t, t
        ), axis=-1).ravel()
        xyt_patch = torch.stack((x_patch, y_patch, t_patch), axis=-1).to(device)

        # get uv
        uv, residual = model_F_mapping(xyt, return_residual=True)
        uv_patch, residual_patch = model_F_mapping(xyt_patch, return_residual=True)
        residual = residual.reshape(samples_batch, 1, 3)
        residual_patch = residual_patch.reshape(samples_batch, -1, 3)

        # sample some frames to be constrained
        tn = torch.randint(self.num_of_frames, (samples_frame, samples_batch)).reshape(-1, 1).to(device) / (self.num_of_frames / 2.0) - 1
        uv = uv.expand(samples_frame, -1, 2).reshape(-1, 2)
        uv_patch = uv_patch.expand(samples_frame, -1, 2).reshape(-1, 2)
        tn_patch = tn.expand((patch_size-1), -1, 1).permute(1, 0, 2).reshape(-1, 1)
        residual_c = model_F_mapping.model_residual(uv.detach(), tn).reshape(samples_frame, samples_batch, 1, 3)
        residual_c_patch = model_F_mapping.model_residual(uv_patch.detach(), tn_patch).reshape(samples_frame, samples_batch, -1, 3)

        # make a mask of areas we can see
        uv = uv.reshape(samples_frame, samples_batch, 2)
        xy_test = xyt.expand(samples_frame, -1, 3).reshape(-1, 3)[:, :2]
        xyt_test = torch.cat((xy_test, tn), dim=-1)
        uv_test = model_F_mapping(xyt_test)
        uv_test = uv_test.reshape(samples_frame, -1, 2)
        # 1: bigger than base, 0: don't compare, -1: smaller than base
        x_mask = torch.cat((
            torch.ones(split[0]), -torch.ones(split[1]-split[0]), torch.zeros(samples_batch-split[1])
        )).expand(samples_frame, -1).to(device)
        y_mask = torch.cat((
            torch.zeros(split[1]), torch.ones(split[2]-split[1]), -torch.ones(samples_batch-split[2])
        )).expand(samples_frame, -1).to(device)
        invisible_mask = torch.zeros(samples_frame, samples_batch, dtype=bool).to(device)
        invisible_mask[x_mask==1] = uv_test[x_mask==1][:, 0] > uv[x_mask==1][:, 0]
        invisible_mask[x_mask==-1] = uv_test[x_mask==-1][:, 0] < uv[x_mask==-1][:, 0]
        invisible_mask[y_mask==1] = uv_test[y_mask==1][:, 1] > uv[y_mask==1][:, 1]
        invisible_mask[y_mask==-1] = uv_test[y_mask==-1][:, 1] < uv[y_mask==-1][:, 1]
        if invisible_mask.any() == False:
            return 0

        # calculate correlation as loss
        residual1 = torch.cat((residual, residual_patch), dim=1).detach() * 255
        residual2 = torch.cat((residual_c, residual_c_patch), dim=2) * 255
        mean1 = residual1.mean(dim=1, keepdim=True)[None]
        mean2 = residual2.mean(dim=2, keepdim=True)
        std1 = residual1.std(dim=1, keepdim=True)[None]
        std2 = residual2.std(dim=2, keepdim=True)
        x1 = residual1 - mean1
        x2 = residual2 - mean2
        loss = 1 - ((x1 * x2).sum(dim=2, keepdim=True) / ((patch_size - 1) * std1 * std2 + 1e-6))
        loss = loss[invisible_mask].mean()
        loss_std = ((std2[invisible_mask] / 255) ** 2).mean()
        return (loss + 16 * loss_std) * self.loss_weight

def get_flow_match(of, of_mask, jif_current, rez, num_of_frames, is_forward):
    next_mask = torch.where(
        of_mask[jif_current[1].ravel(), jif_current[0].ravel(), jif_current[2].ravel()])
    indices = next_mask[0]
    num_next_frames = 2 ** next_mask[1]

    jif_next = jif_current[:, indices, 0]
    next_flows = of[jif_next[1], jif_next[0], :, jif_next[2], next_mask[1]]

    if is_forward == False:
        num_next_frames *= -1
    jif_next_match = torch.stack((
        jif_next[0] + next_flows[:, 0],
        jif_next[1] + next_flows[:, 1],
        jif_next[2] + num_next_frames
    ))
    xyt_next_match = torch.stack((
        jif_next_match[0] / (rez / 2) - 1,
        jif_next_match[1] / (rez / 2) - 1,
        jif_next_match[2] / (num_of_frames / 2) - 1
    )).T

    return xyt_next_match, indices
