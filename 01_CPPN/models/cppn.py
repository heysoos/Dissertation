import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm


class CPPN_block(nn.Module):
    """
    A linear layer
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=True)
        self.afunc = nn.Tanh()

    def forward(self, x):
        x = self.lin(x)
        x = self.afunc(x)
        return x


class Rule(nn.Module):
    def __init__(self, dim_in, dim_z, dim_c, net_size, projection_config=None):
        super().__init__()

        ######################## Regular coordinates ########################
        self.in_l = [nn.Linear(1, net_size[0], bias=False).cuda() for i in range(dim_in - 1)]
        self.in_l.append(nn.Linear(dim_z, net_size[0], bias=False).cuda())
        self.in_l = nn.ModuleList(self.in_l)
        #####################################################################

        ######################### Fourier Features ##########################
        # this is some stuff shown to me by Mohammad Bashiri introduced to me for fourier features
        """
        Tancik, Matthew, et al. "Fourier features let networks learn high frequency functions in low 
        dimensional domains." Advances in Neural Information Processing Systems 33 (2020): 7537-7547.
        """
        if projection_config is not None:
            self.projection_config = projection_config
            project_dims, project_scales = projection_config
            # self.projection_mat = nn.Parameter(project_scales * torch.randn(project_dims, dim_in - 1))
            # self.projection_bias = nn.Parameter(project_scales * torch.randn(project_dims * 2))
            # self.in_coords = nn.Linear(2 * project_dims, net_size[0], bias=False).cuda()
            # self.in_z = nn.Linear(dim_z, net_size[0], bias=False).cuda()

            # self.projection_layers = [nn.Linear(project_scales[i] * torch.randn(project_dims[i], dim_in - 1))
            #                           for i in range(len(project_scales))]

            self.amplitudes = nn.Parameter(torch.randn(2, 1, 1, project_dims[-1]) * project_scales[-1])
            self.frequencies = nn.Parameter(torch.randn(2, 1, 1, project_dims[-1]) * project_scales[-1])
            self.phases = nn.Parameter(torch.randn(2, 1, 1, project_dims[-1]))
            # projection_seq does a random linear transform of the input coordinates to be multipled then by sin/cos
            self.projection_seq = [nn.Linear(dim_in - 1, project_dims[0])]
            self.projection_seq[0].weight.data.normal_(0, std=project_scales[0])
            self.projection_seq[0].bias.data.normal_(0, std=project_scales[0])
            # self.projection_seq.append(nn.Tanh())

            for i in range(len(project_dims) - 1):
                self.projection_seq.append(nn.Linear(project_dims[i], project_dims[i+1]))

                self.projection_seq[-1].weight.data.normal_(0, std=project_scales[i + 1])
                self.projection_seq[-1].bias.data.normal_(0, std=project_scales[i + 1])

                # self.projection_seq.append(nn.Tanh())
            self.projection_seq = nn.Sequential(*self.projection_seq)
            # do something with projection scales here/apply weights:

            self.in_coords = nn.Linear(2 * project_dims[-1], net_size[0], bias=False).cuda()
            self.in_z = nn.Linear(dim_z, net_size[0], bias=False).cuda()
        #####################################################################

        self.seq = []
        #         self.bnls = []
        for i in range(len(net_size) - 1):
            self.seq.append(CPPN_block(in_channels=net_size[i], out_channels=net_size[i + 1]))
        self.seq.append(nn.Linear(net_size[-1], dim_c, bias=False))
        self.seq.append(nn.Sigmoid())
        self.seq = nn.ModuleList(self.seq)

        self.seq.apply(weights_init)




class CPPN(nn.Module):
    def __init__(self, net_size=[32, 32, 32], dim_in=6, dim_z=16, dim_c=3, scale=5, res=512,
                 projection_config=None, use_fourier_features=False):
        super().__init__()

        self.dim_c = dim_c
        self.dim_z = dim_z
        self.init_grid(scale, res, dim_z)  # init self.coords

        self.dim_in = dim_in

        self.rule = Rule(self.dim_in, dim_z, self.dim_c, net_size, projection_config=projection_config)
        self.use_fourier_features = use_fourier_features

        # self.apply(weights_init)

    def _coordinates(self, scale, xres, yres, z, flatten=True, batch_size=1, num_nulls=3):

        # check if z has the same batch dim as batch_size

        z = z.unsqueeze(1)

        # generate X, Y, R coordinate grid
        aspect_ratio = xres / yres
        xv = np.linspace(-scale, scale, xres) * aspect_ratio
        yv = np.linspace(-scale, scale, yres)
        X, Y = np.meshgrid(xv, yv)
        R = np.sqrt(X ** 2 + Y ** 2)

        # placeholder matrices that can be replaced with images or what-have-you....
        null = 0 * R
        nulls = [null for i in range(num_nulls)]

        spacecoords = [X, Y, R] + nulls



        # only kept unflattened for hires images
        if flatten:
            spacecoords = [c.reshape(1, -1, 1) for c in spacecoords]

        # tile for batch size
        spacecoords = [np.tile(c, (batch_size, 1, 1)) for c in spacecoords]

        spacecoords = [torch.cuda.FloatTensor(c) for c in spacecoords]

        return spacecoords + [z]

    def init_grid(self, scale, res, dim_z, z=None, num_nulls=3):
        if z is None:
            z = torch.zeros(dim_z)
        self.coords = [coord for coord in self._coordinates(scale, res, res, z, num_nulls)]

    def reinit(self, empty_cache=False):

        if empty_cache:
            torch.cuda.empty_cache()

        self.rule.seq.apply(weights_init)
    def forward(self, coords):

        if coords is None:
            coords = self.coords

        # coordinates -> fourier features
        if self.use_fourier_features:
            U = self.fourier_features(coords)

        # regular coordinate inputs
        else:
            U = [self.rule.in_l[i](coord) for i, coord in enumerate(coords)]
            U = sum(U)

        # linear layers
        out = nn.Sequential(*self.rule.seq)(U)

        return out

    def fourier_features(self, coords):
        U = self.rule.projection_seq(torch.cat(coords[:-1], dim=-1))
        # U = torch.cat([
        #     torch.cos(U * self.rule.frequencies[0] + self.rule.phases[0]),
        #     torch.sin(U * self.rule.frequencies[1] + self.rule.phases[1])],
        #     dim=-1)
        U = torch.cat([
            self.rule.amplitudes[0] * torch.cos(U * self.rule.frequencies[0] + self.rule.phases[0]),
            self.rule.amplitudes[1] * torch.sin(U * self.rule.frequencies[1] + self.rule.phases[1])],
            dim=-1)
        U = self.rule.in_coords(U) + self.rule.in_z(coords[-1])

        return U


class Sampler():
    def __init__(self, res=512, scale=5):
        self.res = res
        self.scale = scale

    def generate_img(self, cppn, z, scale=None, xres=None, yres=None, coords=None, tensor=False):

        with torch.no_grad():
            if scale is None:
                scale = self.scale
            if xres is None or yres is None:
                xres = self.res
                yres = self.res
            if coords is None:
                coords = cppn._coordinates(scale, xres, yres, z)

            out = cppn.forward(coords)
            if not tensor:
                out = out.cpu().numpy()


        return out.reshape(yres, xres, -1)

    def imshow(self, x):

        img = x

        dim_c = img.shape[2]

        fig, ax = plt.subplots(figsize=(9, 16))

        if dim_c == 1:
            img = img[:, :, 0]
            ax.imshow(img, cmap='Greys')
        else:
            ax.imshow(img)
        ax.set_axis_off()
        plt.show()

    def generate_hires(self, cppn, z, scale=None, x_dim=256, y_dim=256, x_reps=16, y_reps=9, coords=None):

        if scale is None:
            scale = self.scale

        x_dim_big = x_dim * x_reps
        y_dim_big = y_dim * y_reps

        with torch.no_grad():
            if coords is None:
                coords = cppn._coordinates(scale, x_dim_big, y_dim_big, z, flatten=False)

            # just expect all input coords to be non-flattened
            #             else:
            #                 coords = [coord.reshape(-1, y_dim_big, x_dim_big) for coord in coords]

            z = coords[-1]
            coords = coords[:-1]

            out = np.zeros((y_dim_big, x_dim_big, cppn.dim_c))
            for ix in tqdm(range(x_reps)):
                x_start = ix * x_dim
                x_end = x_start + x_dim

                for iy in range(y_reps):
                    y_start = iy * y_dim
                    y_end = y_start + y_dim

                    coords_small = [coord[:, y_start:y_end, x_start:x_end].reshape(1, -1, 1) for coord in coords]
                    coords_small.append(z)

                    # img_section = cppn.forward(coords_small, x_dim, y_dim).reshape(y_dim, x_dim, cppn.dim_c)
                    img_section = cppn.forward(coords_small).reshape(y_dim, x_dim, cppn.dim_c)

                    out[y_start:y_end, x_start:x_end, :] = img_section.cpu().numpy()

        return out


###################### TRAINING FUNCTION ######################

## TODO: apparently doesn't work anymore?
def multiscale_targets(target, coords, res, batch_size, big_factor):
    '''
    Generates a set of targets and their associated coordinate grid randomly across the batch dimension.

    big_factor: value between [0., 1.] that controls how large the crops of the target image tend to be.
                - a value of 1 means the images will not be cropped
                - a value of 0 means the iamges will be randomly cropped across all possible window sizes
    '''
    shape = np.array(target).shape
    target_max = np.array(target).max()
    # crop size control (kind of fucky)
    buffer = int((shape[0] - res[0] - 1) * big_factor), int((shape[1] - res[1] - 1) * big_factor)
    TARGETS = []
    S = []
    T = []
    for i in range(batch_size):

        # random top left index for bounding box
        tl_p = (np.random.randint(0, shape[0] - res[0] - buffer[0]),
                np.random.randint(0, shape[1] - res[1] - buffer[1])
                )
        # random bottom right index w.r.t. model resolution and target resolution
        delta_x_max = (shape[1] - tl_p[1])
        delta_y_max = delta_x_max * shape[0] / shape[1]  # constraint due to aspect ratio / other dim
        max_y = np.min([tl_p[0] + delta_y_max + 1, shape[0]])  # constraint due to image boundries

        # control size of crops with one parameter
        # (this is kind of fucky, should fix, look at 2d hist of x,y)
        buffery2 = int((max_y - tl_p[0] - res[0] - 1) * (big_factor))
        min_y = tl_p[0] + res[0] + buffery2

        #         print(f'(x1, y1): ({tl_p[0]:.2f}. {tl_p[1]:.2f})')
        if int(min_y) >= int(max_y):
            print(f'y2: [{min_y:.2f} : {max_y:.2f}]')
            print(f'x1: {tl_p[1]}, delta_x_max: {delta_x_max:.2f}')
            print(f'y1: {tl_p[0]}. delta_y_max: {delta_y_max:.2f}')
        # bottom right coordinates
        y2 = np.random.randint(min_y, max_y)
        x2 = int(tl_p[1] + (y2 - tl_p[0] + 1) * ((shape[1] - 1) / shape[0]))

        br_p = (y2, x2)

        # calculate scale/translation factors to be used to shift cppn input coordinates
        s = ((y2 - tl_p[0]) / (shape[0] - 1), (x2 - tl_p[1]) / (shape[1] - 1))  # L*/L
        t = (2 * (tl_p[0] / (shape[0] - 1) - 0.5),
             2 * (tl_p[1] / (shape[1] - 1) - 0.5))  # maps point_topleft (0, shape) -> (-1, 1)

        points = (tl_p[1], tl_p[0], br_p[1], br_p[0])  # (x1, y1, x2, y2) top left and bottom right points
        #         print(points)

        target_crop = deepcopy(target)
        target_crop = target_crop.crop(points)
        target_crop = target_crop.resize((res[1], res[0]), Image.ANTIALIAS)
        target_crop = np.array(target_crop)

        S.append(s)
        T.append(t)
        TARGETS.append(target_crop / target_max)

    return np.array(TARGETS), np.array(S), np.array(T)

def multiscale_targets_apply(x, img, XRES, YRES, BATCH_SIZE, big_factor, keep_full=True, num_full=1):
    TARGET, S, T = multiscale_targets(img, x, (YRES, XRES), BATCH_SIZE, big_factor)

    # ensures at least one batch dim is the full res picture
    if keep_full:
        target_orig = np.array(img.resize((XRES, YRES), Image.ANTIALIAS))
        target_orig = target_orig / target_orig.max()
        target_orig = np.tile(target_orig, (num_full, 1, 1, 1))

        TARGET[:num_full] = target_orig

        S[:num_full, :] = np.tile(np.array([1, 1]), (num_full, 1))
        T[:num_full, :] = np.tile(np.array([-1, -1]), (num_full, 1))

    x = [xi.detach().cpu().numpy() for xi in x]
    min_x_old = x[0].min(1)[:, 0]
    min_x_scaled = min_x_old * S[:, 1]
    translation_x = T[:, 1] * np.abs(min_x_old) - min_x_scaled
    x[0] = x[0] * S[:, 1][:, None, None] + translation_x[:, None, None]

    min_y_old = x[1].min(1)[:, 0]
    min_y_scaled = min_y_old * S[:, 0]
    translation_y = T[:, 0] * np.abs(min_y_old) - min_y_scaled
    x[1] = x[1] * S[:, 0][:, None, None] + translation_y[:, None, None]

    x[2] = np.sqrt(x[0] ** 2 + x[1] ** 2)
    TARGET = torch.cuda.FloatTensor(TARGET[:, :, :, :3]).reshape(BATCH_SIZE, -1, 3)

    x = [torch.cuda.FloatTensor(xi) for xi in x]

    return x, TARGET

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1)
        #         nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            stdv = 1. / np.sqrt(m.weight.size(1))
            m.bias.data.uniform_(-stdv, stdv)

#             m.bias.data.fill_(0)
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 1.0)
#         if m.bias is not None:
#             m.bias.data.fill_(0)