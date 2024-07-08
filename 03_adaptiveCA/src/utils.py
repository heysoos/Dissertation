import pygame
import torch
import numpy as np
from itertools import product

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w


# Visualization stuff

def WHEEL_permute(cdim_order, direction, channels):
    cdim_order = np.mod(np.add(cdim_order, direction), channels)

    return cdim_order

def print_text(text, font):
    # text: str of whatever it is that needs to be printed
    f_text = font.render(text, 1, pygame.Color("white"))
    f_bg = pygame.Surface((f_text.get_height(),f_text.get_width()))  # the size of your rect
    f_bg.set_alpha(50)                # alpha level
    f_bg.fill((255,255,255))          # this fills the entire surface

    f_surf = pygame.Surface((f_bg.get_height(), f_bg.get_width()))
    f_surf.blit(f_bg, (0, 0))
    f_surf.blit(f_text, (0, 0))
    return f_surf

def update_fps(clock, font):
    # make a surface to print pygame fps
    fps = str(int(clock.get_fps()))
    fps_text = font.render(fps, 1, pygame.Color("white"))
    fps_bg = pygame.Surface((fps_text.get_height(),fps_text.get_width()))  # the size of your rect
    fps_bg.set_alpha(50)                # alpha level
    fps_bg.fill((255,255,255))          # this fills the entire surface

    fps_surf = pygame.Surface((fps_bg.get_height(), fps_bg.get_width()))
    fps_surf.blit(fps_bg, (0, 0))
    fps_surf.blit(fps_text, (0, 0))
    return fps_surf

def show_param_info(param, name):
    # make a surface to print a parameter name and its value
    font = pygame.font.SysFont("Noto Sans", 12)
    info_str = f'{name}: {param:.4f}'
    info_txt = font.render(info_str, 1, pygame.Color("white"))
    info_bg = pygame.Surface((info_txt.get_height(),info_txt.get_width()))  # the size of your rect
    info_bg.set_alpha(50)                # alpha level
    info_bg.fill((255,255,255))           # this fills the entire surface

    info_surf = pygame.Surface((info_bg.get_height(), info_bg.get_width()))
    pos = (0., 0.)
    info_surf.blit(info_bg, pos)
    info_surf.blit(info_txt, pos)
    return info_surf

def blit_mat(screen, mat, pos=(0, 0), size=(50, 50)):
    # make a surface to plot a matrix
    mat_draw = min_max(mat)
    connectivity_mat = pygame.transform.scale(pygame.surfarray.make_surface(mat_draw * 255.), size)
    screen.blit(connectivity_mat, pos)


def WHEEL_permute(cdim_order, direction, num_channels):
    cdim_order = np.mod(np.add(cdim_order, direction), num_channels)

    return cdim_order

def WHEEL_param(param, direction, increment):
    return 10 ** (np.log10(param) + direction * increment)

def min_max(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())

def click(state, rmb, r=5, s=1, upscale=1, brush_toggle=False):
    '''
    left click action
    state: torch.Tensor of size (1, RESX, RESY, CHANNEL)
    r: radius of brush
    s: smoothing / sigma
    upscale: for when the pygame screen is upscaled, pass the value of upscale
    '''
    xcl, ycl = pygame.mouse.get_pos()
    xcl, ycl = int(xcl / upscale), int(ycl / upscale)
    resx, resy = state.shape[-2:]

    # radial blur
    xm, ym = torch.meshgrid(torch.linspace(-1, 1, 2 * r), torch.linspace(-1, 1, 2 * r))
    rm = torch.sqrt(xm ** 2 + ym ** 2).type(torch.double)
    blur = torch.exp(-rm ** 2 / s ** 2)
    blur = torch.where(rm <= 1., blur, 0.)  # circular mask

    # make a list of all tensor coordinates that are affected by clicking
    range_x = range(xcl - r, xcl + r)
    range_y = range(ycl - r, ycl + r)
    coords = list(product(range_x, range_y))
    idx_i = [c[0] % resx for c in coords]
    idx_j = [c[1] % resy for c in coords]

    # determine if its left or right mouse click to change behaviour
    if rmb:
        brush_coeff = -1.
    else:
        brush_coeff = 1.

    # default behaviour
    if not brush_toggle:
        state[:, :, idx_i, idx_j] += brush_coeff * torch.where(rm.reshape(-1).cuda() <= 1.,
                                                1.,
                                                0.
                                                )
    # alternate behaviour (single channel brush)
    else:
        state[:, 0, idx_i, idx_j] -= brush_coeff * (blur.reshape(-1).cuda() + 1e-10) # change temp
        state[0, 1] = torch.clip(state[0, 1], 1e-6) # clip it so it doesn't go negative

    return state