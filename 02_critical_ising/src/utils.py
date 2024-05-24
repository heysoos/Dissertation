from itertools import product

import numpy as np
import torch
import pygame


'''
These two functions are used to do Left/Right Click mouse actions in the PyGame window.
They expect a torch.Tensor of size (1, WIDTH, HEIGHT, CHANNEL).
'''


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

    if not brush_toggle:
        state[:, 0, idx_i, idx_j] = torch.where(rm.reshape(-1).cuda() <= 1.,
                                                brush_coeff,
                                                state[:, 0, idx_i, idx_j]
                                                )
    else:
        state[:, 1, idx_i, idx_j] -= brush_coeff * (blur.reshape(-1).cuda() + 1e-10) # change temp
        state[0, 1] = torch.clip(state[0, 1], 1e-6) # clip it so it doesn't go negative

    return state


# permute through the channels by scrolling the wheel
# (this function returns the new order of channels)
def WHEEL_permute(cdim_order, direction, channels):
    cdim_order = np.mod(np.add(cdim_order, direction), channels)

    return cdim_order

# change the temperature with the scroll wheel (haven't tested in a while...)
def WHEEL_beta(beta, direction):
    return beta + direction * 0.01

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
    fps = str(int(clock.get_fps()))
    fps_text = font.render(fps, 1, pygame.Color("white"))
    fps_bg = pygame.Surface((fps_text.get_height(),fps_text.get_width()))  # the size of your rect
    fps_bg.set_alpha(50)                # alpha level
    fps_bg.fill((255,255,255))          # this fills the entire surface

    fps_surf = pygame.Surface((fps_bg.get_height(), fps_bg.get_width()))
    fps_surf.blit(fps_bg, (0, 0))
    fps_surf.blit(fps_text, (0, 0))
    return fps_surf