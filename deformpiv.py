import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from openpiv import tools, scaling, pyprocess, validation, filters
from UnFlowNet.models import Network, device, estimate
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Basic functions
"""

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


# interpolation with opencv
def remap(img, x, y):
    x, y = np.float32(x), np.float32(y)
    out = cv2.remap(img, x, y, cv2.INTER_CUBIC)  # INTER_LANCZOS4 INTER_CUBIC INTER_LINEAR
    return out


# griddes sparse vector to dense field
def sparse2dense(x, y, u, v, sz):
    dx, dy = np.meshgrid(np.arange(sz[1]), np.arange(sz[0]))
    dx = (dx-x[0,0])/(x[0,1]-x[0,0])
    dy = (dy-y[0,0])/(y[1,0]-y[0,0])
    du = remap(u, dx, dy)
    dv = remap(v, dx, dy)
    return dx, dy, du, dv


# piv kernel with Open PIV package
def openpiv(frame_a, frame_b, winsz=32, overlap=24):
    # process image pair with extended search area piv algorithm.
    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=winsz, overlap=overlap, dt=1.0, search_area_size=winsz, sig2noise_method='peak2peak')
    u1, v1, mask = validation.sig2noise_val(u.copy(), v.copy(), sig2noise, threshold = 1.5 )
    u = u if np.sum(mask)==np.prod(mask.shape) else u1
    v = v if np.sum(mask)==np.prod(mask.shape) else v1
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    # get window centers coordinates
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=winsz, overlap=overlap)
    return x, y, u, v


# piv kernel with optical flow (opencv)
def opticalflow(img1, img2, level=4):
    flow1 = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, level, 33, 11, 9, 1.3, 0)
    flow2 = cv2.calcOpticalFlowFarneback(img2, img1, None, 0.5, level, 33, 11, 9, 1.3, 0)
    flow = (flow1-flow2)/2
    u, v = flow[...,0], flow[...,1]
    x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
    return x, y, u, v


# piv kernel with deep neural networks
""" https://github.com/erizmr/UnLiteFlowNet-PIV
Unsupervised learning of Particle Image Velocimetry
"""
def _loadNet():
    PATH = './UnFlowNet/UnsupervisedLiteFlowNet_pretrained.pt'
    unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
    unliteflownet.eval()
    unliteflownet.to(device)
    print('unliteflownet load successfully.')

unliteflownet = Network()
_loadNet()

def deeppiv1(img1, img2):
    # The input of the network is recommended to be (256, 256)
    assert img1.shape == img2.shape #== (256,256)
    sz = img1.shape
    h, w = sz[0], sz[1]
    x1 = torch.Tensor(img1/255.0).view(1,1,sz[0],sz[1])
    x2 = torch.Tensor(img2/255.0).view(1,1,sz[0],sz[1])

    if img1.shape != (256, 256):
        x1 = F.interpolate(x1, (256, 256), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, (256, 256), mode='bilinear', align_corners=False)

    y_pre = estimate(x1.to(device), x2.to(device), unliteflownet, train=False)
    y_pre = F.interpolate(y_pre, (h, w), mode='bilinear', align_corners=False)

    u = y_pre[0][0].detach()
    v = y_pre[0][1].detach()

    u = u.numpy()
    v = v.numpy()

    x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
    return x, y, u, v


def deeppiv(img1, img2):
    # The input of the network is recommended to be (256, 256) 
    assert img1.shape == img2.shape #== (256,256)
    sz = img1.shape
    h, w = sz[0], sz[1]
    x1 = torch.Tensor(img1/255.0).view(1,1,sz[0],sz[1])
    x2 = torch.Tensor(img2/255.0).view(1,1,sz[0],sz[1])
    
    if img1.shape[0] != 256 or img1.shape[1] != 256: 
        s = 16
        
        # padding zero to 256+n*(256-2s) >= h+2s
        nx, ny = (h+2*s-256-1)//(256-2*s)+2, (w+2*s-256-1)//(256-2*s)+2
        px = 256+(nx-1)*(256-2*s)-h
        py = 256+(ny-1)*(256-2*s)-w

        pad_x1 = F.pad(x1, (s, py-s, s, px-s), "constant", 0)
        pad_x2 = F.pad(x2, (s, py-s, s, px-s), "constant", 0)
        hp1, wp1 = pad_x1.shape[2], pad_x1.shape[3] 
    #     print("padding size", pad_x1.shape)

        # change the shape to B*1*256*256, reorganise the large image to patches
        win_x1 = nn.Unfold(kernel_size=(256, 256), stride=((256-2*s), (256-2*s)))(pad_x1)
        win_x1 = win_x1.permute(2, 0, 1)  # B*N*(w*h)
        win_x1 = win_x1.reshape(win_x1.shape[0], win_x1.shape[1], 256, 256)  # B*N*w*h
        win_x2 = nn.Unfold(kernel_size=(256, 256), stride=((256-2*s), (256-2*s)))(pad_x2)
        win_x2 = win_x2.permute(2, 0, 1)  # B*N*(w*h)
        win_x2 = win_x2.reshape(win_x2.shape[0], win_x2.shape[1], 256, 256)  # B*N*w*h
    #     print("win_x1.shape", win_x1.shape, x1.shape)

        with torch.no_grad():
            torch.cuda.empty_cache()
            y_pre = estimate(win_x1.to(device), win_x2.to(device), unliteflownet, train=False)

        y_pre = y_pre[:, :, s:-s, s:-s]
        # change back
        sz = (nx, ny, 2, y_pre.shape[2], y_pre.shape[3])
        y_pre = y_pre.reshape(sz)
        y_pre = y_pre.permute(2, 0, 3, 1, 4) # B*2*256*256-> 2*B*256*256
        y_pre = y_pre.reshape(2, (256-2*s)*nx, (256-2*s)*ny)

        y_pre = y_pre.detach().numpy()
        u, v = y_pre[0,:,:], y_pre[1,:,:]
        u = u[:h, :w]
        v = v[:h, :w]
        assert u.shape == img1.shape # check the shape
    else:
        with torch.no_grad():
            torch.cuda.empty_cache()
            y_pre = estimate(x1.to(device), x2.to(device), unliteflownet, train=False)
        u = y_pre[0][0].detach().numpy()
        v = y_pre[0][1].detach().numpy()

    x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
    return x, y, u, v


""" image warping with deformation field or velocity field.
"""
def deform(u, v, delta=1, n_iter=10):
    """ Generate deformation field
    Integrates a vector field via scaling and squaring.
    adopted from https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    """
    assert u.shape == v.shape

    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    u, v = u/delta, v/delta
    dx, dy = u/2**n_iter, v/2**n_iter

    for iter in range(n_iter):
        dx_new = dx + remap(dx, x+dx, y+dy)
        dy_new = dy + remap(dy, x+dx, y+dy)
        dx, dy = dx_new, dy_new

    dx, dy = dx*delta, dy*delta
    return dx, dy


def warping(img1, img2, u, v, method='CDI'):
    # FDI: out1(x,y)=img1(x+u, y+v)
    # FDI, CDI, FDDI, CDDI

    assert img1.shape == img2.shape == u.shape == v.shape

    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    u, v = -u, -v
    if method == 'FDI':
        out1= remap(img1, x+u, y+v)
        out2= img2.copy()
    elif method == 'FDI2':
        out1= img1.copy()
        out2= remap(img2, x-u, y-v)
    elif method == 'CDI':
        out1= remap(img1, x+0.5*u, y+0.5*v)
        out2= remap(img2, x-0.5*u, y-0.5*v)
    elif method == 'FDDI':
        dx, dy = deform(u, v, delta=1)
        out1= remap(img1, x+dx, y+dy)
        out2= img2.copy()
    elif method == 'FDDI2':
        dx, dy = deform(-u, -v, delta=1)
        out1= img1.copy()
        out2= remap(img2, x+dx, y+dy)
    elif method == 'CDDI':
        dx1, dy1 = deform(0.5*u, 0.5*v, delta=1)
        dx2, dy2 = deform(-0.5*u, -0.5*v, delta=1)
        out1= remap(img1, x+dx1, y+dy1)
        out2= remap(img2, x+dx2, y+dy2)
    else:
        raise NotImplementedError
    return out1, out2


# Our wrapper for iterative deformation PIV
class DeformPIV():
    def __init__(self, config):
        self._c = config
        assert self._c.pivmethod in ['opticalflow', 'openpiv', 'deeppiv']
        assert self._c.deform in ['FDI', 'FDI2', 'CDI', 'FDDI', 'FDDI2', 'CDDI']

        self.onepass = eval(self._c.pivmethod) # opticalflow1
        self.warping = self._c.deform

    def compute(self, image1, image2, u=None, v=None):
        # obtain the initial vector field
        if u is not None:
            assert image1.shape == image2.shape == u.shape == v.shape
        else:
            assert image1.shape == image2.shape
            x, y, u, v = self.onepass(image1, image2)

        # iterative operation
        for iter in range(self._c.runs):
            # using a blur trick to make the iteration stable
            smooth_k = 3 if u.shape != image1.shape else 9
            for i in range(2):
                u = cv2.blur(u, (smooth_k,smooth_k)) # using 19
                v = cv2.blur(v, (smooth_k,smooth_k))

            # sparse vector to dense field, if needed
            if u.shape != image1.shape:
                xd, yd, ud, vd = sparse2dense(x, y, u, v, image1.shape)
            else:
                ud, vd = u, v


            # image warping
            # warp = self.warping if iter > -2 else 'CDI'
            img1, img2 = warping(image1, image2, ud, vd, self.warping)

            # update the estimation
            if self._c.pivmethod == 'opticalflow':
                x, y, du, dv = self.onepass(img1, img2, level=0)
            elif self._c.pivmethod == 'openpiv':
                # x, y, du, dv = self.onepass(img1, img2, winsz=16, overlap=8)
                x, y, du, dv = self.onepass(img1, img2)
            elif self._c.pivmethod == 'deeppiv':
                x, y, du, dv = self.onepass(img1, img2)
            else:
                raise NotImplementedError

            if u.shape !=du.shape:
                u = remap(ud, x, y)
                v = remap(ud, x, y)

            u, v = u+du, v+dv
            # print('The mean of increasing amplitude:',np.mean(np.sqrt(du**2+dv**2)))

        # # Debug plot, save the images as .png files
        # for k, img in enumerate([image1, img1, img2, image2]):
        #     fig = plt.figure(figsize=(12,12))
        #     plt.imshow(img)
        #     plt.savefig(f"{iter}img{k+1}.png")
        #     plt.close(fig)
        return x, y, u, v
