"""Particle image generator
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import torch        
from flows import lamb_oseen, sin_flow


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def erf(x):
    """
    It's hard to believe we have to wrapper the erf function from pytorch
    """
    x = torch.tensor(x)
    y = torch.erf(x).cpu().numpy()
    return y


def add_particle2(img_sz, particle):
    """
    Using the erf function to synthesis the particle images
    """
    image = np.zeros(img_sz)
    u, v = np.meshgrid(np.arange(img_sz[1]),np.arange(img_sz[0]))
    
    x_s = np.reshape(particle.x, (-1,1))
    y_s = np.reshape(particle.y, (-1,1))
    dp_s = np.reshape(particle.d, (-1,1))
    intensity_s = np.reshape(particle.i, (-1,1))
    dp_nominal=particle.nd

    for x, y, dp, intensity in zip(x_s, y_s, dp_s, intensity_s):
        ind_x1 = np.int(min(max(0, x-3*dp-2), img_sz[1]-6*dp-3))
        ind_y1 = np.int(min(max(0, y-3*dp-2), img_sz[0]-6*dp-3))
        ind_x2 = ind_x1 + np.int(6*dp+3)
        ind_y2 = ind_y1 + np.int(6*dp+3)
       
        lx = u[ind_y1:ind_y2, ind_x1:ind_x2] -x
        ly = v[ind_y1:ind_y2, ind_x1:ind_x2] -y
        b = dp/np.sqrt(8) # from the Gaussian intensity profile assumption

        img =(erf((lx+0.5)/b)-erf((lx-0.5)/b))*(erf((ly+0.5)/b)-erf((ly-0.5)/b))
        img = img*intensity  
        
        image[ind_y1:ind_y2, ind_x1:ind_x2] =  image[ind_y1:ind_y2, ind_x1:ind_x2]+ img
    
    b_n = dp_nominal/np.sqrt(8)
    partition = 1.5*(erf(0.5/b_n)-erf(-0.5/b_n))**2
    image = np.clip(image/partition,0,1.0) 
    image = image*255.0
    image  = np.round(image)
    return image

def gen_image_pair(config):
    # settings 
    img_sz = (config.img_sz[0]+50,config.img_sz[1]+50) # add boundary 
    ppp = config.ppp
    dp, d_std = config.dp, config.d_std
    i_std = config.i_std
    miss_ratio = config.miss_ratio

    # generate particles' parameters
    p1, p2= AttrDict(), AttrDict()
    p1.num = p2.num = np.round(ppp*np.prod(img_sz)).astype(np.int)
    p1.nd = p2.nd = dp
    p1.x = p2.x = np.random.uniform(0, img_sz[1], p1.num)
    p1.y = p2.y = np.random.uniform(0, img_sz[0], p1.num)
    p1.d = p2.d = np.abs(np.random.randn(p1.num)*d_std+ dp)
    p1.i = p2.i = np.random.randn(p1.num)*i_std+ 0.85

    # generate the flow field
    gx, gy = np.meshgrid(np.arange(img_sz[1]),np.arange(img_sz[0]))
    if config.style=='sin_flow':
        _, _, p1.x, p1.y, p2.x, p2.y = sin_flow(p1.x, p2.y, scale=config.scale)
        u, v, _, _, _, _ = sin_flow(gx, gy, scale=config.scale)
    elif config.style== 'lamb_oseen':
        _, _, x1, y1, x2, y2 = lamb_oseen(p1.x-img_sz[1]/2, p2.y-img_sz[0]/2, Gamma=config.gamma)
        p1.x, p1.y = x1+img_sz[1]/2, y1+img_sz[0]/2
        p2.x, p2.y = x2+img_sz[1]/2, y2+img_sz[0]/2
        u, v, _, _, _, _= lamb_oseen(gx-img_sz[1]/2, gy-img_sz[0]/2, Gamma=config.gamma)
 
    # generate images
    img1 = add_particle2(img_sz,p1)
    img2 = add_particle2(img_sz,p2)
    # img1 = add_particle(img_sz,p1)
    # img2 = add_particle(img_sz,p2)

    img1=img1[25:-25,25:-25]
    img2=img2[25:-25,25:-25]
    u=u[25:-25,25:-25]
    v=v[25:-25,25:-25]
    return img1, img2, u, v

def main():
    styles = ['lamb_oseen', 'sin_flow']
    # styles = ['lamb_oseen']
    gammas = [1e3, 2e3, 3e3]
    scale = [2.5, 5.0, 7.5]

    config = AttrDict
    config.img_sz = (256,256)
    config.ppp = 0.06
    config.dp = 2.5
    config.d_std = 0.1
    config.i_std =0.1
    config.miss_ratio = 0.1
    config.style='lamb_oseen'
    config.gamma = 5e3
    config.scale = 10

    for style in styles:
        config.style = style
        for i in range(3):
            if style == 'sin_flow':
                config.scale = scale[i]
                info = f"sin_{scale[i]}"
            elif style == 'lamb_oseen':
                config.gamma = gammas[i]
                info = f"oseen_{gammas[i]}"

            img1, img2, u, v = gen_image_pair(config)
            cv2.imwrite(info+'img1.png', img1)
            cv2.imwrite(info+'img2.png', img2)
            np.savez(info+'.npz', img1=img1, img2=img2, u=u, v=v)

    plt.figure()
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)

    plt.show()

if __name__=='__main__':
   main()	

