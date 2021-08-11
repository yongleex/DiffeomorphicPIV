import numpy as np
import matplotlib.pyplot as plt


EPS = 1e-10


def angle(u, v):
    # theta = arctan(v/u), but the output should be [0, 2*pi]
    r = np.sqrt(u**2+v**2)
    theta = np.arccos(u/(r+EPS)) # [0, pi]
    theta = theta*(v>=0) + (2*np.pi-theta)*(v < 0)  # [0, 2*pi]
    return theta


def lamb_oseen(x, y, Gamma=2e3, rc=40):
    r = np.sqrt(x**2+y**2)+1e-9
    theta = angle(x, y)
    
    V = Gamma*(1-np.exp(-r**2/rc**2))/(2*np.pi*r)  # circumferential vel
    d_theta = V/r

    u = -V*y/r
    v = V*x/r
    
    x1 = r*np.cos(theta-d_theta/2)
    y1 = r*np.sin(theta-d_theta/2)
    x2 = r*np.cos(theta+d_theta/2)
    y2 = r*np.sin(theta+d_theta/2)
    
    return u, v, x1, y1, x2, y2


def sin_flow(x, y, a=6, b=128, scale=5):
    theta = np.arctan(a*np.cos(2*np.pi*x/b)*2*np.pi/b)
    u = scale*np.cos(theta)
    v = scale*np.sin(theta)
    
    x1, y1 = x, y
    x2, y2 = x, y
    N = 1000
    ds = 0.5*scale/N
    for i in range(N):
        alpha1 = np.arctan(a*np.cos(2*np.pi*x1/b)*2*np.pi/b)
        x1 = x1 - ds*np.cos(alpha1)
        y1 = y1 - ds*np.sin(alpha1)
        
        alpha2 = np.arctan(a*np.cos(2*np.pi*x2/b)*2*np.pi/b)
        x2 = x2 + ds*np.cos(alpha2)
        y2 = y2 + ds*np.sin(alpha2)
        
    return u, v, x1, y1, x2, y2


def test_flow():
    x = np.random.rand(100)*128
    y = np.random.rand(100)*128
    
#     u, v, x1, y1, x2, y2 = lamb_oseen(x, y, Gamma=5e3, rc=20)
    u, v, x1, y1, x2, y2 = sin_flow(x, y, a=20, b=128, scale=25)

    plt.figure(figsize=(10,10))
    plt.scatter(x,y,color='k', linewidth=8)
    plt.scatter(x1,y1,color='r')
    plt.scatter(x2,y2,color='b')
    plt.axis('equal')
    plt.quiver(x1, y1, x2-x1, y2-y1, scale=1, color=(0.5,0.5,0.5), units='xy')
    plt.quiver(x, y, u/2, v/2, scale=1, color=(0.0,0.0,0.5), units='xy')
    plt.show()


if __name__=='__main__':
    test_flow()

