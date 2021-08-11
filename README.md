# DiffeomorphicPIV
** Current repo only contains the experimental results recorded in Jupyter note book. The full code will be open once the manuscript is accepted.**


[![preprint](https://img.shields.io/static/v1?label=arXiv&message=0000.0000&color=B31B1B)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains code for the submitted paper *[Diffeomorphic Particle Image velocimetry](https://arxiv.org/)*. 
In this work, a diffeomorphic PIV technique is proposed to reduce the curvature effect of the non-straight particle trajectory. 
Our diffeomorphic PIV is devoted to estimating the exact velocity vector field instead of computing the deformation field like other existing PIV techniques.


## Install dependencies
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install numpy matplotlib opencv seaborn
conda install -c conda-forge openpiv
conda install -c conda-forge cupy
```


## The experiments
* [Exp1.ipynb](www.xxx.com): Investigate the converge performance w.r.t the iteration number
* [Exp2.ipynb](www.xxx.com): Test on 3 Lamb-Oseen flows and 3 Sin flows;
* [Exp3.ipynb](www.xxx.com): Test on 3 real PIV cases;


### BibTeX

```
@xxx
```

### Questions?
For any questions regarding this work, please email me at [yongli.cv@gmail.com](mailto:yongli.cv@gmail.com).

#### Acknowledgements
Parts of the code/network structures in this repository have been adapted from the following repos:

* [OpenPIV/openpiv-python](https://github.com/OpenPIV/openpiv-python)
* [erizmr/UnLiteFlowNet-PIV](https://github.com/erizmr/UnLiteFlowNet-PIV)
* [opencv/opencv](https://github.com/opencv/opencv)
