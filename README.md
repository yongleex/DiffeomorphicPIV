# DiffeomorphicPIV
__Current repo only contains the experimental results recorded in Jupyter note book. The full code will be open once the manuscript is accepted.__


[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2108.07438&color=B31B1B)](http://arxiv.org/abs/2108.07438)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains code for the submitted paper *[Diffeomorphic Particle Image velocimetry](http://arxiv.org/abs/2108.07438)*. 
In this work, a diffeomorphic PIV technique is proposed to reduce the curvature effect of the non-straight particle trajectory. 
Different from other existing PIV techniques, our diffeomorphic PIV computes the real curved particle trajectory to achieve accurate velocity measurement.

### Motivation 
![movie](https://github.com/yongleex/DiffeomorphicPIV/blob/1364f48b3b448854a0af8ce5d5f316c4b197f3ca/output/movie.gif)
The diffeomorphic PIV finds the curved trajectory (streamline of velocity field) to explain the image displacement between __TWO__ recordings. It is significantly different from the straight-line approximation of existing PIV techniques. More info is referred to the [paper](https://arxiv.org/abs/2108.07438).

## Install dependencies
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install numpy matplotlib opencv seaborn
conda install -c conda-forge openpiv
conda install -c conda-forge cupy
```

### Download the well-trained deep model for DeepPIV
```
cd UnFlowNet
wget https://github.com/erizmr/UnLiteFlowNet-PIV/raw/master/models/UnsupervisedLiteFlowNet_pretrained.pt
```


## The experiments
* [Exp1.ipynb](https://github.com/yongleex/DiffeomorphicPIV/blob/main/Exp1.ipynb): Investigate the converge performance w.r.t the iteration number
* [Exp2.ipynb](https://github.com/yongleex/DiffeomorphicPIV/blob/main/Exp2.ipynb): Test on 3 Lamb-Oseen flows and 3 Sin flows;
* [Exp3.ipynb](https://github.com/yongleex/DiffeomorphicPIV/blob/main/Exp3.ipynb): Test on 3 real PIV cases;


### BibTeX

```
@article{lee2020diffeomorphic,
  title={Diffeomorphic Particle Image Velocimetry
},
  author={Lee, Yong and Mei, Shuang},
  journal={arXiv preprint arXiv:2108.07438},
  year={2021}
}
```

### Questions?
For any questions regarding this work, please email me at [yongli.cv@gmail.com](mailto:yongli.cv@gmail.com).

#### Acknowledgements
Parts of the code/deep net in this repository have been adapted from the following repos:

* [OpenPIV/openpiv-python](https://github.com/OpenPIV/openpiv-python)
* [erizmr/UnLiteFlowNet-PIV](https://github.com/erizmr/UnLiteFlowNet-PIV)
* [opencv/opencv](https://github.com/opencv/opencv)
