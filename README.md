# SISR Degradation Model

This repo contains the scripts for Degradation model present in the [original repository of BSRGAN by Zhang, Kai and Liang et.al](https://github.com/cszn/BSRGAN). The intention is a standalone repo for the degradation model alone for an easier usage.

```
from dgm import run_degradation
lr, hr = run_degradation('lenna.png', scale=4, patch=128, savefig=True, showfig=True)
```