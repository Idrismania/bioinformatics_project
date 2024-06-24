# U-Net model implementation
This repository contains code used to train a UNet model on histopathological breast cancer images, using CODEX-derived binary mask labels. Training images are shape (512, 512, 3) RGB .tif images and label images are (512, 512, C) multi-channel .tif images where the channel to train on is indexed in `dataset.py`

<p align="center">
  <img src="images/architecture_dark.PNG" alt="UNET model" width="60%">
  <br>
</p>

In order to replicate this project, structure a data folder like the following:

<pre>
data/
├─ marker-specific-dataset_01/
│  ├─ he/
│  │  ├─ img_0.tif
│  │  └─ img_1.tif
│  └─ masks/
│     ├─ img_0_mask.tif
│     └─ img_1_mask.tif
│  
└─ marker-specific-dataset_02/
   ├─ he/
   │  ├─ img_0.tif
   │  └─ img_1.tif
   └─ masks/
      ├─ img_0_mask.tif
      └─ img_1_mask.tif
</pre>

In order to configure dlup and pyvips, Add Windows binary files to your windows PATH folder and import them through `os.add_dll_directory("FILE_PATH")`. File paths and hyperparameters are configured through Hydra-Core in `conf/config.yaml` and `src/config.py`.

* pyvips: https://github.com/libvips/build-win64-mxe/releases/tag/v8.15.0 <br>
* openslide (for dlup): https://openslide.org/download/

Important consideration for performance improvement beyond improving the data processing method includes implementation of weighted `BCEwithlogitsloss()`:

```math
\ell\left (\mathbf{\hat{Y}, Y}\right )= - \left (\beta \odot \mathbf{Y}\odot\ln\left (\sigma\left ( \mathbf{\hat{Y}} \right )\right )+\left (1-\mathbf{Y}\right )\odot\ln\left (1-\sigma\left ( \mathbf{\hat{Y}} \right )\right )\right )
```
Where the β factor compensates for low occurence of the positive class.
