# APDrawingGAN++ Jittor Implementation

We provide [Jittor](https://github.com/Jittor/jittor) implementations for our TPAMI 2020 paper "Line Drawings for Face Portraits from Photos using Global and Local Structure based GANs". [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Yi_APDrawingGAN_Generating_Artistic_Portrait_Drawings_From_Face_Photos_With_Hierarchical_CVPR_2019_paper.html)

It is a journal extension of our previous CVPR 2019 work [APDrawingGAN](https://github.com/yiranran/APDrawingGAN).

This project generates artistic portrait drawings from face photos using a GAN-based model.

[PyTorch implementation](https://github.com/yiranran/APDrawingGAN2)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Sample Results
Up: input, Down: output
<p>
    <img src='imgs/sample/140_large-img_1696_real_A.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1615_real_A.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1684_real_A.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1616_real_A.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1673_real_A.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1701_real_A.png' width="16%"/>
</p>
<p>
    <img src='imgs/sample/140_large-img_1696_fake_B.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1615_fake_B.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1684_fake_B.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1616_fake_B.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1673_fake_B.png' width="16%"/>
    <img src='imgs/sample/140_large-img_1701_fake_B.png' width="16%"/>
</p>

## Installation
- To install the dependencies, run
```bash
pip install -r requirements.txt
```

## Apply pretrained model

- 1. Download pre-trained models from [BaiduYun](https://pan.baidu.com/s/1JtHHZfvDQRzzqgC8fUt0cQ)(extract code: 9w83) and rename the folder to `checkpoints`.

- 2. Test for example photos: generate artistic portrait drawings for example photos in the folder `./samples/A_img/example` using models in `checkpoints/apdrawinggan++_author`
``` bash
python test.py
```
Results are saved in `./results/portrait_drawing/apdrawinggan++_author_150/example`

- 3. To test on your own photos: First run preprocess [here](preprocess/readme.md)). Then specify the folder that contains test photos using option `--input_folder`, specify the folder of landmarks using `--lm_folder`, the folder of foreground masks using `--mask_folder`, and the folder of compact masks using `--cmask_folder`, and run the `test.py` again.

## Train

- 1. Download the APDrawing dataset (augmented using histogram matching) from [BaiduYun](https://pan.baidu.com/s/1AsC056toNCQR7-q9eKPH6Q)(extract code: sq62) and put the folder to `data/apdrawing++`.

- 2. Train our model (150 epochs)
``` bash
python apdrawing_gan++.py
```
Models are saved in folder `checkpoints/apdrawing++`

- 3. Test the trained model
``` bash
python test.py --which_epoch 150 --model_name apdrawing++
```
Results are saved in `./results/portrait_drawing/apdrawing++_150/example`

## Citation
If you use this code or APDrawing dataset for your research, please cite our paper.

```
@inproceedings{YiXLLR20,
  title     = {Line Drawings for Face Portraits from Photos using Global and Local Structure based {GAN}s},
  author    = {Yi, Ran and Xia, Mengfei and Liu, Yong-Jin and Lai, Yu-Kun and Rosin, Paul L},
  booktitle = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  doi       = {10.1109/TPAMI.2020.2987931},
  year      = {2020}
}
```