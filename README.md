# A General Framework to Boost 3D GS Initialization for Text-to-3D Generation by Lexical Richness

### [Project Page](vlislab22.github.io/DreamInit/) | [Arxiv](https://arxiv.org/abs/2408.01269)

The official implementation of A General Framework to Boost 3D GS Initialization for Text-to-3D Generation by Lexical Richness.


# Install
```bash
git clone https://github.com/cyjdlhy/Dreaminit.git
cd DreamInit
conda create -n DreamInit python=3.9
conda activate DreamInit
```

You need to first install the suitable torch and torchvision according your environment. The version used in our experiments is 

```
torch==1.13.1+cu117   torchvision==0.14.1+cu117
```

Then you can install other packages by

```
pip install -r requirements.txt
mkdir submodules
cd submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --recursive
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./diff-gaussian-rasterization/
pip install ./simple-knn/
cd ..
```

For second stage, you need to install another corresponding environment, e.g., LucidDreamer.


# Training

```
CUDA_VISIBLE_DEVICES=0 python main.py --prompt "A knight is setting up a campfire" --workspace workspace/test --port 7348 --fp16 --perpneg --lr 7e-5
```

Notice: Learning rate is important for convergence speed.
After training, you can obtain a initial GS file in workspace.
Subsequently, load it in the second-stage method.


# Second-stage
```
# Pool_LucidDreamer
cd ..
cd Pool_LucidDreamer
./train.sh
```


# Acknowledgement

Our code is inspired by [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer) and [DeepFloyd-IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).
Thanks for their outstanding works and open-source!

# Citation

If you find this work useful, a citation will be appreciated via:

```
@inproceedings{jiang2024general,
  title={A General Framework to Boost 3D GS Initialization for Text-to-3D Generation by Lexical Richness},
  author={Jiang, Lutao and Li, Hangyu and Wang, Lin},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6803--6812},
  year={2024}
}
```