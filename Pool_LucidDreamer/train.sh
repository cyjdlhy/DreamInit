export CUDA_VISIBLE_DEVICES=1
python train.py --opt 'configs/base.yaml' --text 'A knight is setting up a campfire.' --workspace 'workspace/test' \
       --init_shape 'our_xyz' --init_prompt 'workspace/test/xyz_0045.pt' --opacity_from_iter 1100 
