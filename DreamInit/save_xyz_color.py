import torch
import argparse

from trainer import *
from generator.DreamInit import DreamInit
from generator.provider import GenerateCircleCameras, cameraList_from_RcamInfos
from diffusers import IFPipeline, StableDiffusionPipeline

from inference import generate_camera_path,parse_args




def save_xyz_color(opt, prompt,model_path,save_xyz_path,save_color_path):
    # opt = parse_args()
    opt.xyzres = True
    opt.fp16 = True
    opt.image_h = opt.h
    opt.image_w = opt.w
    device = torch.device('cuda')
    opt.device = device

    generator = DreamInit(opt).to(device)
    model_ckpt = torch.load(model_path, map_location='cpu')
    generator.load_state_dict(model_ckpt['model'])
    if 'ema' in model_ckpt and opt.ema:
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.99)
        ema.load_state_dict(model_ckpt['ema'])
        ema.copy_to()

    model_key = "/mnt/models--DeepFloyd--IF-I-XL-v1.0/snapshots/c03d510e9b75bce9f9db5bb85148c1402ad7e694"


    pipe = IFPipeline.from_pretrained(
        model_key,
        variant="fp16",
        torch_dtype=torch.float16)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)

    cameras = generate_camera_path(120, opt)

    generator.eval()

    inputs = tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
    embeddings = text_encoder(inputs.input_ids.to(device))[0]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            gaussian_models = generator.gaussian_generate(embeddings)
        # images = generator.render(gaussian_models, [cameras])['rgbs']
        # images = images * 255
        # images = images.permute(0, 2, 3, 1)
        #
        # save_path = opt.save_path+'/'+opt.prompt
        # os.makedirs(save_path, exist_ok=True)
        # imageio.mimwrite(os.path.join(save_path, opt.prompt + '.mp4'),
        #                  images.to(torch.uint8).cpu(), fps=25, quality=8,
        #                  macro_block_size=1)
        # output images to cooresponding folder
        # os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        # for i in range(images.shape[0]):
        #     imageio.imsave(os.path.join(save_path, 'images', f'{i}.png'), images[i].to(torch.uint8).cpu().numpy())


        selected_index = gaussian_models[0]._opacity.squeeze() > 0.01
        xyz = gaussian_models[0]._xyz[selected_index]
        features_dc = gaussian_models[0]._features_dc[selected_index]

        torch.save(xyz, os.path.join(save_xyz_path))
        torch.save(features_dc, os.path.join(save_color_path))
        print(xyz.shape)
        print(features_dc.shape)

if __name__ == '__main__':
    prompt = "a humanoid banana sitting at a desk doing homework."
    model_path = "workspace2/apple_banana_IF_lr5e-5_cbs4_opacity-2__scales2_rotations1_grid48/checkpoints/BrightDreamer_ep0030.pth"
    save_xyz_path = "test_xyz.pt"
    save_color_path= "test_color.pt"
    save_xyz_color(prompt,model_path,save_xyz_path,save_color_path)