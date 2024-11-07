import torch
import torch.nn as nn
from .generator import Generator
from generator.gaussian_utils.gaussian_model import GaussianModel
from generator.gaussian_utils.gaussian_renderer import render
import numpy as np

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class DreamInit(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt=opt
        self.sh_degree = 0
        self.img_channel = 3 + 1 + 4 + 3 + 3 * (self.sh_degree + 1) ** 2
        self.grid_resolution = self.opt.grid_resolution
        self.bound = 1.0
        # self.free_distance = self.opt.free_distance
        self.generator = Generator(opt=opt, hidden_dim=self.opt.hidden_dim)  # opacity and color
        self.pp = PipelineParams()
        self.register_buffer('background', torch.ones(3))



        xyz = torch.stack(torch.meshgrid(torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution),
                                                                         torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution),
                                                                         torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution)), dim=3).reshape(-1,3).contiguous()


        self.register_buffer('xyz', xyz)

        self.fixed_rotations = self.opt.fixed_rotations
        self.fixed_scales = self.opt.fixed_scales


    def gaussian_generate(self, text_embeddings):

        B = text_embeddings.shape[0]
        input = self.xyz.unsqueeze(0).repeat(B, 1, 1)
        gaussians_property = self.generator(input, text_embeddings)

        gaussians_property = gaussians_property.to(torch.float32)
        gaussian_list = []  # (B,)
        for i in range(B):

            gaussian = GaussianModel(self.sh_degree)
            gaussian._xyz = self.xyz
            gaussian._opacity = gaussians_property[i, :, 0:1]
            gaussian._rotation = torch.tensor([self.fixed_rotations,0,0,0]).repeat(gaussians_property[i].shape[0], 1).to("cuda")  # fixed
            gaussian._scaling = torch.tensor([self.fixed_scales]*3).repeat(gaussians_property[i].shape[0], 1).to("cuda")  # fixed
            gaussian._features_dc = (3.545*torch.sigmoid(gaussians_property[i, :, 1:4].reshape(-1, 1, 3)) - 1.7725)
            gaussian._features_rest = gaussians_property[i, :, 4:].reshape(-1, 15, 3)
            gaussian_list.append(gaussian)

        return gaussian_list

    def render(self, gaussians, views):
        B = len(gaussians)
        C = len(views[0])
        rgbs = []
        for i in range(B):
            gaussian = gaussians[i]
            for j in range(C):
                view = views[i][j]
                render_pkg = render(view, gaussian, self.pp, self.background)
                rgb = render_pkg['render']
                rgbs.append(rgb)
        rgbs = torch.stack(rgbs, dim=0)
        return {'rgbs': rgbs}

    def forward(self, text_zs, cameras):
        gaussians = self.gaussian_generate(text_zs)
        outputs = self.render(gaussians, cameras)
        return outputs

    def get_params(self, lr):

        params = [
            {'params': self.parameters(), 'lr': lr},
        ]
        return params