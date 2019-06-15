import torch
import torch.nn as nn
import torch.nn.functional as F

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nout),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nout),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)

# class pose_encoder(nn.Module):
#     def __init__(self, pose_dim, nc=1):
#         super(pose_encoder, self).__init__()
#         nf = 64
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             dcgan_conv(nc, nf),
#             # state size. (nf) x 32 x 32
#             dcgan_conv(nf, nf * 2),
#             # state size. (nf*2) x 16 x 16
#             dcgan_conv(nf * 2, nf * 4),
#             # state size. (nf*4) x 8 x 8
#             dcgan_conv(nf * 4, nf * 8),
#             # state size. (nf*8) x 4 x 4
#             nn.Conv2d(nf * 8, pose_dim, 4, 1, 0),
#             nn.BatchNorm2d(pose_dim),
#             nn.Tanh()
#         )

#     def forward(self, input):
#         return self.main(input)

class encoder(nn.Module):
    def __init__(self, content_dim, pose_dim, nc=1, conditional=False):
        super(encoder, self).__init__()
        self.conditional = conditional
        self.content_dim = content_dim
        self.pose_dim = pose_dim
        nf = 64
        if self.conditional:
            add_nf = 2
        else:
            add_nf = 0
        
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc+add_nf, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf+add_nf, nf*2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf*2+add_nf, nf*4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf*4+add_nf, nf*8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
            nn.Conv2d(nf*8+add_nf, content_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(content_dim),
            nn.Tanh()
        )

    def forward(self, input, cond=None):
        if cond is not None:
            # cond: batch x (pose_dim) x 1 x 1
            assert cond.size(1) == self.pose_dim
            # cond_1: batch x 1 x pose_dim x pose_dim
            # cond_2: batch x 1 x pose_dim x pose_dim
            cond_1 = cond.view(-1, 1, 1, cond.size(1))
            cond_1 = cond_1.repeat(1, 1, cond.size(1), 1)
            cond_1 = F.interpolate(cond_1, size=64)
            cond_2 = cond.view(-1, 1, cond.size(1), 1)
            cond_2 = cond_2.repeat(1, 1, 1, cond.size(1))
            cond_2 = F.interpolate(cond_2, size=64)
            
            
            cond_list = []
            for i in range(5):
                cond_list.append((
                    F.interpolate(cond_1, scale_factor=1/(2**i)),
                    F.interpolate(cond_2, scale_factor=1/(2**i)),
                ))
            
        
        h1 = self.c1(torch.cat([input, cond_list[0][0], cond_list[0][1]], dim=1))
        h2 = self.c2(torch.cat([h1, cond_list[1][0], cond_list[1][1]], dim=1))
        h3 = self.c3(torch.cat([h2, cond_list[2][0], cond_list[2][1]], dim=1))
        h4 = self.c4(torch.cat([h3, cond_list[3][0], cond_list[3][1]], dim=1))
        h5 = self.c5(torch.cat([h4, cond_list[4][0], cond_list[4][1]], dim=1))
        
        return h5
    
class motion_encoder(nn.Module):
    def __init__(self, pose_dim, nc=1):
        super(motion_encoder, self).__init__()
        nf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            dcgan_conv(nc, nf),
            # state size. (nf) x 32 x 32
            dcgan_conv(nf, nf * 2),
            # state size. (nf*2) x 16 x 16
            dcgan_conv(nf * 2, nf * 4),
            # state size. (nf*4) x 8 x 8
            dcgan_conv(nf * 4, nf * 8),
            # state size. (nf*8) x 4 x 4
            nn.Conv2d(nf * 8, pose_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(pose_dim),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class decoder(nn.Module):
    def __init__(self, content_dim, pose_dim, nc=1, conditional=False):
        super(decoder, self).__init__()
        nf = 64
        self.conditional = conditional
        self.content_dim = content_dim
        self.pose_dim = pose_dim
        if self.conditional:
            add_nf = 2
        else:
            add_nf = 0
            
        self.u1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(content_dim+pose_dim, nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
        )
        # state size. (nf*8) x 4 x 4
        self.u2 = dcgan_upconv(nf*8 + add_nf, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.u3 = dcgan_upconv(nf*4 + add_nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.u4 = dcgan_upconv(nf*2 + add_nf, nf)
        # state size. (nf) x 32 x 32
        self.u5 = nn.Sequential(
            nn.ConvTranspose2d(nf + add_nf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, latent, cond=None):
        if cond is not None:
            # cond: batch x (pose_dim) x 1 x 1
            assert cond.size(1) == self.pose_dim
            # cond_1: batch x 1 x pose_dim x pose_dim
            # cond_2: batch x 1 x pose_dim x pose_dim
            cond_1 = cond.view(-1, 1, 1, cond.size(1))
            cond_1 = cond_1.repeat(1, 1, cond.size(1), 1)
            cond_1 = F.interpolate(cond_1, size=64)
            cond_2 = cond.view(-1, 1, cond.size(1), 1)
            cond_2 = cond_2.repeat(1, 1, 1, cond.size(1))
            cond_2 = F.interpolate(cond_2, size=64)
            
            
            cond_list = []
            for i in range(5):
                cond_list.append((
                    F.interpolate(cond_1, scale_factor=1/(2**i)),
                    F.interpolate(cond_2, scale_factor=1/(2**i)),
                ))
                
        h1 = self.u1(torch.cat([latent, cond], dim=1))
        h2 = self.u2(torch.cat([h1, cond_list[-1][0], cond_list[-1][1]], dim=1))
        h3 = self.u3(torch.cat([h2, cond_list[-2][0], cond_list[-2][1]], dim=1))
        h4 = self.u4(torch.cat([h3, cond_list[-3][0], cond_list[-3][1]], dim=1))
        h5 = self.u5(torch.cat([h4, cond_list[-4][0], cond_list[-4][1]], dim=1))
        
        return h5
    
class Generator(nn.Module):
    def __init__(self, content_dim, pose_dim, nc=1):
        super(Generator, self).__init__()
        self.encoder = encoder(content_dim, pose_dim, nc, conditional=True)
        self.decoder = decoder(content_dim, pose_dim, nc, conditional=True)
    
    def forward(self, content, cond=None):
        latent = self.encoder(content, cond)
        out = self.decoder(latent, cond)
        return out

class scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf=512):
        super(scene_discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
            nn.Linear(pose_dim*2, nf),
            nn.ReLU(True),
            nn.Linear(nf, nf),
            nn.ReLU(True),
            nn.Linear(nf, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(torch.cat(input, 1).view(-1, self.pose_dim*2))
        return output

class Discriminator(nn.Module):
    def __init__(self, nc=1):
        super(Discriminator, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc*2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = torch.cat(input, 1)
        return self.main(input)
    
class CondDiscriminator(nn.Module):
    def __init__(self, nc=1):
        super(CondDiscriminator, self).__init__()
        nf = 64
        add_nf = 2
        
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc+add_nf, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf+add_nf, nf*2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf*2+add_nf, nf*4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf*4+add_nf, nf*8)
        # state size. (nf*8) x 4 x 4
        
        self.D = nn.Sequential(
            nn.Conv2d(nf*8+add_nf, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, cond=None):
        if cond is not None:
            # cond: batch x 128 x 1 x 1
            assert cond.size(1) == 128
            # cond_1: batch x 1 x 128 x 128
            # cond_2: batch x 1 x 128 x 128
            cond_1 = cond.view(-1, 1, 1, cond.size(1))
            cond_1 = cond_1.repeat(1, 1, cond.size(1), 1)
            cond_2 = cond.view(-1, 1, cond.size(1), 1)
            cond_2 = cond_2.repeat(1, 1, 1, cond.size(1))
            
            cond_list = []
            for i in range(5):
                cond_list.append((
                    F.interpolate(cond_1, scale_factor=1/(2**(i+1))),
                    F.interpolate(cond_2, scale_factor=1/(2**(i+1))),
                ))
#                 print(cond_list[i][0].shape)
#                 print(cond_list[i][1].shape)
            
        
        h1 = self.c1(torch.cat([input, cond_list[0][0], cond_list[0][1]], dim=1))
        h2 = self.c2(torch.cat([h1, cond_list[1][0], cond_list[1][1]], dim=1))
        h3 = self.c3(torch.cat([h2, cond_list[2][0], cond_list[2][1]], dim=1))
        h4 = self.c4(torch.cat([h3, cond_list[3][0], cond_list[3][1]], dim=1))
        
        out_D = self.D(torch.cat([h4, cond_list[4][0], cond_list[4][1]], dim=1)).view(-1, 1)
        
        return out_D.squeeze()