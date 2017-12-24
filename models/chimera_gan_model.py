import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class ChimeraGANModel(BaseModel):
    def name(self):
        return 'ChimeraGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        if not hasattr(self.opt, "isG3"):
            self.opt.isG3 = False
        if not self.opt.isG3:
            self.input_A = self.Tensor(nb, opt.output_nc, size, size)
            self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.mask_A = self.Tensor(nb, opt.input_nc, size, size)
        self.mask_B = self.Tensor(nb, opt.input_nc, size, size)
        self.mask_AA = self.Tensor(nb, opt.input_nc, size, size)
        self.mask_AB = self.Tensor(nb, opt.input_nc, size, size)
        self.mask_BA = self.Tensor(nb, opt.input_nc, size, size)
        self.mask_BB = self.Tensor(nb, opt.input_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # archi: 
        #       m               -> G_A -> f_A , (f_A , r_A)|m           -> D_A 
        #       m               -> G_B -> f_B , (f_B , r_B)|m           -> D_B 
        #       (m2, f_A, f_B)  -> G_C -> f_C , (r_A or r_B , f_C)|m2   -> D_C
        # m:    heat map
        # m2:   two channel heat map
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_C = networks.define_G(2* opt.output_nc + 2* opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_C = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.netG_C, 'G_C', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_C, 'D_C', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_AC_pool = ImagePool(opt.pool_size)
            self.fake_BC_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GC = torch.optim.Adam(self.netG_C.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_GC)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_C)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        networks.print_network(self.netG_C)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'

        if not self.opt.isG3:
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            self.input_A.resize_(input_A.size()).copy_(input_A)
            self.input_B.resize_(input_B.size()).copy_(input_B)

        mask_A = input['mA']
        mask_B = input['mB']
        mask_AA = input['mAA']
        mask_AB = input['mAB']
        mask_BA = input['mBA']
        mask_BB = input['mBB']
        self.mask_A.resize_(mask_A.size()).copy_(mask_A)
        self.mask_B.resize_(mask_B.size()).copy_(mask_B)
        self.mask_AA.resize_(mask_AA.size()).copy_(mask_AA)
        self.mask_AB.resize_(mask_AB.size()).copy_(mask_AB)
        self.mask_BA.resize_(mask_BA.size()).copy_(mask_BA)
        self.mask_BB.resize_(mask_BB.size()).copy_(mask_BB)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        if not self.opt.isG3:
            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
        self.cond_A = Variable(self.mask_A)
        self.cond_B = Variable(self.mask_B)
        self.cond_AA = Variable(self.mask_AA)
        self.cond_AB = Variable(self.mask_AB)
        self.cond_BA = Variable(self.mask_BA)
        self.cond_BB = Variable(self.mask_BB)

    def test(self):
        cond_A = Variable(self.mask_A, volatile=True)
        fake_A = self.netG_A(cond_A)
        self.fake_A = fake_A.data

        cond_B = Variable(self.mask_B, volatile=True)
        fake_B = self.netG_B(cond_B)
        self.fake_B = fake_B.data


        cond_AA = Variable(self.mask_AA)
        cond_AB = Variable(self.mask_AB)
        fake_AB = self.netG_B(cond_A)
        mask_AC = torch.cat((fake_A, fake_AB, cond_AA, cond_AB), 1)
        fake_AC = self.netG_C(mask_AC)
        self.fake_AB = fake_AB.data
        self.fake_AC = fake_AC.data


        cond_BA = Variable(self.mask_BA)
        cond_BB = Variable(self.mask_BB)
        fake_BA = self.netG_A(cond_B)
        mask_BC = torch.cat((fake_BA, fake_B, cond_BA, cond_BB), 1)
        fake_BC = self.netG_C(mask_BC)
        self.fake_BA = fake_BA.data
        self.fake_BC = fake_BC.data
        
        

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)
        self.loss_D_B = loss_D_B.data[0]
        
    def backward_D_C(self):
        fake_AC = self.fake_AC_pool.query(self.fake_AC)
        fake_BC = self.fake_BC_pool.query(self.fake_BC)
        loss_D_C  = self.backward_D_basic(self.netD_C, self.real_A, fake_AC)
        loss_D_C += self.backward_D_basic(self.netD_C, self.real_B, fake_BC)
        loss_D_C *= 0.5
        self.loss_D_C = loss_D_C.data[0]

    def backward_GC(self):
        # mod: A, B, C
        lambda_idt = self.opt.identity
        lambda_ = self.opt.lambda_C
        
        fake_AA = self.netG_A(self.cond_A)
        fake_AB = self.netG_B(self.cond_A)
        cond_AA, cond_AB = self.cond_AA, self.cond_AB
        real_A = self.real_A * torch.cat((cond_AA, cond_AA, cond_AA), 1)
        
        fake_BA = self.netG_A(self.cond_B)
        fake_BB = self.netG_B(self.cond_B)
        cond_BA, cond_BB = self.cond_BA, self.cond_BB
        real_B = self.real_B * torch.cat((cond_BB, cond_BB, cond_BB), 1)
            
        mask_AC = torch.cat((fake_AA, fake_AB, cond_AA, cond_AB), 1)
        mask_BC = torch.cat((fake_BA, fake_BB, cond_BA, cond_BB), 1)
        
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if mask_AC is fed.
            idt_AC = self.netG_C(mask_AC)
            idt_AC = idt_AC * torch.cat((cond_AA, cond_AA, cond_AA), 1)
            loss_idt_AC = self.criterionIdt(idt_AC, real_A) * lambda_ * lambda_idt

            # G_A should be identity if mask_BC is fed.
            idt_BC = self.netG_C(mask_BC)
            idt_BC = idt_BC * torch.cat((cond_BB, cond_BB, cond_BB), 1)
            loss_idt_BC = self.criterionIdt(idt_BC, real_B) * lambda_ * lambda_idt

            self.idt_AC = idt_AC.data
            self.idt_BC = idt_BC.data
            self.loss_idt_AC = loss_idt_AC.data[0]
            self.loss_idt_BC = loss_idt_BC.data[0]
        else:
            loss_idt_AC = 0
            loss_idt_BC = 0
            self.loss_idt_AC = 0
            self.loss_idt_BC = 0

        # GAN loss D_C(G_C(CA))
        fake_AC = self.netG_C(mask_AC)
        pred_fake = self.netD_C(fake_AC)
        loss_G_AC = self.criterionGAN(pred_fake, True)
        
        # GAN loss D_C(G_C(CB))
        fake_BC = self.netG_C(mask_BC)
        pred_fake = self.netD_C(fake_BC)
        loss_G_BC = self.criterionGAN(pred_fake, True)

        
        # combined loss
        loss_G = loss_G_AC + loss_G_BC + loss_idt_AC + loss_idt_BC
        loss_G.backward()
        
        
        self.fake_AB = fake_AB.data
        self.fake_BA = fake_BA.data
        self.fake_AC = fake_AC.data
        self.fake_BC = fake_BC.data
        self.loss_G_AC = loss_G_AC.data[0]
        self.loss_G_BC = loss_G_BC.data[0]
    
    def backward_G(self):
        # mod: A, B
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        real_A = self.real_A * torch.cat((self.cond_A, self.cond_A, self.cond_A), 1)
        real_B = self.real_B * torch.cat((self.cond_B, self.cond_B, self.cond_B), 1)
        mask_A = self.cond_A
        mask_B = self.cond_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if mask_A is fed.
            idt_A = self.netG_A(mask_A)
            loss_idt_A = self.criterionIdt(idt_A, real_A) * lambda_A * lambda_idt
            # G_A should be identity if real_B is fed.
            idt_B = self.netG_B(mask_B)
            loss_idt_B = self.criterionIdt(idt_B, real_B) * lambda_B * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_A = self.netG_A(mask_A)
        pred_fake = self.netD_A(fake_A)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_B = self.netG_B(mask_B)
        pred_fake = self.netD_B(fake_B)
        loss_G_B = self.criterionGAN(pred_fake, True)

        
        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        
        self.fake_A = fake_A.data
        self.loss_G_A = loss_G_A.data[0]
        self.fake_B = fake_B.data
        self.loss_G_B = loss_G_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A, G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # G_C
        self.optimizer_GC.zero_grad()
        self.backward_GC()
        self.optimizer_GC.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_C
        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B),
                                 ('D_C', self.loss_D_C), ('G_AC', self.loss_G_AC), ('G_BC', self.loss_G_BC)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
            ret_errors['idt_AC'] = self.loss_idt_AC
            ret_errors['idt_BC'] = self.loss_idt_BC
        return ret_errors

    def get_current_visuals(self):
        cond_A = util.tensor2im(self.mask_A)
        cond_AA = util.tensor2im(self.mask_AA)
        cond_B = util.tensor2im(self.mask_B)
        cond_BB = util.tensor2im(self.mask_BB)
        fake_A = util.tensor2im(self.fake_A)
        fake_B = util.tensor2im(self.fake_B)
        fake_AB = util.tensor2im(self.fake_AB)
        fake_BA = util.tensor2im(self.fake_BA)
        fake_AC = util.tensor2im(self.fake_AC)
        fake_BC = util.tensor2im(self.fake_BC)

        ret_visuals = OrderedDict([('fake_A', fake_A), ('fake_B', fake_B),
                                   ('cond_A', cond_A), ('cond_AA', cond_AA),
                                   ('cond_B', cond_B), ('cond_BB', cond_BB),
                                   ('fake_AB', fake_AB), ('fake_BA', fake_BA),
                                   ('fake_AC', fake_AC), ('fake_BC', fake_BC)])
        if not self.opt.isG3:
            ret_visuals['real_A'] = util.tensor2im(self.input_A)
            ret_visuals['real_B'] = util.tensor2im(self.input_B)

        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_AC'] = util.tensor2im(self.idt_AC)
            ret_visuals['idt_BC'] = util.tensor2im(self.idt_BC)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netG_C, 'G_C', label, self.gpu_ids)
        self.save_network(self.netD_C, 'D_C', label, self.gpu_ids)
