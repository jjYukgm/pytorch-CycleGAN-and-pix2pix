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


class Chimera2GANModel(BaseModel):
    def name(self):
        return 'Chimera2GANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.isG3 = opt.isG3

        if not self.isG3:
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
        #       m               -> Cyc_A -> f_A , (f_A , r_A)|m -> D_A
        #       m               -> Cyc_B -> f_B , (f_B , r_B)|m -> D_B
        #       (m2, f_A, f_B)  -> G_mC  -> f_C , patch loss    -> D_C
        #       f_C             -> G_Cm  -> f_m2, f_m2|m2       -> D_C
        # m:    heat map
        # m2:   two channel heat map
        self.netG_mA = networks.define_G(opt.input_nc, opt.output_nc,
                                         opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_mB = networks.define_G(opt.input_nc, opt.output_nc,
                                         opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isG3:
            self.netG_mC = networks.define_G(2* opt.output_nc + 2* opt.input_nc, opt.output_nc,
                                             opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if not self.isG3:
                self.netG_Am = networks.define_G(opt.output_nc, opt.input_nc,
                                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
                self.netG_Bm = networks.define_G(opt.output_nc, opt.input_nc,
                                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
                self.netD_mA = networks.define_D(opt.output_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_mB = networks.define_D(opt.output_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_Am = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_Bm = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                self.netG_Cm = networks.define_G(opt.output_nc, 2* opt.input_nc,
                                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
                self.netD_mC = networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_Cm = networks.define_D(2* opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if self.isG3 or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_mA, 'G_mA', which_epoch)
            self.load_network(self.netG_mB, 'G_mB', which_epoch)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            if self.isG3:
                self.load_network(self.netG_mC, 'G_mC', which_epoch)
            if self.isTrain:
                if not self.isG3:
                    self.load_network(self.netG_Am, 'G_Am', which_epoch)
                    self.load_network(self.netG_Bm, 'G_Bm', which_epoch)
                    self.load_network(self.netD_mA, 'D_mA', which_epoch)
                    self.load_network(self.netD_mB, 'D_mB', which_epoch)
                    self.load_network(self.netD_Am, 'D_Am', which_epoch)
                    self.load_network(self.netD_Bm, 'D_Bm', which_epoch)
                else:
                    self.load_network(self.netG_Cm, 'G_Cm', which_epoch)
                    self.load_network(self.netD_mC, 'D_mC', which_epoch)
                    self.load_network(self.netD_Cm, 'D_Cm', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_mA_pool = ImagePool(opt.pool_size)
            self.fake_mB_pool = ImagePool(opt.pool_size)
            self.fake_Am_pool = ImagePool(opt.pool_size)
            self.fake_Bm_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            if not self.isG3:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_mA.parameters(), self.netG_mB.parameters(),
                                                                    self.netG_Bm.parameters(), self.netG_Bm.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_mA = torch.optim.Adam(self.netD_mA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_mB = torch.optim.Adam(self.netD_mB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_Am = torch.optim.Adam(self.netD_Am.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_Bm = torch.optim.Adam(self.netD_Bm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D_mA)
                self.optimizers.append(self.optimizer_D_mB)
                self.optimizers.append(self.optimizer_D_Am)
                self.optimizers.append(self.optimizer_D_Bm)
            else:
                self.optimizer_GC = torch.optim.Adam(itertools.chain(self.netG_mC.parameters(), self.netG_Cm.parameters()),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_mC = torch.optim.Adam(self.netD_mC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_Cm = torch.optim.Adam(self.netD_Cm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_GC)
                self.optimizers.append(self.optimizer_D_mC)
                self.optimizers.append(self.optimizer_D_Cm)
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_mA)
        networks.print_network(self.netG_mB)
        if self.isG3:
            networks.print_network(self.netG_mC)
        if self.isTrain:
            if self.isG3:
                networks.print_network(self.netG_Cm)
                networks.print_network(self.netD_Cm)
                networks.print_network(self.netD_mC)
            else:
                networks.print_network(self.netG_Am)
                networks.print_network(self.netG_Bm)
                networks.print_network(self.netD_mA)
                networks.print_network(self.netD_mB)
                networks.print_network(self.netD_Am)
                networks.print_network(self.netD_Bm)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'

        if not self.isG3:
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
        self.cond_A = Variable(self.mask_A)
        self.cond_B = Variable(self.mask_B)
        self.mask_norm = Variable(self.Tensor([1]))

        if self.isG3:
            self.cond_AA = Variable(self.mask_AA)
            self.cond_AB = Variable(self.mask_AB)
            self.cond_BA = Variable(self.mask_BA)
            self.cond_BB = Variable(self.mask_BB)
        else:
            mnorn = self.Tensor([1]).expand_as(self.mask_A)
            mask_tmp = self.mask_A
            mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
            self.real_A = Variable(self.input_A * mask_tmp)
            mask_tmp = self.mask_B
            mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
            self.real_B = Variable(self.input_B * mask_tmp)

    def test(self):
        cond_A = Variable(self.mask_A, volatile=True)
        fake_A = self.netG_mA(cond_A)
        self.fake_mA = fake_A.data

        cond_B = Variable(self.mask_B, volatile=True)
        fake_B = self.netG_mB(cond_B)
        self.fake_mB = fake_B.data

        cond_AA = Variable(self.mask_AA)
        cond_AB = Variable(self.mask_AB)
        fake_AB = self.netG_mB(cond_A)
        mask_AC = torch.cat((fake_A, fake_AB, cond_AA, cond_AB), 1)
        fake_AC = self.netG_mC(mask_AC)
        self.fake_AB = fake_AB.data
        self.fake_AC = fake_AC.data

        cond_BA = Variable(self.mask_BA)
        cond_BB = Variable(self.mask_BB)
        fake_BA = self.netG_mA(cond_B)
        mask_BC = torch.cat((fake_BA, fake_B, cond_BA, cond_BB), 1)
        fake_BC = self.netG_mC(mask_BC)
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

    def backward_D_mA(self):
        fake_mA = self.fake_mA_pool.query(self.fake_mA)
        loss_D_A = self.backward_D_basic(self.netD_mA, self.real_A, fake_mA)
        self.loss_D_mA = loss_D_A.data[0]

    def backward_D_Am(self):
        fake_Am = self.fake_Am_pool.query(self.fake_Am)
        loss_D_A = self.backward_D_basic(self.netD_Am, self.cond_A, fake_Am)
        self.loss_D_Am = loss_D_A.data[0]

    def backward_D_mB(self):
        fake_mB = self.fake_mB_pool.query(self.fake_mB)
        loss_D_B = self.backward_D_basic(self.netD_mB, self.real_B, fake_mB)
        self.loss_D_mB = loss_D_B.data[0]

    def backward_D_Bm(self):
        fake_Bm = self.fake_Bm_pool.query(self.fake_Bm)
        loss_D_B = self.backward_D_basic(self.netD_Bm, self.cond_B, fake_Bm)
        self.loss_D_Bm = loss_D_B.data[0]
        
    def backward_D_mC(self):
        fake_AC = self.fake_mA_pool.query(self.fake_AC)
        fake_BC = self.fake_mB_pool.query(self.fake_BC)
        loss_D_C  = self.backward_D_basic(self.netD_mC, self.real_A, fake_AC)
        loss_D_C += self.backward_D_basic(self.netD_mC, self.real_B, fake_BC)
        loss_D_C *= 0.5
        self.loss_D_mC = loss_D_C.data[0]

    def backward_D_Cm(self):
        fake_ACm = self.fake_Am_pool.query(self.fake_ACm)
        fake_BCm = self.fake_Bm_pool.query(self.fake_BCm)
        cond_AC =  torch.cat((self.cond_AA, self.cond_AB), 1)
        cond_BC =  torch.cat((self.cond_BA, self.cond_BB), 1)
        loss_D_C  = self.backward_D_basic(self.netD_Cm, cond_AC, fake_ACm)
        loss_D_C += self.backward_D_basic(self.netD_Cm, cond_BC, fake_BCm)
        loss_D_C *= 0.5
        self.loss_D_Cm = loss_D_C.data[0]

    def backward_GC(self):
        # mod: A, B, C
        lambda_idt = self.opt.identity
        lambda_ = self.opt.lambda_C
        
        fake_AA = self.netG_mA(self.cond_A)
        fake_AB = self.netG_mB(self.cond_A)
        real_AA = fake_AA * (torch.cat((self.cond_AA, self.cond_AA, self.cond_AA), 1) +
                             self.mask_norm.expand_as(fake_AA)) / \
                            (self.mask_norm * 2).expand_as(fake_AA)
        real_AB = fake_AB * (torch.cat((self.cond_AB, self.cond_AB, self.cond_AB), 1) +
                             self.mask_norm.expand_as(fake_AA)) / \
                            (self.mask_norm * 2).expand_as(fake_AA)
        
        fake_BA = self.netG_mA(self.cond_B)
        fake_BB = self.netG_mB(self.cond_B)
        real_BB = fake_BB * (torch.cat((self.cond_BB, self.cond_BB, self.cond_BB), 1) +
                  self.mask_norm.expand_as(fake_BB)) / \
                (self.mask_norm * 2).expand_as(fake_BB)
        real_BA = fake_BA * (torch.cat((self.cond_BA, self.cond_BA, self.cond_BA), 1) +
                  self.mask_norm.expand_as(fake_BB)) / \
                (self.mask_norm * 2).expand_as(fake_BB)
            
        mask_AC = torch.cat((fake_AA, fake_AB, self.cond_AA, self.cond_AB), 1)
        mask_BC = torch.cat((fake_BA, fake_BB, self.cond_BA, self.cond_BB), 1)
        


        # GAN loss D_C(G_C(CA))
        fake_AC = self.netG_mC(mask_AC)
        pred_fake = self.netD_mC(fake_AC)
        loss_G_AC = self.criterionGAN(pred_fake, True)
        
        # GAN loss D_C(G_C(CB))
        fake_BC = self.netG_mC(mask_BC)
        pred_fake = self.netD_mC(fake_BC)
        loss_G_BC = self.criterionGAN(pred_fake, True)

        # cycle loss
        fake_ACm = self.netG_Cm(fake_AC)
        loss_cycle_AC = self.criterionCycle(fake_ACm, torch.cat((self.cond_AA, self.cond_AB), 1)) * lambda_
        pred_fake = self.netD_Cm(fake_ACm)
        loss_G_ACm = self.criterionGAN(pred_fake, True)
        fake_BCm = self.netG_Cm(fake_BC)
        loss_cycle_BC = self.criterionCycle(fake_BCm, torch.cat((self.cond_BA, self.cond_BB), 1)) * lambda_
        pred_fake = self.netD_Cm(fake_BCm)
        loss_G_BCm = self.criterionGAN(pred_fake, True)

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if mask_AC is fed.
            idt_AC = fake_AC * (torch.cat((self.cond_AB, self.cond_AB, self.cond_AB), 1) +
                                self.mask_norm.expand_as(fake_AA)) / \
                               (self.mask_norm * 2).expand_as(fake_AA)
            loss_idt_AC = self.criterionIdt(idt_AC, real_AB) * lambda_ * lambda_idt
            idt_AC = fake_AC * (torch.cat((self.cond_AA, self.cond_AA, self.cond_AA), 1) +
                               self.mask_norm.expand_as(fake_AA)) / \
                              (self.mask_norm * 2).expand_as(fake_AA)
            loss_idt_AC += self.criterionIdt(idt_AC, real_AA) * lambda_ * lambda_idt

            # G_A should be identity if mask_BC is fed.
            idt_BC = fake_AC * (torch.cat((self.cond_BA, self.cond_BA, self.cond_BA), 1) +
                                self.mask_norm.expand_as(fake_BB)) / \
                               (self.mask_norm * 2).expand_as(fake_BB)
            loss_idt_BC = self.criterionIdt(idt_BC, real_BA) * lambda_ * lambda_idt
            idt_BC = fake_AC * (torch.cat((self.cond_BB, self.cond_BB, self.cond_BB), 1) +
                               self.mask_norm.expand_as(fake_BB)) / \
                              (self.mask_norm * 2).expand_as(fake_BB)
            loss_idt_BC += self.criterionIdt(idt_BC, real_BB) * lambda_ * lambda_idt

            self.loss_idt_AC = loss_idt_AC.data[0]
            self.loss_idt_BC = loss_idt_BC.data[0]
        else:
            loss_idt_AC = 0
            loss_idt_BC = 0
            self.loss_idt_AC = 0
            self.loss_idt_BC = 0
        
        # combined loss
        loss_G = loss_G_AC + loss_G_BC + loss_G_ACm + loss_G_BCm + \
                 loss_cycle_AC + loss_cycle_BC +\
                 loss_idt_AC + loss_idt_BC
        loss_G.backward()
        
        
        self.fake_AA = fake_AA.data
        self.fake_AB = fake_AB.data
        self.fake_BA = fake_BA.data
        self.fake_BB = fake_BB.data
        self.fake_AC = fake_AC.data
        self.fake_BC = fake_BC.data
        self.fake_ACm = fake_ACm.data
        self.fake_BCm = fake_BCm.data
        self.loss_G_AC = loss_G_AC.data[0]
        self.loss_G_BC = loss_G_BC.data[0]
        self.loss_G_ACm = loss_G_ACm.data[0]
        self.loss_G_BCm = loss_G_BCm.data[0]
        self.loss_cycle_AC = loss_cycle_AC.data[0]
        self.loss_cycle_BC = loss_cycle_BC.data[0]
    
    def backward_G(self):
        # mod: A, B
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B


        # GAN loss D_A(G_A(A))
        fake_mA = self.netG_mA(self.cond_A)
        pred_fake = self.netD_mA(fake_mA)
        loss_G_mA = self.criterionGAN(pred_fake, True)
        fake_Am = self.netG_Am(self.real_A)
        pred_fake = self.netD_Am(fake_Am)
        loss_G_Am = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_mB = self.netG_mB(self.cond_B)
        pred_fake = self.netD_mB(fake_mB)
        loss_G_mB = self.criterionGAN(pred_fake, True)
        fake_Bm = self.netG_Bm(self.real_B)
        pred_fake = self.netD_Bm(fake_Bm)
        loss_G_Bm = self.criterionGAN(pred_fake, True)

        # Cycle loss G_A(G_A(A))
        rec_Am = self.netG_Am(fake_mA)
        loss_cycle_Am = self.criterionCycle(rec_Am, self.cond_A) * lambda_A
        rec_mA = self.netG_mA(fake_Am)
        loss_cycle_mA = self.criterionCycle(rec_mA, self.real_A) * lambda_A

        # Cycle loss G_B(G_A(B))
        rec_Bm = self.netG_Bm(fake_mB)
        loss_cycle_Bm = self.criterionCycle(rec_Bm, self.cond_B) * lambda_B
        rec_mB = self.netG_mB(fake_Bm)
        loss_cycle_mB = self.criterionCycle(rec_mB, self.real_B) * lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if mask_A is fed.
            loss_idt_mA = self.criterionIdt(fake_mA, self.real_A) * lambda_A * lambda_idt
            loss_idt_Am = self.criterionIdt(fake_Am, self.cond_A) * lambda_A * lambda_idt
            # G_A should be identity if real_B is fed.
            loss_idt_mB = self.criterionIdt(fake_mB, self.real_B) * lambda_B * lambda_idt
            loss_idt_Bm = self.criterionIdt(fake_Bm, self.cond_B) * lambda_B * lambda_idt

            self.loss_idt_mA = loss_idt_mA.data[0]
            self.loss_idt_mB = loss_idt_mB.data[0]
            self.loss_idt_Am = loss_idt_Am.data[0]
            self.loss_idt_Bm = loss_idt_Bm.data[0]
        else:
            loss_idt_mA = loss_idt_mB = loss_idt_Am = loss_idt_Bm = 0
            self.loss_idt_mA = 0
            self.loss_idt_mB = 0
            self.loss_idt_Am = 0
            self.loss_idt_Bm = 0

        # combined loss
        loss_G = loss_G_mA + loss_G_mB + loss_G_Am + loss_G_Bm +  \
                 loss_idt_mA + loss_idt_mB + loss_idt_Am + loss_idt_Bm + \
                 loss_cycle_Am + loss_cycle_mA + loss_cycle_Bm + loss_cycle_mB
        loss_G.backward()
        
        self.fake_mA = fake_mA.data
        self.loss_G_mA = loss_G_mA.data[0]
        self.fake_mB = fake_mB.data
        self.loss_G_mB = loss_G_mB.data[0]
        self.fake_Am = fake_Am.data
        self.loss_G_Am = loss_G_Am.data[0]
        self.fake_Bm = fake_Bm.data
        self.loss_G_Bm = loss_G_Bm.data[0]
        self.rec_Am = rec_Am.data
        self.loss_cycle_Am = loss_cycle_Am.data[0]
        self.rec_mA = rec_mA.data
        self.loss_cycle_mA = loss_cycle_mA.data[0]
        self.rec_Bm = rec_Bm.data
        self.loss_cycle_Bm = loss_cycle_Bm.data[0]
        self.rec_mB = rec_mB.data
        self.loss_cycle_mB = loss_cycle_mB.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        if not self.isG3:
            # G_A, G_B
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # D_mA
            self.optimizer_D_mA.zero_grad()
            self.backward_D_mA()
            self.optimizer_D_mA.step()
            # D_Am
            self.optimizer_D_Am.zero_grad()
            self.backward_D_Am()
            self.optimizer_D_Am.step()
            # D_mB
            self.optimizer_D_mB.zero_grad()
            self.backward_D_mB()
            self.optimizer_D_mB.step()
            # D_Bm
            self.optimizer_D_Bm.zero_grad()
            self.backward_D_Bm()
            self.optimizer_D_Bm.step()
        else:
            # G_C
            self.optimizer_GC.zero_grad()
            self.backward_GC()
            self.optimizer_GC.step()
            # D_mC
            self.optimizer_D_mC.zero_grad()
            self.backward_D_mC()
            self.optimizer_D_mC.step()
            # D_Cm
            self.optimizer_D_Cm.zero_grad()
            self.backward_D_Cm()
            self.optimizer_D_Cm.step()

    def get_current_errors(self):
        if not self.isG3:
            ret_errors = OrderedDict([('D_mA', self.loss_D_mA), ('G_mA', self.loss_G_mA),
                                      ('D_mB', self.loss_D_mB), ('G_mB', self.loss_G_mB),
                                      ('D_Am', self.loss_D_Am), ('G_Am', self.loss_G_Am),
                                      ('D_Bm', self.loss_D_Bm), ('G_Bm', self.loss_G_Bm),
                                      ('Cyc_mA', self.loss_cycle_mA), ('Cyc_Am', self.loss_cycle_Am),
                                      ('Cyc_mB', self.loss_cycle_mB), ('Cyc_Bm', self.loss_cycle_Bm)])
            if self.opt.identity > 0.0:
                ret_errors['idt_mA'] = self.loss_idt_mA
                ret_errors['idt_mB'] = self.loss_idt_mB
                ret_errors['idt_Am'] = self.loss_idt_Am
                ret_errors['idt_Bm'] = self.loss_idt_Bm
            return ret_errors
        else:
            ret_errors = OrderedDict([ ('G_mC', self.loss_D_mC), ('G_BC', self.loss_G_BC),
                                       ('G_Cm', self.loss_D_Cm), ('G_BCm', self.loss_G_BCm)])
            if self.opt.identity > 0.0:
                ret_errors['idt_AC'] = self.loss_idt_AC
                ret_errors['idt_BC'] = self.loss_idt_BC
            return ret_errors

    def get_current_visuals(self):
        if not self.isG3:
            cond_A = util.tensor2im(self.mask_A)
            cond_B = util.tensor2im(self.mask_B)
            fake_mA = util.tensor2im(self.fake_mA)
            fake_mB = util.tensor2im(self.fake_mB)
            fake_Am = util.tensor2im(self.fake_Am)
            fake_Bm = util.tensor2im(self.fake_Bm)
            rec_mA = util.tensor2im(self.rec_mA)
            rec_mB = util.tensor2im(self.rec_mB)
            rec_Am = util.tensor2im(self.rec_Am)
            rec_Bm = util.tensor2im(self.rec_Bm)
            input_A = util.tensor2im(self.input_A)
            input_B = util.tensor2im(self.input_B)
            real_A = util.tensor2im(self.real_A.data)
            real_B = util.tensor2im(self.real_B.data)

            ret_visuals = OrderedDict([('cond_A', cond_A), ('fake_mA', fake_mA), ('fake_Am', fake_Am),
                                       ('cond_B', cond_B), ('fake_mB', fake_mB), ('fake_Bm', fake_Bm),
                                                           ('rec_mA', rec_mA), ('rec_mB', rec_mB),
                                                           ('rec_Am', rec_Am), ('rec_Bm', rec_Bm),
                                                           ('input_A', input_A), ('input_B', input_B),
                                                           ('real_A', real_A), ('real_B', real_B)])
        else:
            cond_A = util.tensor2im(self.mask_A)
            cond_AA = util.tensor2im(self.mask_AA)
            cond_B = util.tensor2im(self.mask_B)
            cond_BB = util.tensor2im(self.mask_BB)
            fake_AA = util.tensor2im(self.fake_AA)
            fake_BB = util.tensor2im(self.fake_BB)
            fake_AB = util.tensor2im(self.fake_AB)
            fake_BA = util.tensor2im(self.fake_BA)
            fake_AC = util.tensor2im(self.fake_AC)
            fake_BC = util.tensor2im(self.fake_BC)

            ret_visuals = OrderedDict([('cond_A', cond_A),   ('cond_AA', cond_AA),
                                       ('fake_AA', fake_AA), ('fake_AB', fake_AB),
                                       ('cond_B', cond_B),   ('cond_BB', cond_BB),
                                       ('fake_BB', fake_BB), ('fake_BA', fake_BA),
                                       ('fake_AC', fake_AC), ('fake_BC', fake_BC)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_mA, 'G_mA', label, self.gpu_ids)
        self.save_network(self.netG_mB, 'G_mB', label, self.gpu_ids)
        if not self.isG3:
            self.save_network(self.netD_mA, 'D_mA', label, self.gpu_ids)
            self.save_network(self.netD_mB, 'D_mB', label, self.gpu_ids)
            self.save_network(self.netG_Am, 'G_Am', label, self.gpu_ids)
            self.save_network(self.netD_Am, 'D_Am', label, self.gpu_ids)
            self.save_network(self.netG_Bm, 'G_Bm', label, self.gpu_ids)
            self.save_network(self.netD_Bm, 'D_Bm', label, self.gpu_ids)
        else:
            self.save_network(self.netG_mC, 'G_mC', label, self.gpu_ids)
            self.save_network(self.netD_mC, 'D_mC', label, self.gpu_ids)
            self.save_network(self.netG_Cm, 'G_Cm', label, self.gpu_ids)
            self.save_network(self.netD_Cm, 'D_Cm', label, self.gpu_ids)
