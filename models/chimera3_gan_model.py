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


class Chimera3GANModel(BaseModel):
    def name(self):
        return 'Chimera3GANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.isG3 = opt.isG3
        
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
        if self.isG3:
            self.netG_C = networks.define_G(2* opt.output_nc + 2* opt.input_nc, opt.output_nc,
                                            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.isG3:
                self.netD_C = networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if self.isG3 or (self.isTrain or opt.continue_train):
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            if self.isG3:
                self.load_network(self.netG_C, 'G_C', which_epoch)
            if self.isTrain:
                if self.isG3:
                    self.load_network(self.netD_C, 'D_C', which_epoch)
                else:
                    self.load_network(self.netD_A, 'D_A', which_epoch)
                    self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            if self.isG3:
                self.optimizer_GC = torch.optim.Adam(self.netG_C.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_GC)
                self.optimizers.append(self.optimizer_D_C)
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D_A)
                self.optimizers.append(self.optimizer_D_B)
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isG3:
            networks.print_network(self.netG_C)
        if self.isTrain:
            if self.isG3:
                networks.print_network(self.netD_C)
            else:
                networks.print_network(self.netD_A)
                networks.print_network(self.netD_B)
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
        self.mask_norm = Variable(self.Tensor([1]))
        self.cond_A = Variable(self.mask_A)
        self.cond_B = Variable(self.mask_B)
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
        fake_A = self.netG_A(cond_A)
        self.fake_AA = fake_A.data
        mask_norm = Variable(self.Tensor([1]))
        
        mnorn = mask_norm.expand_as(fake_A)
        mask_tmp = cond_A
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_A = (fake_A * mask_tmp).detach()

        cond_B = Variable(self.mask_B, volatile=True)
        fake_B = self.netG_B(cond_B)
        self.fake_BB = fake_B.data
        mask_tmp = cond_B
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_B = (fake_B * mask_tmp).detach()


        cond_AA = Variable(self.mask_AA)
        cond_AB = Variable(self.mask_AB)
        fake_AB = self.netG_B(cond_A)
        mask_tmp = cond_A
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_AB = (fake_AB * mask_tmp).detach()
        mask_AC = torch.cat((fake_A, fake_AB, cond_AA, cond_AB), 1)
        fake_AC = self.netG_C(mask_AC)
        fake_AC = (fake_AC * mask_tmp).detach()
        self.fake_AB = fake_AB.data
        self.fake_AC = fake_AC.data


        cond_BA = Variable(self.mask_BA)
        cond_BB = Variable(self.mask_BB)
        fake_BA = self.netG_A(cond_B)
        mask_tmp = cond_B
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_BA = (fake_BA * mask_tmp).detach()
        mask_BC = torch.cat((fake_BA, fake_B, cond_BA, cond_BB), 1)
        fake_BC = self.netG_C(mask_BC)
        fake_BC = (fake_BC * mask_tmp).detach()
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
        fake_AC = self.fake_A_pool.query(self.fake_AC)
        fake_BC = self.fake_B_pool.query(self.fake_BC)
        fake_AA = Variable(self.fake_AA)
        fake_BB = Variable(self.fake_BB)
        loss_D_C  = self.backward_D_basic(self.netD_C, fake_AA, fake_AC)
        loss_D_C += self.backward_D_basic(self.netD_C, fake_BB, fake_BC)
        loss_D_C *= 0.5
        self.loss_D_C = loss_D_C.data[0]

    def backward_GC(self):
        # mod: A, B, C
        lambda_idt = self.opt.identity
        lambda_ = self.opt.lambda_C
        
        
        fake_AA = self.netG_A(self.cond_A)
        fake_AB = self.netG_B(self.cond_A)

        
        mnorn = self.mask_norm.expand_as(fake_AA)
        mask_tmp = self.cond_AA
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        real_AA = (fake_AA * mask_tmp).detach()
        
        mask_tmp = self.cond_AB
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        real_AB = (fake_AB * mask_tmp).detach()
        # real_AA = fake_AA * (torch.cat((self.cond_AA, self.cond_AA, self.cond_AA), 1) +
        #                      self.mask_norm.expand_as(fake_AA)) / \
        #                     (self.mask_norm * 2).expand_as(fake_AA)
        # real_AB = fake_AB * (torch.cat((self.cond_AB, self.cond_AB, self.cond_AB), 1) +
        #                      self.mask_norm.expand_as(fake_AA)) / \
        #                     (self.mask_norm * 2).expand_as(fake_AA)
        # 
        fake_BA = self.netG_A(self.cond_B)
        fake_BB = self.netG_B(self.cond_B)
        
        mask_tmp = self.cond_BB
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        real_BB = (fake_BB * mask_tmp).detach()
        
        mask_tmp = self.cond_BA
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        real_BA = (fake_BA * mask_tmp).detach()
        # real_BB = fake_BB * (torch.cat((self.cond_BB, self.cond_BB, self.cond_BB), 1) +
        #           self.mask_norm.expand_as(fake_BB)) / \
        #         (self.mask_norm * 2).expand_as(fake_BB)
        # real_BA = fake_BA * (torch.cat((self.cond_BA, self.cond_BA, self.cond_BA), 1) +
        #           self.mask_norm.expand_as(fake_BB)) / \
        #         (self.mask_norm * 2).expand_as(fake_BB)
            
        mask_tmp = self.cond_A
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_AA = (fake_AA * mask_tmp).detach()
        fake_AB = (fake_AB * mask_tmp).detach()
        mask_tmp = self.cond_B
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_BB = (fake_BB * mask_tmp).detach()
        fake_BA = (fake_BA * mask_tmp).detach()
        mask_AC = torch.cat((fake_AA, fake_AB, self.cond_AA, self.cond_AB), 1)
        mask_BC = torch.cat((fake_BA, fake_BB, self.cond_BA, self.cond_BB), 1)
        
        

        # GAN loss D_C(G_C(CA))
        fake_AC = self.netG_C(mask_AC)
        pred_fake = self.netD_C(fake_AC)
        loss_G_AC = self.criterionGAN(pred_fake, True)
        
        # GAN loss D_C(G_C(CB))
        fake_BC = self.netG_C(mask_BC)
        pred_fake = self.netD_C(fake_BC)
        loss_G_BC = self.criterionGAN(pred_fake, True)

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if mask_AC is fed.

            mask_tmp = self.cond_AB
            mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
            idt_AC = fake_AC * mask_tmp
            # idt_AC = fake_AC * (torch.cat((self.cond_AB, self.cond_AB, self.cond_AB), 1) +
            #                     self.mask_norm.expand_as(fake_AA)) / \
            #                    (self.mask_norm * 2).expand_as(fake_AA)
            loss_idt_AC = self.criterionIdt(idt_AC, real_AB) * lambda_ * lambda_idt

            mask_tmp = self.cond_AA
            mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
            idt_AC = fake_AC * mask_tmp
            # idt_AC = fake_AC * (torch.cat((self.cond_AA, self.cond_AA, self.cond_AA), 1) +
            #                    self.mask_norm.expand_as(fake_AA)) / \
            #                   (self.mask_norm * 2).expand_as(fake_AA)
            loss_idt_AC += self.criterionIdt(idt_AC, real_AA) * lambda_ * lambda_idt

            # G_A should be identity if mask_BC is fed.
            
            mask_tmp = self.cond_BA
            mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
            idt_BC = fake_BC * mask_tmp
            # idt_BC = fake_AC * (torch.cat((self.cond_BA, self.cond_BA, self.cond_BA), 1) +
            #                     self.mask_norm.expand_as(fake_BB)) / \
            #                    (self.mask_norm * 2).expand_as(fake_BB)
            loss_idt_BC = self.criterionIdt(idt_BC, real_BA) * lambda_ * lambda_idt
            mask_tmp = self.cond_BB
            mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
            idt_BC = fake_BC * mask_tmp
            # idt_BC = fake_AC * (torch.cat((self.cond_BB, self.cond_BB, self.cond_BB), 1) +
            #                    self.mask_norm.expand_as(fake_BB)) / \
            #                   (self.mask_norm * 2).expand_as(fake_BB)
            loss_idt_BC += self.criterionIdt(idt_BC, real_BB) * lambda_ * lambda_idt

            self.idt_AC = idt_AC.data
            self.idt_BC = idt_BC.data
            self.loss_idt_AC = loss_idt_AC.data[0]
            self.loss_idt_BC = loss_idt_BC.data[0]
        else:
            loss_idt_AC = 0
            loss_idt_BC = 0
            self.loss_idt_AC = 0
            self.loss_idt_BC = 0
        
        # combined loss
        loss_G = loss_G_AC + loss_G_BC + loss_idt_AC + loss_idt_BC
        loss_G.backward()
        
        
        self.fake_AA = fake_AA.data
        self.fake_AB = fake_AB.data
        self.fake_BA = fake_BA.data
        self.fake_BB = fake_BB.data
        self.fake_AC = fake_AC.data
        self.fake_BC = fake_BC.data
        self.loss_G_AC = loss_G_AC.data[0]
        self.loss_G_BC = loss_G_BC.data[0]
    
    def backward_G(self):
        # mod: A, B
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        
        # GAN loss D_A(G_A(A))
        fake_A = self.netG_A(self.cond_A)
        mnorn = self.mask_norm.expand_as(fake_A)
        mask_tmp = self.cond_A
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_A = fake_A * mask_tmp
        pred_fake = self.netD_A(fake_A)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_B = self.netG_B(self.cond_B)
        mask_tmp = self.cond_B
        mask_tmp = (torch.cat((mask_tmp, mask_tmp, mask_tmp), 1) + mnorn) / (mnorn + mnorn)
        fake_B = fake_B * mask_tmp
        pred_fake = self.netD_B(fake_B)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if mask_A is fed.
            loss_idt_A = self.criterionIdt(fake_A, self.real_A) * lambda_A * lambda_idt
            # G_A should be identity if real_B is fed.
            loss_idt_B = self.criterionIdt(fake_B, self.real_B) * lambda_B * lambda_idt

            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
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
        if self.isG3:
            # G_C
            self.optimizer_GC.zero_grad()
            self.backward_GC()
            self.optimizer_GC.step()
            # D_C
            self.optimizer_D_C.zero_grad()
            self.backward_D_C()
            self.optimizer_D_C.step()
        else:
            # G_A, G_B
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # D_A
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
            # D_B
            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()

    def get_current_errors(self):
    
        if self.isG3:
            ret_errors = OrderedDict([('D_C', self.loss_D_C), ('G_AC', self.loss_G_AC), 
                                      ('G_BC', self.loss_G_BC)])
            if self.opt.identity > 0.0:
                ret_errors['idt_AC'] = self.loss_idt_AC
                ret_errors['idt_BC'] = self.loss_idt_BC
        else:
            ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A),
                                     ('D_B', self.loss_D_B), ('G_B', self.loss_G_B)])
            if self.opt.identity > 0.0:
                ret_errors['idt_A'] = self.loss_idt_A
                ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        cond_A = util.tensor2im(self.mask_A)
        cond_B = util.tensor2im(self.mask_B)
        
        if self.isG3:
            cond_AA = util.tensor2im(self.mask_AA)
            cond_BB = util.tensor2im(self.mask_BB)
            fake_AA = util.tensor2im(self.fake_AA)
            fake_BB = util.tensor2im(self.fake_BB)
            fake_AB = util.tensor2im(self.fake_AB)
            fake_BA = util.tensor2im(self.fake_BA)
            fake_AC = util.tensor2im(self.fake_AC)
            fake_BC = util.tensor2im(self.fake_BC)

            ret_visuals = OrderedDict([('fake_AA', fake_AA), ('fake_BB', fake_BB),
                                       ('cond_A', cond_A), ('cond_AA', cond_AA),
                                       ('cond_B', cond_B), ('cond_BB', cond_BB),
                                       ('fake_AB', fake_AB), ('fake_BA', fake_BA),
                                       ('fake_AC', fake_AC), ('fake_BC', fake_BC)])

            if self.opt.isTrain and self.opt.identity > 0.0:
                ret_visuals['idt_AC'] = util.tensor2im(self.idt_AC)
                ret_visuals['idt_BC'] = util.tensor2im(self.idt_BC)
        else:
            fake_A = util.tensor2im(self.fake_A)
            fake_B = util.tensor2im(self.fake_B)
            real_A = util.tensor2im(self.real_A.data)
            real_B = util.tensor2im(self.real_B.data)
            input_A = util.tensor2im(self.input_A)
            input_B = util.tensor2im(self.input_B)

            ret_visuals = OrderedDict([('fake_A', fake_A), ('fake_B', fake_B),
                                       ('cond_A', cond_A), ('cond_B', cond_B),
                                       ('real_A', real_A), ('real_B', real_B),
                                       ('input_A', input_A), ('input_B', input_B)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        if self.isG3:
            self.save_network(self.netG_C, 'G_C', label, self.gpu_ids)
            self.save_network(self.netD_C, 'D_C', label, self.gpu_ids)
        else:
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
            self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
