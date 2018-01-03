
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'bicycle_gan':
        assert(opt.dataset_mode == 'biunaligned')
        from .bicycle_gan_model import BiCycleGANModel
        model = BiCycleGANModel()
    elif opt.model == 'chimera_gan':
        assert(opt.dataset_mode == 'munaligned')
        from .chimera_gan_model import ChimeraGANModel
        model = ChimeraGANModel()
    elif opt.model == 'chimera2_gan':
        assert(opt.dataset_mode == 'munaligned')
        from .chimera2_gan_model import Chimera2GANModel
        model = Chimera2GANModel()
    elif opt.model == 'Chimera3GANModel':
        assert(opt.dataset_mode == 'munaligned')
        from .chimera3_gan_model import Chimera3GANModel
        model = Chimera3GANModel()
    elif opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
