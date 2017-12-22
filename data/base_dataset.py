import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
# for split mask
import torch
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def combineTransform(A_img, mA_img, opt):

    # conbine image and mask to do pair transform
    wid, hei = A_img.size
    AmA = Image.new('RGB', (wid*2, hei))
    AmA.paste(A_img, (0, 0))
    AmA.paste(mA_img, (wid, 0))

    AmA = AmA.resize((opt.loadSize * 2, opt.loadSize), Image.BICUBIC)
    # transform
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    AmA = transform(AmA)

    w_total = AmA.size(2)
    w = int(w_total / 2)
    h = AmA.size(1)
    w_offset = random.randint(0, max(0, w - opt.fineSize - 1))
    h_offset = random.randint(0, max(0, h - opt.fineSize - 1))

    A  = AmA[:, h_offset:h_offset + opt.fineSize,
         w_offset:w_offset + opt.fineSize]
    mA = AmA[:, h_offset:h_offset + opt.fineSize,
         w + w_offset:w + w_offset + opt.fineSize]

    # mA to grayscale
    tmp = mA[0, ...] * 0.299 + mA[1, ...] * 0.587 + mA[2, ...] * 0.114
    mA = tmp.unsqueeze(0)

    return A, mA
def splitMask(mA):

    return _oneNineSplit(mA)
def _oneNineSplit( mA ):
    '''
    random center in range 1/3~2/3 h, 1/3~2/3 w
    random slope deltaH, deltaW
    random assign mask A, B
    :param mA:
    :return:
    '''

    # get bound x, y
    ind = np.where(mA.sum(dim=2).data.numpy() > 0)
    wmin = ind[2].min()
    wmax = ind[2].max()
    ind = np.where(mA.sum(dim=3).data.numpy() > 0)
    hmin = ind[2].min()
    hmax = ind[2].max()

    # gen rand in 1/3 bounding
    wcen = (wmax - wmin) * random.uniform(0, 1) / 3. + (wmax + wmin) * 0.5
    hcen = (hmax - hmin) * random.uniform(0, 1) / 3. + (hmax + hmin) * 0.5

    # gen slope
    wsli = random.uniform(-1, 1)
    hsli = random.uniform(-1, 1)

    # split the mask
    mAA = mAB = mA.data.numpy()
    hei = mA.data.shape[2]
    wid = mA.data.shape[3]
    if wsli == 0:
        mAB[:, :, 0:hcen, :] = 0.
        mAA[:, :, hcen:, :] = 0.
    elif hsli == 0:
        mAB[:, :, :, :wcen] = 0.
        mAA[:, :, :, wcen:] = 0.
    else:
        slope = hsli / wsli
        for h in range(hei):
            for w in range(wid):
                if w == wcen:
                    if slope > 0.:
                        mAA[:, :, h, w] = 0.
                    else:
                        mAB[:, :, h, w] = 0.
                        mAB[:, :, h, w] = 0.
                elif (h - hcen)/(w - wcen) > slope:
                    mAA[:, :, h, w] = 0.
                else:
                    mAB[:, :, h, w] = 0.

    # random change AA, AB
    if random.uniform(0, 1) > 0.5:
        mAA, mAB = mAB, mAA

    # numpy to tensor
    mAA = torch.from_numpy(mAA)
    mAB = torch.from_numpy(mAB)
    return mAA, mAB
def convertMask(m_img):
    mA = None
    mAA = None
    mAB = None
    return mA, mAA, mAB