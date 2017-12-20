import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, combineTransform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class MUnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_mA = os.path.join(opt.dataroot, 'maskA')
        self.dir_mB = os.path.join(opt.dataroot, 'maskB')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.mA_paths = make_dataset(self.dir_mA)
        self.mB_paths = make_dataset(self.dir_mB)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.mA_paths = sorted(self.mA_paths)
        self.mB_paths = sorted(self.mB_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # transform is in CombineTransform()
        # self.transform = get_transform(opt)
        if hasattr(self.opt, "how_many"):
            self.opt.no_rand = True
        else:
            self.opt.no_rand = False

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        mA_path = self.mA_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.no_rand:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        mB_path = self.mB_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        mA_img = Image.open(A_path).convert('RGB')
        mB_img = Image.open(B_path).convert('RGB')


        # transform
        A, mA = combineTransform(A_img, mA_img, self.opt)
        B, mB = combineTransform(B_img, mB_img, self.opt)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'mA': mA, 'mB': mB,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'MUnalignedDataset'
