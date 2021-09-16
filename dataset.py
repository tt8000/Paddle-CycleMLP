import paddle
import paddle.vision.transforms as T
import numpy as np
from PIL import Image

from ppcls.data.preprocess.ops.autoaugment import ImageNetPolicy
from ppcls.data.preprocess.ops.random_erasing import RandomErasing
from utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CycleMLPdataset(paddle.io.Dataset):

    def __init__(self, img_dir, txtpath, mode='train', transform=None):
        """
        Image classification reading class
        args:
            img_dir: Image folder.
            txtpath: TXT file path.
            transform: Data enhancement
        """
        super(CycleMLPdataset, self).__init__()
        assert mode in ['train', 'val', 'test'], "mode is one of ['train', 'val', 'test]"
        self.mode = mode
        self.transform = transform
        self.data = []
        with open(txtpath, 'r') as f:
            for line in f.readlines():
                if mode != 'test':
                    img_path, label = line.strip().split(' ')
                    self.data.append([img_dir + '/' + img_path, label])
                else:
                    self.data.append(img_dir + '/' + line.strip())
    
    def __getitem__(self, idx):
        if self.mode != 'test':
            img = Image.open(self.data[idx][0]).convert('RGB')
            label = self.data[idx][1]
            if self.transform:
                img = self.transform(img)
            return img.astype('float32'), np.array(label, dtype='int64')
        else:
            img = Image.open(self.data[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img.astype('float32')
    
    def __len__(self):
        
        return len(self.data)


def build_transfrom(is_train, args):

    transform = []
    resize_im = args.input_size > 32

    if is_train:
        transform.extend([
            T.RandomResizedCrop(size=args.input_size, interpolation=args.train_interpolation),
            T.RandomHorizontalFlip(),
            ImageNetPolicy(),
            T.ColorJitter(*([args.color_jitter]*3)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            RandomErasing(EPSILON=args.reprob, mean=IMAGENET_DEFAULT_MEAN)
        ])

        if not resize_im:
            transform.append(T.RandomCrop(args.input_size, padding=4))

    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            transform.append(T.Resize(size, interpolation=args.train_interpolation))
            transform.append(T.CenterCrop(size=args.input_size))

        transform.extend([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    return T.Compose(transform)
