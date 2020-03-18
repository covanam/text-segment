import torchvision
import torch.utils.data
import os
import h5py
import cv2
import torch


IMG_FOLDER = 'data/imgs/'
MASK_FOLDER = 'data/masks/'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, im_size=(512, 1024)):
        self.im_size = im_size  # width, height
        self.img_paths = os.listdir(IMG_FOLDER)
        self.mask_paths = [x.replace('.jpg', '.h5') for x in self.img_paths]

        self.tf = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        im_path = os.path.join(IMG_FOLDER, self.img_paths[idx])
        mask_path = os.path.join(MASK_FOLDER, self.mask_paths[idx])

        im = cv2.imread(im_path)
        im = cv2.resize(im, self.im_size)

        with h5py.File(mask_path, 'r') as hf:
            mask = hf['data'][:]

        return self.tf(im), self.tf(mask)

    def __len__(self):
        return len(self.img_paths)


def _test():
    data = Dataset()
    print(len(data))

    x = data[len(data)-1]
    print(x[0].shape, x[1].shape)

    m = x[1][x[1] > 0]
    print(torch.max(m), torch.min(m))


if __name__ == '__main__':
    _test()
