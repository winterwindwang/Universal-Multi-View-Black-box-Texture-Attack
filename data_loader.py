from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
import imageio
import torch.nn.functional as F
import cv2

def load_fn(img_paht):
    return Image.open(img_paht).convert("RGB")

def load_fn_v1(img_path):
    img = imageio.imread(img_path)
    img = np.asarray(img)
    try:
        if img.shape[2] == 4:
            img = img[:, :, :3]
    except:
        pass
    return img

class SimulatorDataset(Dataset):
    def __init__(self, data_folder, transform_img, image_size=512):
        # self.transform = transform
        self.transform_img = transform_img
        self.image_size = image_size

        datas = []
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            datas.append(file_path)
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        file_path = self.datas[index]
        data = np.load(file_path)
        img = data['img']
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        mask = data['mask']

        if self.transform_img is not None:
            # img = F.interpolate(img[None,...], (self.image_size, self.image_size))[0]
            # mask = F.interpolate(mask[None,...], (self.image_size, self.image_size))[0]
            img = cv2.resize(img.transpose(1,2,0), (self.image_size, self.image_size))
            img = img.transpose(2,0,1)
            mask = cv2.resize(mask, (self.image_size, self.image_size))

        data_dict = {
            "mask": mask,
            "real_img": img,
            "veh_trans": veh_trans,
            "cam_trans":cam_trans,
        }
        return data_dict


class SubsamplePointcloud(object):
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[1], size=self.N)
        data_out[None] = points[:, indices]
        data_out['normals'] = normals[:, indices]

        return data_out

class ResizeImage(object):
    def __init__(self, size, order=1):
        self.size = size
        self.order = order

    def __call__(self, img):
        img_out = resize(img, self.size, order=self.order,
                         clip=False, mode='constant',
                         anti_aliasing=False)
        img_out = img_out.astype(img.dtype)
        return img_out





if __name__ == "__main__":


    from torchvision import transforms
    from torch.utils.data import DataLoader
    data_folder = r'F:\DataSource\DAS\phy_attack\train_files'

    point_path = r'asset/DJS2022_v1.obj'
    texture_path = r'asset/body.png'

    point_subsampling = 2048
    point_transform = [SubsamplePointcloud(point_subsampling)]
    point_transform = transforms.Compose(point_transform)

    transform = transforms.Compose([
        ResizeImage((224, 224), order=0),
        transforms.ToTensor()
    ])

    dataset = SimulatorDataset(data_folder, transform)

    data_loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=True)
    for batch in data_loader:
        print(111)