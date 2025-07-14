import copy
import glob
import random

import cv2
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import imageio
from PIL import Image, ImageOps
import torchvision
from torchvision import transforms
from open3d_render import Open3DRender
import datetime



class RD(object):
    def __init__(self, texture_seed, seed_size,
                 patch_size, interative_step, model, model_name, seed_pool_path, device, block_type, local_search=False, eps=8.0/255,  initial_mode='mannual'):
        """

        :param population_num: 种群数量
        :param initial_mode: 种群初始化策略
        """
        self.patch_size = patch_size
        self.texture_seed = texture_seed
        self.seed_size = seed_size
        self.interative_step = interative_step
        self.render = Open3DRender(obj_file=r'F:\PythonPro\DualAttentionAttack\src\textures\DJS2022.obj')
        tex_mask = cv2.imread('assets/bj80_body_mask.png')
        self.tex_mask = np.logical_or(tex_mask[:, :, 0], tex_mask[:, :, 1], tex_mask[:, :, 2])
        self.obj_conf = 0.1
        self.initial_mode = initial_mode
        self.eps = eps
        self.min_circle_num = 5
        self.max_circle_num=15
        self.min_circle_area_ratio = 0.001
        self.max_circle_area_ratio = 0.01
        self.model_name = model_name
        self.block_type = block_type
        self.local_search = local_search


        # load image under random noise: patch_seed + noise
        img = cv2.imread(texture_seed)
        img = img[::-1, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.seed_img = np.array(img)
        #
        # img = Image.open(texture_seed).convert("RGB")
        # img = ImageOps.fit(img, (2048,2048))

        self.model = model
        self.device = device
        self.seed_pool = self.load_image_pool(seed_pool_path)

    def load_image_pool(self, seed_pool_path):
        try:
            path_list = glob.glob(os.path.join(seed_pool_path, "*.jpg"))
        except:
            path_list = glob.glob(os.path.join(seed_pool_path, "*.png"))

        image_list = []
        for path in path_list:
            img = Image.open(path).convert("RGB")
            image_list.append(img)

        return image_list


    def check_points(self, point_x, point_y, radius, pop_list, gamma=0.8):
        """
        确保生成的圆之间没有重叠
        :param point_lr: 左上角点
        :param point_rh: 右下角点
        :param image:
        :return:
        """
        for pop in pop_list:
            if not self.tex_mask[point_x,point_y]:
                return False
            point_x_tmp, point_y_tmp, radius_tmp, _,= pop
            d_r = abs(radius + radius_tmp)
            d_center = np.sqrt((point_x - point_x_tmp)**2 + (point_y - point_y_tmp)**2)
            if d_center >= d_r: # 如果两个圆的圆心之间的距离大于两个圆的半径之和，则满足条件
                continue
            elif d_center >= d_r * gamma: # 允许两个圆之间存在0.2的重复率
                continue
            else:
                return False
        return True

    def get_rotation(self, point_x, point_y):
        # 分左右
        if point_x < 500:  # 车后备箱
            rotation = [
                [-45, 0, 21],
                [-45, 0, 33],
                [-45, 0, 39],
                [-45, 0, 228],
                [-45, 0, 234],
                [-45, 0, 240],
            ]
        elif point_x >= 1520 and point_y > 565:  # 车引擎盖
            rotation = [
                [-45, 0, 0],
                [-45, 0, 3],
                [-45, 0, 42],
                [-45, 0, 81],
                [-45, 0, 99],
            ]
        elif point_x >= 590 and point_x <= 1210 and point_y > 700:  # 车左侧
            rotation = [
                [-45, 0, 3],
                [-45, 0, 15],
                [-45, 0, 21],
                [-45, 0, 48],
                [-45, 0, 54],
                [-45, 0, 123],
                [-45, 0, 141],
            ]
        elif point_x >= 590 and point_x <= 1210 and point_y <= 325:  # 车右侧
            rotation = [
                [-45, 0, 0],
                [-45, 0, 12],
                [-45, 0, 18],
                [-45, 0, 75],
                [-45, 0, 81],
                [-45, 0, 120],
                [-45, 0, 201],
            ]

        elif point_x >= 590 and point_x <= 1210 and point_y >320 and point_y < 700:  # 车顶
            rotation = [
                [-45, 0, 90],
                [-45, 0, 96],
                [-45, 0, 102],
                [-45, 0, 156],
                [-45, 0, 168],
                [-45, 0, 180],
            ]
        else:
            return None
        return rotation

    def get_fitness_score(self, texture, rotations):
        """get fitess score for single angle"""
        # TODO 分角度设置评估 （分成左右，车顶，然后分角度评估）
        image_list = self.render.get_render_by_rotation(texture, rotations)
        total = 0
        pred_cls = 0
        f_score_list = []

        if "yolo" in self.model_name:
            output = self.model(image_list)
            preds = output.xyxy
            cls_name = [2, 7]

            f_score, pred_cnt, batch_size = self.process_output(preds, cls_name, self.model_name)
            pred_cls += pred_cnt
            f_score_list.append(f_score)
        else:
            image_list = [transforms.ToTensor()(img) for img in image_list]
            images = torch.cat(image_list)
            image = torchvision.transforms.ToTensor()(images)
            output = self.model(image)

            total += len(image_list)
        f_avg_score = np.round(np.mean(f_score_list), 5)

        return f_avg_score, pred_cls

    def make_texture_population(self, texture, pop_list):
        """根据每次搜索的值绘制纹理"""
        for p in pop_list:
            # TODO 在每个位置上，选择攻击效果最好的图片作为当前的位置的贴纸（是否要保证贴纸的多样性？）
            if self.local_search:
                texture_copy = copy.deepcopy(texture)
                best_image_seed = None
                minium_obj_score = np.inf
                minium_pred_cls = np.inf
                unchange = 0

                for tmp_p in pop_list:
                    tmp_point_x, tmp_point_y, tmp_radius, tmp_image_seed = tmp_p
                    if tmp_point_y < 330:  # 位于汽车左侧时，为使显示的图案正向，需要对图片进行顺时针旋转
                        tmp_image_seed = cv2.rotate(tmp_image_seed, cv2.cv2.ROTATE_90_CLOCKWISE)
                    elif tmp_point_y >= 700 and tmp_point_y <= 1024:  # 位于汽车右侧时，为使显示的图案正向，需要对图片进行顺时针旋转
                        tmp_image_seed = cv2.rotate(tmp_image_seed, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # DONE 在合理范围内，再随机旋转特定角度
                    if np.random.rand() < 0.5:
                        h, w = tmp_image_seed.shape[:2]
                        center = (h // 2, w // 2)
                        angle = np.random.randint(-30, 30)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        tmp_image_seed = cv2.warpAffine(tmp_image_seed, M, (w, h))
                    tmp_tex = np.zeros(texture_copy.shape, dtype=np.uint8)
                    if "circle" == self.block_type:
                        tmp_tex[tmp_point_x - tmp_radius:tmp_point_x + tmp_radius, tmp_point_y - tmp_radius: tmp_point_y + tmp_radius, :] = tmp_image_seed
                        texture_copy = np.where(tmp_tex == 255, tmp_tex, texture_copy)
                    elif "black_white_box" == self.block_type:
                        # 为白色块
                        if 255 in tmp_image_seed:
                            tmp_tex[tmp_point_x - tmp_radius:tmp_point_x + tmp_radius,
                            tmp_point_y - tmp_radius: tmp_point_y + tmp_radius, :] = tmp_image_seed
                            texture_copy = np.where(tmp_tex == 255, tmp_tex, texture_copy)
                        else:
                            texture_copy[tmp_point_x - tmp_radius:tmp_point_x + tmp_radius,
                            tmp_point_y - tmp_radius: tmp_point_y + tmp_radius, :] = tmp_image_seed
                    else:
                        # 为黑色块，旋转后不影响
                        tmp_tex[tmp_point_x - tmp_radius:tmp_point_x + tmp_radius, tmp_point_y - tmp_radius: tmp_point_y + tmp_radius, :] = tmp_image_seed
                        texture_copy = np.where(tmp_tex == 255, tmp_tex, texture_copy)

                    rotations = self.get_rotation(tmp_point_x, tmp_point_y)
                    f_avg_score, pred_cls = self.get_fitness_score(texture_copy, rotations)
                    if pred_cls < minium_pred_cls:
                        minium_pred_cls = pred_cls
                        minium_obj_score = f_avg_score
                        best_image_seed = tmp_image_seed
                        unchange = 0
                    else:
                        unchange += 1
                    # 如果没能使攻击效果更好，那么就比较预测得分是否会更低
                    if unchange >= 1 and f_avg_score < minium_obj_score:
                        best_image_seed = tmp_image_seed

                image_seed = best_image_seed
            else:
                point_x, point_y, radius, image_seed = p
                if point_y < 330:  # 位于汽车左侧时，为使显示的图案正向，需要对图片进行顺时针旋转
                    image_seed = cv2.rotate(image_seed, cv2.cv2.ROTATE_90_CLOCKWISE)
                elif point_y >=700 and point_y <= 1024: # 位于汽车右侧时，为使显示的图案正向，需要对图片进行顺时针旋转
                    image_seed = cv2.rotate(image_seed, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                # DONE 在合理范围内，再随机旋转特定角度
                if np.random.rand() < 0.5:
                    h, w = image_seed.shape[:2]
                    center = (h // 2, w//2)
                    angle = np.random.randint(-30,30)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    image_seed = cv2.warpAffine(image_seed, M, (w, h))

            texture[point_x - radius:point_x+radius, point_y-radius: point_y+radius,:] = image_seed
        return texture


    def get_texture(self, texture):
        w, h, c = texture.shape
        h = h // 2
        # h = int(h * (3 / 4))
        texture_area = w * h
        pop_list = []
        circle_num = np.random.randint(self.min_circle_num, self.max_circle_num)
        seed_pool_idx = np.random.choice(list(range(len(self.seed_pool))),circle_num, replace=False)
        for i in range(circle_num):
            ratio = np.random.uniform(self.min_circle_area_ratio, self.max_circle_area_ratio)
            radius = round(np.sqrt(ratio * texture_area))
            point_x = np.random.randint(radius, w - radius)
            point_y = np.random.randint(radius, h - radius)
            # 确保生成的新圆与现有圆的重复小于某个阈值
            while i > 0 and not self.check_points(point_x, point_y, radius, pop_list, gamma=1.0):
                ratio = np.random.uniform(self.min_circle_area_ratio, self.max_circle_area_ratio)
                radius = round(np.sqrt(ratio * texture_area))
                point_x = np.random.randint(radius, w - radius)
                point_y = np.random.randint(radius, h - radius)

            # 自然图像
            if "nature_iamge" == self.block_type:
                image_seed = self.seed_pool[seed_pool_idx[i]]
                image_seed = ImageOps.fit(image_seed, (2 * radius, 2 * radius))
                image_seed = np.array(image_seed)
            # 黑白块 2018 CVPR
            elif "black_white_box" == self.block_type:
                if np.random.rand() < 0.5:
                    image_seed = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
                else:
                    image_seed = np.ones((2 * radius, 2 * radius, 3), dtype=np.uint8) * 255
            # 圆
            elif "circle" == self.block_type:
                red = np.random.randint(0, 255)
                green = np.random.randint(0, 255)
                blue = np.random.randint(0, 255)
                image_seed = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
                cv2.circle(image_seed, (point_x, point_y), radius, (red, green, blue), thickness=-1)
            # 对抗补丁
            elif "adv_patch" == self.block_type:
                #
                image_seed = self.seed_pool[seed_pool_idx[i]]
                image_seed = ImageOps.fit(image_seed, (2 * radius, 2 * radius))
                image_seed = np.array(image_seed)
            # 矩形
            elif "square_block" == self.block_type:
                red = np.random.randint(0, 255)
                green = np.random.randint(0, 255)
                blue = np.random.randint(0, 255)
                tmp_img = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
                image_seed = (red, green, blue) - tmp_img

        pop_list.append((point_x, point_y, radius, image_seed))
        texture = self.make_texture_population(texture, pop_list)
        return texture


    def get_rendered_image_batch(self, tex, batch_size=8):
        rendered_images = self.render.render_batch(tex)
        rendered_images = np.array(rendered_images)
        # Image.fromarray(rendered_images[0]).show()
        def gen_batch(images, batch_size):
            batch_num = images.shape[0] // batch_size
            for i in range(batch_num):
                yield images[i*batch_size: (i+1)*batch_size, :,:,:]
        return gen_batch(rendered_images, batch_size)


    def search(self):
        unchange_step = 0
        best_fintness = np.inf
        best_f_avg_score = np.inf

        f_avg_score, fitness_score, test_data_total = self.calculate_fitness(self.seed_img, save_dir)
        print(f"Clean pred, fitness score (pred num): {fitness_score}, Total: {test_data_total}, obj score: {f_avg_score}")

        for itr in range(self.interative_step):

            start_time = datetime.datetime.now()
            texture_origin = copy.deepcopy(self.seed_img)
            texture_adv = self.get_texture(texture_origin)

            f_avg_score, fitness_score, test_data_total = self.calculate_fitness(texture_adv, save_dir)

            if fitness_score < 0 or unchange_step > 50:
                Image.fromarray(texture_adv).save(f"{save_dir}/texture_before_exit.png")
                print("break: ", itr)
                break

            if best_fintness > fitness_score:
                best_fintness = fitness_score
                Image.fromarray(texture_adv).save(f"{save_dir}/best_texture.png")
                # 如果有新的fitness，那么obj得分将重新技术
                best_f_avg_score = f_avg_score
            elif best_fintness == fitness_score:
                unchange_step += 1
                # 如果识别个数保持不变，则保留可以使相同等级下让Obj得分最低的那个纹理
                if f_avg_score < best_f_avg_score:
                    best_f_avg_score = f_avg_score
                    Image.fromarray(texture_adv).save(f"{save_dir}/best_pred_and_obj_texture.png")
            end_time = datetime.datetime.now() - start_time
            print(f"Itr: {itr}, time: {end_time.seconds}, fitness score (pred num): {best_fintness}, Total: {test_data_total}, obj score: {best_f_avg_score}")

        return texture_adv

    def process_output(self, preds, cls_name, model_name):
        pred_cn = 0
        f_score_list = []
        if "yolo" in model_name:
            for pred in preds:
                is_contain = [True if cls in cls_name else False for cls in pred[:, 5]]
                if sum(is_contain) > 0:
                    f_score = [(p - self.obj_conf).item() for p in pred[is_contain][:, 4]]
                    f_score = [fs for fs in f_score if fs > 0]
                else:
                    f_score = [0]
                flag = 1 if sum(is_contain) >= 1 else 0
                pred_cn += flag
                f_score_list.append(f_score)
        else:
            pass
        f_avg_score = np.round(np.mean(f_score_list), 5)
        return f_avg_score, pred_cn, len(f_score_list)

    def calculate_fitness(self, texture, save_dir_tmp):

        # calculate fitness score
        batch_generator = self.get_rendered_image_batch(texture)

        f_score_sum = []
        pred_cnt_sum = 0
        data_num = 0
        for images in batch_generator:
            with torch.no_grad():
                if "yolo" in self.model_name:
                    try:
                        images = [transforms.ToPILImage()(img) for img in images]
                    except:
                        images = [Image.fromarray(img) for img in images]
                    images[0].save(f"{save_dir_tmp}/rendered_car.png")

                else:
                    # construct data for faster rcnn
                    pass
                    transforms.ToPILImage()(images[0]).save(f"{save_dir_tmp}/rendered_car.png")
                output = model(images)

            if "yolo" in self.model_name:
                preds = output.xyxy
                cls_name = [2,7]
                f_score, pred_cnt, batch_size = self.process_output(preds, cls_name, self.model_name)

                f_score_sum.append(f_score)
                pred_cnt_sum += pred_cnt
                data_num += batch_size
            else:
                cls_name = [3]
                f_score, pred_cnt, batch_size = self.process_output(preds, cls_name, self.model_name)

                f_score_sum.append(f_score)
                pred_cnt_sum += pred_cnt
                data_num += batch_size

        f_avg_score = np.round(np.mean(f_score_sum), 5)
        return f_avg_score, pred_cnt_sum, data_num



if __name__ == "__main__":
    import os
    from torchvision.models import detection
    from data_loader import SimulatorDataset
    from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, sampler
    import time
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    seed_path = 'assets/body.png'
    population_num = 20
    patch_size = 32
    seed_size = 128
    iterative_step = 1000
    image_size = 2048
    # model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - 6, custom                    output = self.model(rendered_img)

    model.to(device)
    model.eval()


    time_str = time.strftime("%m-%d-%H-%M", time.localtime())

    save_dir = f"F:/PythonPro/BlackboxTexture/saved_textures/EXP_NAME_{time_str}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform_img = transforms.Compose([
        transforms.ToTensor()
    ])
    block_type = ["black_white_box", "nature_iamge", "adv_patch", "square_block", "circle"]
    local_search = True

    pde = RD(texture_seed=seed_path,
              seed_size=seed_size,
              patch_size=patch_size,
              model=model,
              model_name="yolo",
              interative_step=iterative_step,
             seed_pool_path = 'seed_image_pools',
             block_type = block_type[0],
             local_search=local_search,
             device=device)
    pde.search()

