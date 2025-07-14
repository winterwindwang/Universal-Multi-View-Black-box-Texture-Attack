import copy
import glob
import random

import cv2
import torch
import torch.nn as nn
import numpy as np
import imageio
from PIL import Image, ImageOps
import torchvision
from torchvision import transforms
from open3d_render import Open3DRender
import datetime
import yaml


class RD(object):
    def __init__(self, args, texture_seed, interative_step, model, model_name, seed_pool_path, device, block_type, local_search):
        """
        :param population_num: 种群数量
        """
        self.args = args
        self.interative_step = interative_step
        self.render = Open3DRender(obj_file=args.obj_file)
        tex_mask = cv2.imread(args.tex_mask)
        tex_mask = tex_mask[::-1, :, :]
        self.tex_mask = np.logical_or(tex_mask[:, :, 0], tex_mask[:, :, 1], tex_mask[:, :, 2])
        self.obj_conf = args.obj_conf
        self.min_circle_num = args.min_circle_num
        self.max_circle_num = args.max_circle_num
        self.min_circle_area_ratio = args.min_circle_area_ratio
        self.max_circle_area_ratio = args.max_circle_area_ratio
        self.model_name = model_name
        self.block_type = block_type
        self.local_search = local_search

        # load image under random noise: patch_seed + noise
        img = cv2.imread(texture_seed)
        img = img[::-1, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.seed_img = np.array(img)

        self.model = model
        self.device = device
        self.seed_pool = self.load_image_pool(seed_pool_path)

    def load_image_pool(self, seed_pool_path):
        try:
            path_list1 = glob.glob(os.path.join(seed_pool_path, "*.jpg"))
            path_list2 = glob.glob(os.path.join(seed_pool_path, "*.png"))
            path_list = path_list1 + path_list2
        except:
            OSError("No such file")

        evaluate_indices = np.random.choice(len(path_list), args.eval_images)
        path_list = np.array(path_list)[evaluate_indices]
        image_list = []
        for path in path_list:
            img = Image.open(path).convert("RGB")
            image_list.append(img)

        return image_list

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
        elif point_x >= 590 and point_x <= 1210 and point_y > 320 and point_y < 700:  # 车顶
            rotation = [
                [-45, 0, 90],
                [-45, 0, 96],
                [-45, 0, 102],
                [-45, 0, 156],
                [-45, 0, 168],
                [-45, 0, 180],
            ]
        else:
            rotation = [
                [-45, 0, 0],
                [-45, 0, 3],
                [-45, 0, 12],
                [-45, 0, 48],
                [-45, 0, 84],
                [-45, 0, 168],
            ]
        return rotation

    def get_fitness_score(self, texture, rotations):
        """get fitess score for single angle"""
        # TODO 分角度设置评估 （分成左右，车顶，然后分角度评估）
        image_list = self.render.get_render_by_rotation(texture, rotations)
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
            images = [transforms.ToTensor()(img).to(self.device) for img in image_list]
            output = self.model(images)
            cls_name = [3, 8]
            f_score, pred_cnt, batch_size = self.process_output(output, cls_name, self.model_name)
            pred_cls += pred_cnt
            f_score_list.append(f_score)
        total = len(image_list)
        f_avg_score = np.round(np.mean(f_score_list), 5)
        pAt05 = np.round(pred_cls / total, 5)
        return f_avg_score, pAt05

    def local_search_for_best_image_seed(self, texture, point, radius, images):
        """根据每次搜索的值绘制纹理"""
        point_x, point_y = point
        best_image_seed = None
        minium_obj_score = np.inf
        minium_pAt05 = np.inf
        unchange = 0

        for image in images:
            texture_copy = copy.deepcopy(texture)
            image = ImageOps.fit(image, (2 * radius, 2 * radius))
            image = np.array(image)
            # 位于汽车左侧时，为使显示的图案正向，需要对图片进行顺时针旋转
            if point_y < 325:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # 位于汽车右侧时，为使显示的图案正向，需要对图片进行顺时针旋转
            elif point_y >= 700 and point_y <= 1024 or (point_x > 1210 and point_y > 565):
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # DONE 在合理范围内，再随机旋转特定角度
            h, w, c = image.shape
            if np.random.rand() < 0.5:
                center = (h // 2, w // 2)
                angle = np.random.randint(-30, 30)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))

            texture_copy[point_x - radius:point_x + radius, point_y - radius: point_y + radius, :] = image
            # Image.fromarray(cv2.circle(self.tex_mask*255, (point_y, point_x), radius, (0,255,0), thickness=-1)).show()
            rotations = self.get_rotation(point_x, point_y)

            f_avg_score, pAt05 = self.get_fitness_score(texture_copy, rotations)
            if pAt05 < minium_pAt05:
                minium_pAt05 = pAt05
                minium_obj_score = f_avg_score
                best_image_seed = image
                unchange = 0
            else:
                unchange += 1
            # 如果没能使攻击效果更好，那么就比较预测得分是否会更低
            if unchange >= 1 and f_avg_score < minium_obj_score:
                best_image_seed = image

        return best_image_seed

    def check_points(self, point_x, point_y, radius, pop_list, gamma=0.8):
        """
        确保生成的圆之间没有重叠
        :param point_lr: 左上角点
        :param point_rh: 右下角点
        :param image:
        :return:
        """
        for pop in pop_list:
            if not self.tex_mask[point_x, point_y]:
                return False
            point_x_tmp, point_y_tmp, radius_tmp = pop
            d_r = abs(radius + radius_tmp)
            d_center = np.sqrt((point_x - point_x_tmp) ** 2 + (point_y - point_y_tmp) ** 2)
            if d_center >= d_r:  # 如果两个圆的圆心之间的距离大于两个圆的半径之和，则满足条件
                continue
            elif d_center >= d_r * gamma:  # 允许两个圆之间存在0.2的重复率
                continue
            else:
                return False
        return True

    def get_populuations(self, texture):
        """
        get the population
        :param texture:
        :return: 圆心集合，半径集合， 图片集合
        """
        w, h, c = texture.shape
        h = h // 2
        # h = int(h * (3 / 4))
        texture_area = w * h

        pop_list_tmp = []
        center_points = []
        radiuses = []
        patches = []
        circle_num = np.random.randint(self.min_circle_num, self.max_circle_num)
        seed_pool_idx = np.random.choice(list(range(len(self.seed_pool))), circle_num, replace=False)
        for i in range(circle_num):
            ratio = np.random.uniform(self.min_circle_area_ratio, self.max_circle_area_ratio)
            radius = round(np.sqrt(ratio * texture_area))
            point_x = np.random.randint(radius, w - radius)
            point_y = np.random.randint(radius, h - radius)
            # 确保生成的新圆与现有圆的重复小于某个阈值
            while i > 0 and not self.check_points(point_x, point_y, radius, pop_list_tmp, gamma=1.0):
                ratio = np.random.uniform(self.min_circle_area_ratio, self.max_circle_area_ratio)
                radius = round(np.sqrt(ratio * texture_area))
                point_x = np.random.randint(radius, w - radius)
                point_y = np.random.randint(radius, h - radius)
            if "nature_image" == self.block_type or "adv_patch" == self.block_type or "animals" == self.block_type:
                image_seed = self.seed_pool[seed_pool_idx[i]]
            else:
                # 黑白块 2018 CVPR
                if "black_white" == self.block_type:
                    if np.random.rand() < 0.5:
                        image_seed = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
                    else:
                        image_seed = np.ones((2 * radius, 2 * radius, 3), dtype=np.uint8) * 255
                # 圆
                elif "circle" == self.block_type:
                    red = np.random.randint(0, 255)
                    green = np.random.randint(0, 255)
                    blue = np.random.randint(0, 255)
                    image_seed = (red, green, blue)
                # 矩形
                elif "square_attack" == self.block_type:
                    red = np.random.randint(0, 255)
                    green = np.random.randint(0, 255)
                    blue = np.random.randint(0, 255)
                    tmp_img = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
                    image_seed = (red, green, blue) - tmp_img

            pop_list_tmp.append((point_x, point_y, radius))
            center_points.append((point_x, point_y))
            radiuses.append(radius)
            patches.append(image_seed)
        return center_points, radiuses, patches

    def get_texture(self, texture):
        result_images = []
        centers, radiuses, images = self.get_populuations(texture)

        for i, batch in enumerate(zip(centers, radiuses, images)):
            point, radius, image = batch
            point_x, point_y = point
            # 局部搜索  ==>  搜索最佳图片
            if self.local_search:  # local + global search
                if "nature_image" == self.block_type or "adv_patch" == self.block_type or "animals" == self.block_type:
                    image = self.local_search_for_best_image_seed(texture, point, radius, images)
                    texture[point_x - radius:point_x + radius, point_y - radius: point_y + radius, :] = image
                elif "circle" == self.block_type:
                    color = image
                    cv2.circle(texture, (point_y, point_x), radius, color, thickness=-1)
                else:
                    texture[point_x - radius:point_x + radius, point_y - radius: point_y + radius, :] = image
                result_images.append(image)
            else:  # only global search
                if "nature_image" == self.block_type or "adv_patch" == self.block_type or "animals" == self.block_type:
                    image = ImageOps.fit(image, (2 * radius, 2 * radius))
                    image = np.array(image)
                    # 位于汽车左侧时，为使显示的图案正向，需要对图片进行调整
                    # 顺时针旋转
                    if point_y < 325:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    # 位于汽车右侧时，为使显示的图案正向，需要对图片进行顺时针旋转
                    elif point_y >= 700 and point_y <= 1024 or (point_x > 1210 and point_y > 565):
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # DONE 在合理范围内，再随机旋转特定角度
                    if np.random.rand() < 0.5:
                        h, w = image.shape[:2]
                        center = (h // 2, w // 2)
                        angle = np.random.randint(-30, 30)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        image = cv2.warpAffine(image, M, (w, h))
                    texture[point_x - radius:point_x + radius, point_y - radius: point_y + radius, :] = image
                elif "circle" == self.block_type:
                    color = image
                    cv2.circle(texture, (point_y, point_x), radius, color, thickness=-1)
                else:
                    texture[point_x - radius:point_x + radius, point_y - radius: point_y + radius, :] = image
                result_images.append(image)
        return texture, centers, radiuses, result_images

    def get_rendered_image_batch(self, tex, batch_size=8):
        rendered_images = self.render.render_batch(tex)
        rendered_images = np.array(rendered_images)

        # Image.fromarray(rendered_images[0]).show()
        def gen_batch(images, batch_size):
            batch_num = images.shape[0] // batch_size
            for i in range(batch_num):
                yield images[i * batch_size: (i + 1) * batch_size, :, :, :]

        return gen_batch(rendered_images, batch_size)

    def search(self):
        unchange_total = 0
        best_fintness = np.inf
        best_f_avg_score = np.inf
        f_avg_score, fitness_score, test_data_total = self.calculate_fitness(self.seed_img, save_dir)
        print(
            f"Clean pred, fitness score (pred num): {fitness_score}, Total: {test_data_total}, obj score: {f_avg_score}")

        for itr in range(self.interative_step):

            start_time = datetime.datetime.now()
            texture_origin = copy.deepcopy(self.seed_img)
            texture_adv, centers, radiuses, images = self.get_texture(texture_origin)

            f_avg_score, fitness_score, test_data_total = self.calculate_fitness(texture_adv, save_dir)
            if fitness_score < 0:
                Image.fromarray(texture_adv).save(f"{save_dir}/texture_before_exit.png")
                print("break: ", itr)
                break

            if best_fintness > fitness_score:
                best_fintness = fitness_score
                Image.fromarray(texture_adv).save(f"{save_dir}/best_texture.png")
                np.savez(f"{save_dir}/best_texture.npz", centers=centers, radiuses=radiuses, images=images)
                # 如果有新的fitness，那么obj得分将重新计算
                best_f_avg_score = f_avg_score
                unchange_total = 0
            elif best_fintness == fitness_score:
                # 如果识别个数保持不变，则保留可以使相同等级下让Obj得分最低的那个纹理
                if f_avg_score < best_f_avg_score:
                    best_f_avg_score = f_avg_score
                    unchange_total = 0
                    Image.fromarray(texture_adv).save(f"{save_dir}/best_pred_and_obj_texture.png")
                    np.savez(f"{save_dir}/best_texture.npz", centers=centers, radiuses=radiuses, images=images)
                else:
                    unchange_total += 1
            else:
                unchange_total += 1

            if unchange_total >= 500:
                print(f"break after the result remains unchange of {unchange_total} step: ", itr)
                break

            end_time = datetime.datetime.now() - start_time
            with open(os.path.join(save_dir, "log.txt"), 'a') as f:
                print(
                    f"Itr: {itr}, time: {end_time.seconds}, fitness score (pred num): {best_fintness}, Total: {test_data_total}, obj score: {best_f_avg_score}")
                f.write(f"Itr: {itr}, time: {end_time.seconds}, fitness score (pred num): {best_fintness}, Total: {test_data_total}, obj score: {best_f_avg_score} \n")
        return texture_adv

    def process_output(self, preds, cls_name, model_name):
        pred_cn = 0
        f_score_list = []
        total = 0
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
                f_score_list.extend(f_score)
                total += 1
        else:
            for pred in preds:
                is_contain = [True if cls in cls_name else False for cls in pred['labels']]
                if sum(is_contain) > 0:
                    f_score = [(p - self.obj_conf).item() for p in pred['scores'][is_contain]]
                    f_score = [fs for fs in f_score if fs > 0]
                else:
                    f_score = [0]
                flag = 1 if sum(is_contain) >= 1 else 0
                pred_cn += flag
                f_score_list.extend(f_score)
                total += 1

        f_avg_score = np.round(np.mean(f_score_list), 5)
        return f_avg_score, pred_cn, total

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
                    images = [transforms.ToTensor()(img).to(self.device) for img in images]
                    transforms.ToPILImage()(images[0]).save(f"{save_dir_tmp}/rendered_car.png")
                output = model(images)

            if "yolo" in self.model_name:
                preds = output.xyxy
                cls_name = [2, 7]
                f_score, pred_cnt, batch_size = self.process_output(preds, cls_name, self.model_name)

                f_score_sum.append(f_score)
                pred_cnt_sum += pred_cnt
                data_num += batch_size
            else:
                cls_name = [3, 8]
                f_score, pred_cnt, batch_size = self.process_output(output, cls_name, self.model_name)
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
    import argparse

    # run code: nohup python -u TDE_location_natural_images_open3d_normal.py --model_name "yolo" --block_idx 0 --local_search > train_adversarial_texture_natural_with_local_search_1024.out 2>&1 &
    parser = argparse.ArgumentParser(description="Random Search Parameters")
    parser.add_argument("--save_dir", type=str, default="saved_textures")
    parser.add_argument("--yaml_file", type=str, default="config.yml", help="the settings config")
    parser.add_argument("--block_idx", type=int, default=5,
                        help='["nature_image", "adv_patch", "animals",  "black_white", "square_attack", "circle"]')
    parser.add_argument("--local_search", action="store_true", default=False, help='is local search?')
    known_args, remaining = parser.parse_known_args()

    with open(known_args.yaml_file, 'r', encoding="utf-8") as fr:
        yaml_file = yaml.safe_load(fr)
        parser.set_defaults(**yaml_file)
    args = parser.parse_args(remaining)

    block_type = ["nature_image", "adv_patch", "animals", "black_white", "square_attack", "circle"]

    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    save_dir = f"saved_textures/EXP_NAME_{time_str}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    with open(os.path.join(save_dir, "log.txt"), 'a') as f:
        f.write(block_type[args.block_idx]+args.__str__())
        print(args)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    sys_flag = "win"
    if "win" in sys_flag:
        if "yolo" in args.model_name:
            # model = torch.hub   .load('ultralytics/yolov5', 'yolov5x')
            model = torch.hub.load(r'C:\Users\idrl\.cache\torch\hub\ultralytics_yolov5_master', 'custom',
                                   path=r'C:\Users\idrl\.cache\torch\hub\ultralytics_yolov5_master\yolov5x.pt', source="local")
            model.conf = args.conf_thresh
        elif "faster_rcnn" in args.model_name:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=args.conf_thresh)
        elif "mask_rcnn" in args.model_name:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_score_thresh=args.conf_thresh)
        elif "retina_net" in args.model_name:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, score_thresh=args.conf_thresh)
    else:
        if "yolo" in args.model_name:
            model = torch.hub.load('/home/wdh/pythonpro/ultralytics_yolov5_master', 'custom',
                                   path='/home/wdh/pythonpro/ultralytics_yolov5_master/yolov5x.pt', source="local")
            model.conf = args.conf_thresh
        elif "faster_rcnn" in args.model_name:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=args.conf_thresh)
        elif "mask_rcnn" in args.model_name:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_score_thresh=args.conf_thresh)
        elif "retina_net" in args.model_names:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, score_thresh=args.conf_thresh)

    model.to(device)
    model.eval()

    transform_img = transforms.Compose([
        transforms.ToTensor()
    ])

    seed_pool_path = args.block_type[block_type[args.block_idx]]
    pde = RD(args,
            texture_seed=args.texture_path,
             model=model,
             model_name=args.model_name,
             interative_step=args.iterative_step,
             seed_pool_path=seed_pool_path,
             block_type=block_type[args.block_idx],
             local_search=args.local_search,
             device=device,
             )
    pde.search()

