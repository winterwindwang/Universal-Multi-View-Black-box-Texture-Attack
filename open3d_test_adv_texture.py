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


def get_fitness_score( texture, rotations):
    """get fitess score for single angle"""
    # TODO 分角度设置评估 （分成左右，车顶，然后分角度评估）
    image_list = render.get_render_by_rotation(texture, rotations)
    pred_cls = 0
    f_score_list = []
    if "yolo" in model_name:
        output = model(image_list)
        preds = output.xyxy
        cls_name = [2, 7]
        f_score, pred_cnt, batch_size = process_output(preds, cls_name, model_name)
        pred_cls += pred_cnt
        f_score_list.append(f_score)
    else:
        images = [transforms.ToTensor()(img).to(device) for img in image_list]
        output = model(images)
        cls_name = [3, 8]
        f_score, pred_cnt, batch_size = process_output(output, cls_name, model_name)
        pred_cls += pred_cnt
        f_score_list.append(f_score)
    total = len(image_list)
    f_avg_score = np.round(np.mean(f_score_list), 5)
    pAt05 = np.round(pred_cls / total, 5)
    return f_avg_score, pAt05


def get_rendered_image_batch(tex, batch_size=8):
        rendered_images = render.render_batch_test(tex)
        rendered_images = np.array(rendered_images)
        # Image.fromarray(rendered_images[0]).show()
        def gen_batch(images, batch_size):
            batch_num = images.shape[0] // batch_size
            for i in range(batch_num):
                yield images[i * batch_size: (i + 1) * batch_size, :, :, :]
        return gen_batch(rendered_images, batch_size)


def process_output( preds, cls_name, model_name):
    pred_cn = 0
    f_score_list = []
    total = 0
    if "yolo" in model_name:
        for pred in preds:
            is_contain = [True if cls in cls_name else False for cls in pred[:, 5]]
            if sum(is_contain) > 0:
                f_score = [p.item() for p in pred[is_contain][:, 4]]
                f_score = [fs for fs in f_score if fs > 0]
            else:
                f_score = [0]
            flag = 1 if sum(is_contain) >= 1 else 0
            pred_cn += flag
            f_score_list.append(f_score)
            total += 1
    else:
        for pred in preds:
            is_contain = [True if cls in cls_name else False for cls in pred['labels']]
            if sum(is_contain) > 0:
                f_score = [p.item() for p in pred['scores'][is_contain]]
                f_score = [fs for fs in f_score if fs > 0]
            else:
                f_score = [0]
            flag = 1 if sum(is_contain) >= 1 else 0
            pred_cn += flag
            f_score_list.extend(f_score)
            total += 1
    f_avg_score = np.round(np.mean(f_score_list), 5)
    return f_avg_score, pred_cn, total


def calculate_fitness(texture):
    # calculate fitness score
    batch_generator = get_rendered_image_batch(texture)

    f_score_sum = []
    pred_cnt_sum = 0
    data_num = 0
    for images in batch_generator:
        with torch.no_grad():
            if "yolo" in model_name:
                try:
                    images = [transforms.ToPILImage()(img) for img in images]
                except:
                    images = [Image.fromarray(img) for img in images]
            else:
                # construct data for faster rcnn
                images = [transforms.ToTensor()(img).to(device) for img in images]
            output = model(images)

        if "yolo" in model_name:
            preds = output.xyxy
            cls_name = [2, 7]
            f_score, pred_cnt, batch_size = process_output(preds, cls_name, model_name)

            f_score_sum.append(f_score)
            pred_cnt_sum += pred_cnt
            data_num += batch_size
        else:
            cls_name = [3, 8]
            f_score, pred_cnt, batch_size = process_output(output, cls_name, model_name)
            f_score_sum.append(f_score)
            pred_cnt_sum += pred_cnt
            data_num += batch_size

    f_avg_score = np.round(np.mean(f_score_sum), 5)
    return f_avg_score, pred_cnt_sum, data_num


def get_model(model_name_list):
    model_list = {}
    for model_name in model_name_list:
        if "yolo" in model_name:
            model = torch.hub.load('ultralytics/yolov5',
                                   'yolov5x')  # or yolov5n - 6, custom                    output = model(rendered_img)
            model.conf = 0.5
        elif "faster_rcnn" in model_name:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5)
        elif "mask_rcnn" in model_name:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5)
        elif "retina_net" in model_name:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, score_thresh=0.5)
        model.to(device)
        model.eval()
        model_list[model_name] = model
    return model_list


if __name__ == "__main__":
    import os
    from torchvision.models import detection
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Random Search Parameters")
    parser.add_argument("--model_name", type=str, default="faster_rcnn")
    parser.add_argument("--sys_flag", type=str, default="win")
    parser.add_argument("--texture_dir", type=str, default=r"D:/ExperimentResult/BackBoxAttack/RandomSearch/adversarial_texture")
    parser.add_argument("--block_idx", type=int, default=0,
                        help='["nature_image", "adv_patch", "black_white_box", "square_block", "circle"]')
    parser.add_argument("--iterative_step", type=int, default=10000, help='maximum iterative steps')
    parser.add_argument("--local_search", action="store_true", default=False, help='is local search?')
    args = parser.parse_args()


    render = Open3DRender(obj_file='assets/DJS2022.obj')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    texture_dict = {
        # "Compared_methods":{
        #     "Compared_origin": f"{args.texture_dir}/Compared_origin.png",
        #     "Compared_CAMOU": f"{args.texture_dir}/Compared_CAMOU.png",
        #     "Compared_DPA": f"{args.texture_dir}/Compared_DPA.bmp",
        #     "Compared_wutong": f"{args.texture_dir}/Compared_wutong.png",
        # },
        # "yolo":{
        #     # "square_block": f"{args.texture_dir}/",
        #     # "black_white_box": f"{args.texture_dir}/",
        #     # "circle": f"{args.texture_dir}/",
        #     "adv_patch": f"{args.texture_dir}/yolov5x_advpatch_12_120.png",
        #     "nature_image": f"{args.texture_dir}/yolov5x_natural_5_120.png",
        # },
        # "faster_rcnn": {
        #     # "square_block": f"{args.texture_dir}/",
        #     # "black_white_box": f"{args.texture_dir}/",
        #     # "circle": f"{args.texture_dir}/",
        #     "adv_patch": f"{args.texture_dir}/fasterrcnn_advpatch_37_120.png",
        #     "nature_image": f"{args.texture_dir}/fasterrcnn_natural_51_120.png",
        # },
        # "mask_rcnn": {
        #     # "square_block": f"{args.texture_dir}/",
        #     # "black_white_box": f"{args.texture_dir}/",
        #     # "circle": f"{args.texture_dir}/",
        #     "adv_patch": f"{args.texture_dir}/maskrcnn_advpatch_59_120.png",
        #     "nature_image": f"{args.texture_dir}/maskrcnn_natural_57_120.png",
        # },
        "retina_net": {
            # "square_block": f"{args.texture_dir}/",
            # "black_white_box": f"{args.texture_dir}/",
            # "circle": f"{args.texture_dir}/",
            # "adv_patch": f"{args.texture_dir}/retina_net_advpatch_9_120.png",
            "nature_image": r"F:\PythonPro\BlackboxTexture\saved_textures\EXP_NAME_11-10-15-17\best_texture.png",
            # "nature_image": f"{args.texture_dir}/retina_net_natural_32_120.png",
        },
    }

    model_name = args.model_name
    sys_flag = "win"
    if "win" in sys_flag:
        if "yolo" in model_name:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - 6, custom                    output = model(rendered_img)
            model.conf = 0.5
        elif "faster_rcnn" in model_name:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5)
        elif "mask_rcnn" in model_name:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5)
        elif "retina_net" in model_name:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, score_thresh=0.5)

    else:
        if "yolo" in model_name:
            model = torch.hub.load('/home/wdh/pythonpro/ultralytics_yolov5_master', 'custom',
                                   path='/home/wdh/pythonpro/ultralytics_yolov5_master/yolov5x.pt', source="local")
            model.conf = 0.5
        elif "faster_rcnn" in model_name:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5)
        elif "mask_rcnn" in model_name:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5)
        elif "retina_net" in model_name:
            model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, score_thresh=0.5)
    model_name = ["yolo", "faster_rcnn", "mask_rcnn", "retina_net"]
    models = get_model(model_name)


    local_search = args.local_search
    block_idx = args.block_idx
    block_type = ["nature_image", "adv_patch", "black_white_box", "square_block", "circle"]
    if "nature_image" == block_type[block_idx]:
        seed_pool_path = 'seed_image_pools/cartoon'
    elif "adv_patch" == block_type[block_idx]:
        seed_pool_path = 'seed_image_pools/adv_patch'
    else:
        seed_pool_path = 'seed_image_pools/cartoon'

    for key, value in texture_dict.items():
        for atk, tex_file in value.items():
            specific_infor = f"White-box model: {key}, texture generated by: {atk}"
            print(specific_infor)
            texture_image = cv2.imread(tex_file)
            # texture_image = texture_image[::-1, :, :]
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            # batch_generator = get_rendered_image_batch(texture_image)

            current_metrics = {}
            for model_name, model in models.items():
                f_avg_score, pred_cnt_sum, data_num = calculate_fitness(texture_image)
                tmp = {
                    "obj_score": f_avg_score,
                    "corrected_num": pred_cnt_sum,
                    "total_num": data_num,
                    "P@0.5": round(100 * pred_cnt_sum / data_num, 2),
                }

                current_metrics[model_name] = tmp

            print(current_metrics, "\n")

