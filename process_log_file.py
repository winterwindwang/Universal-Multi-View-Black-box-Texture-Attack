import matplotlib.pyplot as plt
import numpy as np
import copy
from glob import glob
import os
import torch
import pickle
import pandas as pd


def plot_image(result_dict, save_dir):
    for key, value in result_dict.items():
        x_axis = value['step_num']
        y_axis = value['f_score']
        file_name = f'{save_dir}/{key}.png'
        plt.figure()
        plt.plot(x_axis, y_axis)
        plt.xlabel("iterative step")
        plt.ylabel("fitness score (obj_score - conf)")
        plt.title(key)
        plt.tight_layout()
        plt.savefig(file_name)
        print(f"Image save file name: {save_dir}")
        # plt.show()

def plot_image_single_file(result_dict, save_dir):
    x_axis = result_dict['Itr']
    y_axis_fitness = result_dict['fitness_score']
    y_axis_obj_score = result_dict['obj_score']

    # 显示fitess score与迭代次数曲线
    plt.plot(x_axis, y_axis_fitness, 'b', label='Fitness score')
    # plt.scatter(x, T, color='b', marker='s')

    plt.xlabel('Itr')
    plt.ylabel('Fitness score', color='black')
    plt.tick_params(axis='y', labelcolor='black')
    # 图例
    plt.legend(loc='upper left')

    # 显示压力与时间曲线
    ax2 = plt.twinx()
    ax2.plot(x_axis, y_axis_obj_score, 'r', label='Obj score')
    # ax2.scatter(x, P, color='r', marker='*', s=50)
    ax2.set_ylabel('Obj score', color='black')
    plt.tick_params(axis='y', labelcolor='black')
    # 图例
    plt.legend(loc='upper right')
    plt.savefig(save_dir, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Image save file name: {save_dir}")


def plot_image_dict(result_dict, save_dir):
    plt.figure()
    for key, value in result_dict.iterrows():
        key_name = value['file_name']
        type_name = ""
        for type in ["natural", "advpatch", "black_white", "square_block", "circle"]:
            if type in key_name:
                type_name = type

        value_data = value['data']

        x_axis = list(range(len(value_data['Itr'])))
        y_axis = value_data['fitness_score']

        plt.plot(x_axis, y_axis, label=type_name)
    plt.xlabel("Itr")
    plt.ylabel("Fitness score")
    # plt.title(key)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_dir, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Image save file name: {save_dir}")


def load_log_file(file_path):
    result_dict = {}
    tmp_dcit = {
        "step_num":[],
        "f_score":[]
    }
    with open(file_path, "r") as  fr:
        for line in fr.readlines():
            if line.strip()=="":
                continue

            line = line.strip('\n')
            step_str = eval(line.split(",")[0].split(":")[1])
            top1_f_score = eval(line.split(",")[1].split(":")[-1].split('[')[1])
            seneor_setting = line.split(",")[-1].split(":")[-1]
            if seneor_setting not in result_dict:
                result_dict[seneor_setting] = copy.deepcopy(tmp_dcit)
            result_dict[seneor_setting]['step_num'].append(step_str)
            result_dict[seneor_setting]['f_score'].append(top1_f_score)
    # print(111)
    return result_dict

def list2dict(list_data):
    res = dict()
    for item in list_data:
        kv = item.split(':')
        if "pred num" in kv[0]:
            key = 'fitness_score'
        else:
            key = kv[0].strip()
        res[key] = kv[1]
    return res

def load_log_file_more_inform(file_path):
    result_dict = {}
    result_dict = {
        "Itr":[],
        "fitness_score":[],
        "Total":[],
        "obj_score":[],
    }
    flag = False
    with open(file_path, "r") as  fr:
        for line in fr.readlines():
            # if "Clean pred" not in line.strip():
            #     continue
            if flag:
                line = line.strip('\n')
                valid_info = [item for item in line.split(",") if ':' in item]
                data_dict = list2dict(valid_info)
                result_dict['Itr'].append(int(data_dict['Itr']))
                result_dict['Total'].append(float(data_dict['Total']))
                result_dict['fitness_score'].append(float(data_dict['fitness_score']))
                result_dict['obj_score'].append(float(data_dict['obj score']))
            if "Clean pred" in line.strip():
                flag = True
                line = line.strip('\n')
                # step_str = eval(line.split(",")[0].split(":")[1])
                valid_info = [item for item in line.split(",") if ':' in item]
                data_dict = list2dict(valid_info)
                result_dict['Itr'].append(00)
                result_dict['Total'].append(float(data_dict['Total']))
                result_dict['fitness_score'].append(float(data_dict['fitness_score']))
                result_dict['obj_score'].append(float(data_dict['obj score']))

    return result_dict

def filter_data(data_dict):
    new_dict = {}
    for key, value in data_dict.items():
        if len(value['Itr']) > 1000:
            new_dict[key] = value
    return new_dict



def get_log_file_from_dir(data_folder, save_dir):

    log_file_list = glob(os.path.join(data_folder, "*.out"))
    log_file_dict = {}

    for i, file_path in enumerate(log_file_list):
        fileanme = os.path.basename(file_path).replace(".out", ".png")
        save_name = os.path.join(save_dir, fileanme)
        try:
            result_dict = load_log_file_more_inform(file_path)
        except:
            print(save_name)

        log_file_dict[save_name] = result_dict
        plot_image_single_file(result_dict, save_name)
        # print(save_name, "Finished!!")

    # 筛选出迭代次数超过1000次的结果
    # new_dict_with_over_1000_step = filter_data(log_file_dict)

    data_df = pd.DataFrame(list(log_file_dict.items()), columns=['file_name', 'data'], index=None)
    data_yolo = data_df[data_df['file_name'].str.contains("yolo")]
    data_yolo = data_yolo[data_yolo['file_name'].str.contains("1024")]
    # plot_image_dict(data_yolo, r"D:\ExperimentResult\BackBoxAttack\RandomSearch\attack_yolo.png")
    print(111)
    ########### save data #############
    # with open(r"D:\ExperimentResult\BackBoxAttack\RandomSearch\data_dcit_1120.pickle", 'wb') as fw:
    #     pickle.dump(log_file_dict, fw, pickle.HIGHEST_PROTOCOL)
    ########### load data #############
    # with open(r"D:\ExperimentResult\BackBoxAttack\RandomSearch\data_dcit_1120.pickle", 'wb') as fr:
    #     result_dict = pickle.load(fr)


def evaluate_rendered_car_yolo(image_dir):
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - 6, custom                    output = self.model(rendered_img)
    yolo.conf=0.45
    path_list = glob(os.path.join(image_dir, "*.png"))
    total_cnt = 0
    asr_cnt = 0
    for path in path_list:
        cls_name = [2]
        output = yolo(path)
        result = output.xyxy[0]
        is_contain = [True if pred[5] in cls_name else False for pred in result]
        if len(is_contain) == 0:
            asr_cnt += 1
        else:
            pass
        total_cnt += 1
    print(f"Total: {total_cnt}, success count: {asr_cnt}, asr: {round(100*asr_cnt / total_cnt, 2)}")

# file_path = r'F:\PythonPro\BlackboxTexture\logs\yolov5x\logs_file_for_yolov5x.out'
# save_dir = r'F:\PythonPro\BlackboxTexture\logs\yolov5x'

#
# log_file_dict = load_log_file(file_path)

# plot_image(log_file_dict, save_dir)
# print("Complete!!")

# eval_path_folder = r'F:\PythonPro\BlackboxTexture\logs\yolov5x\EXP_NAME_09-23-13-10'
# evaluate_rendered_car_yolo(eval_path_folder)

file_path = r'D:\ExperimentResult\BackBoxAttack\RandomSearch\adverssarial_log_files'
save_dir = r'D:\ExperimentResult\BackBoxAttack\RandomSearch\adversarial_log_fiels_plot'
get_log_file_from_dir(file_path, save_dir)