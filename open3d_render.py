import copy
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization as vis
import os
import cv2
import torch
from PIL import Image


class Open3DRender():
    def __init__(self, obj_file, width=800, height=800):
        self.mesh = o3d.io.read_triangle_mesh(obj_file, True)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
        self.mesh.compute_vertex_normals()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height, visible=False)
        self.mesh_origin_coordinate = self.mesh.get_center()

    def get_translation(self):
        translation_list = []
        # for x in range(-800,1001,200):
        #     for y in range(-1000,1001,200):
        #         for z in range(-1000,1000,200):
        #             translation_list.append([x,y,z])
        translation_list = [
            # [-800, 500, 100],
            # [800, 500, 100], # 左下侧
            # [800 , 0, 1000],   #  右上侧
            [0, 0, 0],  # 左上侧
            # [800 , 0, -200],  # 右下侧
            # [-800, 0, 200],  # 左下侧
            # [-800, 0, -200],  # 左上侧

        ]
        return translation_list

    def render_batch(self, tex):
        self.mesh.textures = [o3d.geometry.Image(tex)]
        translations = self.get_translation()
        # translations = [[0, 0, 0]]
        # pitchs = [-44,-45, -50] # 40 和50的视角容易多为俯视视角
        yaws = [-45] # 40 和50的视角容易多为俯视视角
        pitchs = [0] # 40 和50的视角容易多为俯视视角
        rolls = list(range(0, 361, 2)) # yaw, pitch, roll
        image_list = []
        for translation in translations:
            for yaw in yaws:
                for pitch in pitchs:
                    for roll in rolls:
                        rotation = [yaw, pitch, roll]
                        image, R1 = self.get_rotate_render_image(self.vis, self.mesh, rotation, translation)
                        # img = Image.fromarray(image.astype(np.uint8))
                        # location_str = "_".join([str(i) for i in translation])
                        # rotation_str = "_".join([str(i) for i in rotation])
                        # save_dir = os.path.join(r"F:\PythonPro\BlackboxTexture\texture_reference\tmp", location_str)
                        # if not os.path.exists(save_dir):
                        #     os.mkdir(save_dir)
                        # img.save(f'{save_dir}/{rotation_str}.png')
                        image_list.append(image)
        return image_list

    def render_batch_test(self, tex):
        self.mesh.textures = [o3d.geometry.Image(tex)]
        translations = self.get_translation()
        # translations = [[0, 0, 0]]
        # pitchs = [-44,-45, -50] # 40 和50的视角容易多为俯视视角
        yaws = [-45] # 40 和50的视角容易多为俯视视角
        pitchs = [0] # 40 和50的视角容易多为俯视视角
        rolls = list(range(0, 361, 2)) # yaw, pitch, roll
        image_list = []
        for translation in translations:
            for yaw in yaws:
                for pitch in pitchs:
                    for roll in rolls:
                        rotation = [yaw, pitch, roll]
                        image, R1 = self.get_rotate_render_image(self.vis, self.mesh, rotation, translation)
                        image_list.append(image)
        return image_list


    def get_render_by_rotation(self, texture, rotations):
        self.mesh.textures = [o3d.geometry.Image(texture)]
        translations = self.get_translation()
        image_list = []
        for translation in translations:
            for rotation in rotations:
                image, R1 = self.get_rotate_render_image(self.vis, self.mesh, rotation, translation)
                image = Image.fromarray(image)
                image_list.append(image)
        return image_list

    def get_rotate_render_image(self, vis, mesh, rotate_angle, translation):
        R = self.mesh.get_rotation_matrix_from_xyz(rotate_angle)
        mesh_temp = copy.deepcopy(mesh)
        mesh_temp.rotate(R, mesh_temp.get_center())
        vis.add_geometry(mesh_temp)
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.clear_geometries()
        image = np.asarray(image) * 255
        return image.astype(np.uint8), R



if __name__ == "__main__":

    # tex_mask_path = r'F:\PythonPro\DualAttentionAttack\src\textures\body_mask_final.png'
    # mask = cv2.imread(tex_mask_path)
    # mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
    #
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # # mas = mask[:,:,::-1]
    # #
    #
    # mask[mask >2] = 255
    #
    # kernel = np.ones((4, 4), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=4)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # cv2.imwrite(r"F:\PythonPro\DualAttentionAttack\src\textures\bj80_body_mask.png", mask)
    # # mask[mask >= 10] = 255
    # # mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
    # print(11)

    render = Open3DRender(r'F:\PythonPro\DualAttentionAttack\src\textures\DJS2022.obj')


    tex_path = r'F:\PythonPro\DualAttentionAttack\src\textures\body_copy.png'



    tex = cv2.imread(tex_path)
    tex = tex[::-1, :, :]
    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)

    images = render.render_batch(tex, batch_size=12)
    for i, img in enumerate(images):
        pass
    print(11)
