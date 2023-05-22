# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/14 21:00
# @Author  : Fengjie Fan
# @File    : seam_carving.py
import cv2
import numpy as np

import seam_carving
import matplotlib.pyplot as plt

def drew_seam_in_image(image, path):
    image_copy = np.copy(image)
    for row in range(path.size):
        col = path[row]
        image_copy[row, col] = (255, 0, 0)
    plt.imshow(image_copy)
    plt.show()

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 图像无需标记保护区域或移除区域
    NO_MASK = ""

    # 待处理图像路
    input_filename = 'E:/lovePython/seam-carving-reproduction/images/in/madoka2.jpg'
    # 图像处理完的输出路径
    output_filename = ""
    # 图像保护部分路径，如果没有则设置为 NO_MASK
    protected_mask = 'E:/lovePython/seam-carving-reproduction/images/in/remove.jpg'
    # 图像移除部分路径，如果没有则设置为 NO_MASK
    removal_mask = 'E:/lovePython/seam-carving-reproduction/images/in/save.jpg'
    # 图像处理后的长宽
    out_height = 512
    out_width = 512

    # 初始化
    my_seam_carving = seam_carving.SeamCarving(input_filename, output_filename, out_height, out_width, protected_mask, removal_mask)
    # # 计算图像能量并打印图像
    # image_read = cv2.imread(input_filename)
    # energy_map = my_seam_carving.image_energy_calculation(np.copy(image_read))
    # plt.imshow(energy_map)
    # plt.show()
    # # 查找最佳 seam 并将其在图像上标红
    # path = my_seam_carving.find_seam(energy_map)
    # drew_seam_in_image(image_read, path)

    # 对于给定 energy_map 查找 seam
    # energy_myself = np.array([1, 5, 7, 9, 2, 3, 2, 6, 1, 1, 5, 5, 2, 1, 4, 2, 1, 9, 1, 8]).reshape(4, 5)
    # find_seam = my_seam_carving.find_seam(energy_myself)

    # 将图像进行收缩
    my_seam_carving.image_resizing()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(my_seam_carving.input_image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(my_seam_carving.output_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()

    # my_seam_carving.image_resizing_without_mask()
    # my_seam_carving.save_image()
