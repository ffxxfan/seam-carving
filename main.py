# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/14 21:00
# @Author  : Fengjie Fan
# @File    : seam_carving.py
import cv2
import numpy as np

import seam_carving
import matplotlib.pyplot as plt
import add_local


def drew_seam_in_image(image, path):
    """
    该函数根据给定路径在输入图像的副本上绘制红色接缝。

    Args:
      image: 我们要在其上绘制接缝的原始图像。
      path: `path` 参数是一维 numpy 数组，表示要在图像中绘制的接缝。数组中的每个元素代表该行中作为接缝一部分的像素的列索引。
    """
    image_copy = np.copy(image)
    for row in range(path.size):
        col = path[row]
        # 一条线可能不明显，这些将周围五个像素都标红
        for index in range(5):
            image_copy[row, col + index] = (0, 0, 255)
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('E:/lovePython/seam-carving-reproduction/images/out/scenery_seam.jpg', image_copy)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 图像无需标记保护区域或移除区域
    NO_MASK = ""

    # 待处理图像路
    input_filename = 'images/in/scenery.jpg'
    # 图像处理完的输出路径
    output_filename = 'E:/lovePython/seam-carving-reproduction/images/out/scenery_result.png'
    # 图像保护部分路径，如果没有则设置为 NO_MASK
    protected_mask = NO_MASK
    # 图像移除部分路径，如果没有则设置为 NO_MASK
    removal_mask = NO_MASK
    # 图像处理后的长宽
    out_height = 900
    out_width = 900

    # 初始化
    my_seam_carving = seam_carving.SeamCarving(input_filename, output_filename, out_height, out_width, protected_mask,
                                               removal_mask)
    # # 计算图像能量并打印图像
    # image_read = cv2.imread(input_filename)
    # energy_map = my_seam_carving.energy_map_without_mask(np.copy(image_read))
    # plt.imshow(energy_map)
    # plt.show()
    # # 查找最佳 seam 并将其在图像上标红
    # path = my_seam_carving.find_seam(energy_map)
    # drew_seam_in_image(image_read, path)
    # cv2.imwrite('E:/lovePython/seam-carving-reproduction/images/out/ocean_energy.jpg', energy_map)

    # 对于给定 energy_map 查找 seam
    # energy_myself = np.array([1, 5, 7, 9, 2, 3, 2, 6, 1, 1, 5, 5, 2, 1, 4, 2, 1, 9, 1, 8]).reshape(4, 5)
    # find_seam = my_seam_carving.find_seam(energy_myself)

    # # 图像重定位
    # my_seam_carving.image_resizing()
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(my_seam_carving.input_image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(my_seam_carving.output_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show()
    # # 存储处理后的图像
    # my_seam_carving.save_image()

    # 物体移除
    my_seam_carving.image_resizing()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(my_seam_carving.input_image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(my_seam_carving.output_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()
    # 存储处理后的图像
    my_seam_carving.save_image()

    # # 将图像进行放大
    # my_seam_carving.image_enlarging()
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(my_seam_carving.input_image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(my_seam_carving.output_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show()
    # # # 存储放大后的图像
    # my_seam_carving.save_image()

    # 将图像进行物体移除


    # # 将非矩形全景图还原为矩形
    # my_add_local = add_local.RectangleWarping(input_filename, output_filename)
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(my_add_local.input_image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(my_add_local.add_local(), cv2.COLOR_BGR2RGB))
    # plt.show()
    # my_add_local.save_image()

    # my_seam_carving.image_resizing_without_mask()
    # my_seam_carving.save_image()
