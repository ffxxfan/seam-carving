# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/22 20:34
# @Author  : Fengjie Fan
# @File    : add_local.py
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

import seam_carving


class RectangleWarping:
    """  Rectangling Panoramic Images via Warping 论文部分复现
    完成论文中第一阶段，即使用 add seam 方式实现全景图的 local warping 的功能。
    """
    def __init__(self, input_filename, output_filename):
        self.output_filename = output_filename
        self.input_filename = input_filename
        # 输入的图像
        self.input_image = cv2.imread(input_filename)
        # 输出的图像
        self.output_image = np.copy(self.input_image)
        # 方向不可以随便修改值
        # 表示最长段的方向为图像最左边
        self.BOUNDARY_SEGMENT_LEFT = 2
        # 表示最长段的方向为图像最右边
        self.BOUNDARY_SEGMENT_RIGHT = 0
        # 表示最长段的方向为图像最上方
        self.BOUNDARY_SEGMENT_UP = 1
        # 表示最长段的方向为图像最下方
        self.BOUNDARY_SEGMENT_DOWN = 3
        # 记录最长边界段当前方向
        self.boundary_direction = self.BOUNDARY_SEGMENT_RIGHT
        # 记录当前图像的当前方向
        self.image_direction = self.BOUNDARY_SEGMENT_RIGHT
        # 记录当前最长边界段的起始位置
        self.boundary_begin = 0
        self.boundary_end = 0
        # 掩盖像素值为 [0, 0, 0] 位置的能量，避免计算最优 seam 时计算到它
        self.energy_mask_value = 1000000

    def add_local(self):
        """
        此函数将局部接缝添加到图像，直到它变成矩形。

        Returns:
          添加接缝的修改后的输出图像，直到它变成矩形图像。
        """
        # 通过 add_seam 插入像素，直到将输出的图像变为矩形图像，并返回结果
        while self.has_boundary():
            # 查找最长边界段
            self.find_shape_max()
            # 将图像的最长边界段旋转到最右侧
            self.output_image = np.rot90(self.output_image, self.boundary_direction * -1)
            self.image_direction = self.boundary_direction
            self.boundary_direction = self.BOUNDARY_SEGMENT_RIGHT
            # 查找 seam
            seam_found = self.find_seam()
            # output_image 中添加 seam
            self.output_image = self.add_seam(seam_found)
            # 将图像旋转回原本方向
            self.output_image = np.rot90(self.output_image, self.image_direction)
            self.image_direction = self.BOUNDARY_SEGMENT_RIGHT
        return self.output_image

    def has_boundary(self):
        """
        此函数通过旋转对象并检查右边界来检查对象在任何方向上是否有边界。

        Returns:
          一个布尔值，指示图像是否有边界。
        """
        # 判断是否有边界
        has_boundaries = False
        for direction in range(4):
            if self.has_right_boundary():
                has_boundaries = True
            self.output_image = np.rot90(self.output_image, 1)
        return has_boundaries

    def has_right_boundary(self):
        """
        此函数通过遍历行并检查每行的最后一个像素来检查输出图像的右边界是否存在。

        Returns:
          一个布尔值，指示输出图像的右边界是否存在。如果右边界存在，则函数返回 True，否则返回 False。
        """
        # output_image 的长宽
        height, width = self.output_image.shape[: 2]
        # 判断 output_image 右边是否有边
        for row in range(height):
            if self.output_image[row, width - 1, 0] == 0 \
                and self.output_image[row, width - 1, 1] == 0 \
                    and self.output_image[row, width - 1, 2] == 0:
                return True
        return False


    def add_seam(self, seam_path):
        """
        该函数通过将接缝右侧的所有元素向右移动并在接缝处插入新的像素值来为图像添加接缝。

        Args:
          seam_path: `seam_path` 是一个整数列表，表示要添加到图像中的接缝路径。列表中的每个元素代表将添加接缝的行中像素的列索引。

        Returns:
          插入接缝后更新的输出图像。
        """
        # 图像的宽
        width = self.output_image.shape[1]
        # 插入 seam，所有 seam 右边的元素都向右移动
        for row in range(self.boundary_begin, self.boundary_end):
            insert_position = math.ceil(seam_path[row - self.boundary_begin])
            for tier in range(3):
                # 计算插入像素值
                if insert_position == 0:
                    left_value = self.output_image[row, insert_position, tier]
                else:
                    left_value = self.output_image[row, insert_position - 1, tier]
                cur_value = math.ceil(left_value / 2 + self.output_image[row, insert_position, tier] / 2)
                # cur_value = self.output_image[row, insert_position, tier]
                # seam 右边的元素向右移动
                for col in range(insert_position, width):
                    next_value = self.output_image[row, col, tier]
                    # 元素右移
                    self.output_image[row, col, tier] = cur_value
                    cur_value = next_value
                    # # 若替换元素为 0 则停止右移
                    # if next_value <= 0:
                    #     break
        return self.output_image

    def find_seam(self):
        """
        此功能使用接缝雕刻技术在图像的特定边界内找到最佳接缝路径。

        Returns:
          seam carving 算法在输入图像的指定边界内找到的接缝路径。
        """
        # seam_carving
        my_seam_carving = seam_carving.SeamCarving(self.input_filename, self.output_filename, self.input_image.shape[0], self.input_image.shape[1])
        # 计算能量矩阵
        resizing_image = self.output_image[self.boundary_begin: self.boundary_end]
        energy_map = my_seam_carving.energy_map_without_mask(resizing_image)
        # 将为 [0, 0, 0] 像素大小的位置能量改为 self.energy_mask_value
        for row in range(energy_map.shape[0]):
            for col in range(energy_map.shape[1]):
                if resizing_image[row, col, 0] == 0 \
                    and resizing_image[row, col, 1] == 0 \
                        and resizing_image[row, col, 2] == 0:
                    energy_map[row, col] = self.energy_mask_value
        # 查找最优 seam 路径
        seam_path = my_seam_carving.find_seam(energy_map)
        return seam_path

    def find_shape_max(self):
        """
        该函数在四个方向上找到最长的边界线段并返回起点和终点位置，并记录最长边界的方向

        Returns:
          包含图像中最长边界段的开始和结束位置的元组。
        """
        # 记录图像最长的边界起始和结束位置
        boundary_best_begin = 0
        boundary_best_end = 0
        # 记录图像当前最长边的长度
        boundary_best_len = 0
        # 查找最长边界段并记录当前图像方向
        for direction in range(4):
            boundary_begin, boundary_end = self.find_shape()
            if boundary_end - boundary_begin > boundary_best_len:
                self.boundary_direction = direction
                boundary_best_begin = boundary_begin
                boundary_best_end = boundary_end
                boundary_best_len = boundary_end - boundary_begin
            # 顺时针旋转图像 90 度
            self.output_image = np.rot90(self.output_image, -1)
        # 更新当前最长边界的起始位置
        self.boundary_end = boundary_best_end
        self.boundary_begin = boundary_best_begin

    def find_shape(self):
        """
        此函数找到图像右侧最长的边界段，并返回其起始和结束位置，以及图像的高度和宽度。

        Returns:
          图像右侧最长边界段的起始和结束位置。
          boundary_begin, boundary_end 分表表示最长边界段的起始位置，
          如[0,1] 则表示图像 [0, width - 1] 像素值为 0，[1, width - 1] 像素值 > 0
        """
        # 输出图像的长高
        height, width = self.output_image.shape[: 2]
        # 存储当前图像最右边的最长边界段的起点位置和终点位置
        boundary_best_begin = 0
        boundary_best_end = 0
        # 存储当前最长边的长度
        boundary_best_len = 0
        # 计算当前图像最右边最长边界段
        row = 0
        while row < height:
            # 查找边界段的起始位置
            while row < height and np.sum(self.output_image[row, width - 1] > 0) > 0:
                row = row + 1
            boundary_begin = row
            # 查找边界段的结束位置
            while row < height and np.sum(self.output_image[row, width - 1] == 0) == 3:
                row = row + 1
            boundary_end = row
            # 若边界段的长度大于当前最好的长度 boundary_best_len，则更新当前段
            if boundary_end - boundary_begin > boundary_best_len:
                boundary_best_len = boundary_best_end - boundary_best_begin
                boundary_best_end = boundary_end
                boundary_best_begin = boundary_begin
        return boundary_best_begin, boundary_best_end

    def save_image(self):
        """
        将处理后的图片 self.output_image 输出到文件 self.output_filename
        """
        cv2.imwrite(self.output_filename, self.output_image)
