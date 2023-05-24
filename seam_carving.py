# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/14 23:56
# @Author  : Fengjie Fan
# @File    : seam_carving.py
import math

import cv2
import numpy as np

class SeamCarving:
    def __init__(self, input_filename, output_filename, out_height, out_width, protected_mask_filename='', removal_mask_filename=''):
        """
        此函数为计算接缝路径的图像处理算法初始化参数和常量。

        Args:
          input_filename: 将被处理的输入图像的文件名。
          output_filename: 将保存已处理图像的文件的名称。
          out_height: 输出图像的所需高度。
          out_width: 输出图像的所需宽度。
          protected_mask_filename: 指定输入图像中受保护区域的掩码图像的文件名。这些区域在接缝雕刻过程中不会被移除。如果没有指定保护区，该参数可以留空。
          removal_mask_filename: 指定在接缝雕刻过程中应从输入图像中删除哪些像素的掩码图像的文件名。
        """
        # 初始化参数
        self.removal_mask_filename = removal_mask_filename
        self.protected_mask_filename = protected_mask_filename
        self.out_width = out_width
        self.out_height = out_height
        self.output_filename = output_filename
        self.input_filename = input_filename
        # 读取输入图像并将存储和为 np.float64 格式，并将其复制以便后续处理
        self.input_image = cv2.imread(self.input_filename)
        self.output_image = np.copy(self.input_image)
        # 输入图像的大小
        self.input_image_height = self.input_image.shape[0]
        self.input_image_width = self.input_image.shape[1]
        # 常量，用于计算 seam 路径
        # energy_path[m, n] = UP 表示从 [m - 1, n] 进入当前路径
        self.UPPER_RIGHT = 2
        # energy_path[m, n] = UPPER_LEFT 表示从 [m - 1, n - 1] 进入当前路径
        self.UPPER_LEFT = 1
        # energy_path[m, n] = UPPER_RIGHT 表示从 [m - 1, n - 1] 进入当前路径
        self.UP = 0
        # 表示图像保护部分，energy_map[m][n] = 1 表示保护部分， energy_map[m][n] == -1 表示必须删除部分
        self.protected_mask = 1
        self.removal_mask = -1
        # 表示图像保护部分和移除部分的能量
        self.protected_mask_energy = 1000000000
        self.removal_mask_energy = -1000000000
        # 存储自身的 mask_map
        self.mask_map = np.zeros(self.input_image.shape[: 2])
        # 存储自身的
        # 判断 protected_mask 和 removal_mask 是否存在
        if self.protected_mask_filename == "" and self.removal_mask_filename == "":
            self.has_mask = False
        else:
            self.has_mask = True

    def image_enlarging(self):
        """
        此函数通过先将图像缩小到较小尺寸，然后再将其恢复到原始尺寸来放大图像。
        """
        # 将图像尺寸缩小
        self.image_resizing()
        # 将图像尺寸放大到原尺寸
        self.out_height = self.input_image_height
        self.out_width = self.input_image_width
        self.image_resizing()

    def image_resizing(self):
        """
        此功能在考虑受保护和移除蒙版的同时调整图像大小。
        """
        # 输出图像长宽
        height, width = self.output_image.shape[: 2]
        # 存储 mask 矩阵，mask_map[m][n] == 0 表示无 mask；
        # mask_map[m][n] == 1 表示 protected_mask；mask_map[m][n] == -1 表示 removal_mask
        self.mask_map = np.zeros((height, width))
        # 判断 protected_mask 和 removal_mask 是否存在，若存在则更新 mask 矩阵
        if self.has_mask:
            if self.protected_mask_filename != "":
                self.mask_map = np.copy(
                    self.get_mask_map(self.protected_mask_filename, self.mask_map, update_data=self.protected_mask))
            if self.removal_mask_filename != "":
                # 若存在需要移除的部分，则先移除 removal_mask 覆盖部分，再将其拓展为原图像大小
                self.mask_map = np.copy(
                    self.get_mask_map(self.removal_mask_filename, self.mask_map, update_data=self.removal_mask))
                self.image_resizing_with_mask()
        # 删除 / 插入 seam
        row_number, col_number = self.image_notRemoval_seam_size()
        self.image_resizing_without_mask(row_number, col_number)
        return self.output_image

    def get_mask_map(self, filename, mask_map, update_data):
        """
        此函数读取图像文件，根据图像的非零像素更新掩码图，并返回更新后的掩码图和一行中非零像素的最大数量。

        Args:
          filename: filename 参数是一个字符串，表示需要读取的图像文件的路径。
          mask_map: 表示掩模图的 numpy 数组，其中每个像素值表示图像中的相应像素是否应在接缝雕刻操作期间保留或删除。
          update_data: 将用于更新 mask_map 中相应点的值。
                update_data == -1 是表示为 removal_mask; update_data == 1 时表示为 protected_mask

        Returns:
          包含更新的 mask_map 和 seam_size 的元组。
        """
        # 读取文件图像
        image = cv2.imread(filename).astype(np.float64)
        # mask_map 长宽
        height, width = mask_map.shape[: 2]
        # 图像像素 > 0 的点更新 mask_map 对应点为 update_data
        for row in range(height):
            for col in range(width):
                for piles in range(3):
                    if image[row, col, piles] > 0:
                        mask_map[row, col] = update_data
        return mask_map

    def image_removal_seam_size(self, mask_map):
        """
        此函数根据给定的蒙版贴图计算要删除的水平和垂直接缝的数量。

        Args:
          mask_map: 参数“mask_map”是一个 2D numpy 数组，
                表示图像的二进制掩码。数组中的值为 0 或 1，其中 0 表示应保留在图像中的像素，1 表示应删除的像素。

        Returns:
          一个元组，其中包含给定蒙版贴图中水平接缝的数量和垂直接缝的数量。
        """
        height_number = 0
        width_number = 0
        # 计算水平和垂直 seam 数量
        for row in range(mask_map.shape[0]):
            tmp_width_number = 0
            for col in range(mask_map.shape[1]):
                if mask_map[row][col] == -1:
                    tmp_width_number = tmp_width_number + 1
            # 更新水平 seam 数量
            if tmp_width_number > 0:
                height_number = height_number + 1
            # 更新垂直 seam 数量
            width_number = max(width_number, tmp_width_number)
        # 返回水平和垂直 seam 数量，便于后续操作
        return height_number, width_number

    def image_resizing_with_mask(self):
        # 记录修改后的图像
        resizing_image = np.copy(self.output_image)
        # 记录删除的 seam 数量
        seam_number = 0
        # 当存在需要删除的元素时，删除 seam
        while np.sum(self.mask_map < 0) > 0:
            energy_map = self.energy_map_without_mask(resizing_image)
            energy_map = self.energy_calculation_with_mask(energy_map)
            seam_found = self.find_seam(energy_map)
            # 删除 seam
            resizing_image = self.delete_seams(seam_found, resizing_image)
            self.mask_map = self.delete_mask_map(seam_found, self.mask_map)
            seam_number = seam_number + 1
        self.output_image = np.copy(resizing_image)
        # 存储插入的 seam 路径
        seams_path = np.zeros((seam_number, self.mask_map.shape[0]))
        # 恢复原图像尺寸
        # 查找需要删除的 seam
        for dummy in range(seam_number):
            energy_map = self.energy_map_without_mask(resizing_image)
            energy_map = self.energy_calculation_with_mask(energy_map)
            seam_found = self.find_seam(energy_map)
            seams_path[dummy] = seam_found
            # 删除 seam
            resizing_image = self.delete_seams(seam_found, resizing_image)
            self.mask_map = self.delete_mask_map(seam_found, self.mask_map)
        # 对每列进行排序
        seams_path = np.sort(seams_path, axis=0)
        # 图像插入并拷贝到 output_image
        # 恢复图像
        self.output_image = np.copy(self.insert_seams(seams_path, self.output_image))
        return self.output_image

    def delete_mask_map(self, seam_found, mask_map):
        height, width = mask_map.shape[: 2]
        # 存储删除 seam 后的图像
        output_image_new = np.zeros((height, width - 1))
        # 按行删除 seam
        for row in range(height):
            col = seam_found[row]
            output_image_new[row, :] = np.delete(mask_map[row, :], [col])
        # 返回删除 seam 后的图像
        return output_image_new

    def image_notRemoval_seam_size(self):
        """
        此函数计算要在图像中删除或插入的行数和列数。

        Returns:
          一个元组，其中包含需要删除或插入的行数和列数，以便将输入图像的大小调整为所需的输出大小。
        """
        # 计算需要删除 / 插入的行列数
        row_number = self.input_image_height - self.out_width
        col_number = self.input_image_width - self.out_height
        return row_number, col_number

    def energy_calculation_with_mask(self, energy_map):
        # 能量图的长宽
        height, width = energy_map.shape[: 2]
        for row in range(height):
            for col in range(width):
                if self.mask_map[row, col] == self.protected_mask:
                    energy_map[row, col] = self.protected_mask_energy
                elif self.mask_map[row, col] == self.removal_mask:
                    energy_map[row, col] = self.removal_mask_energy
        return energy_map
        # # 记录能量路径
        # energy_path = np.zeros(height, width)
        # # 记录最大能量
        # max_energy = 0
        # # 更新能量图第一层，动态规划初始化
        # for col in range(width):
        #     if mask_map[0][col] == self.removal_mask:
        #         energy_map[0][col] = energy_map[0][col] * -1
        # # 更新能量图
        #  对能量图怎么计算我也很迷惑，感觉还是直接将需要保护的部分置为一个很大的数字
        #  然后需要移除的部分置为一个很小的数字，python 整数计算不会溢出！我感觉还是我之前想太多了
        # for row in range(1, height):
        #     for col in range(width):
        #         # 若当前元素为保护元素，无需计算能量
        #         if mask_map[row, col] > 0:
        #             continue
        #         else:
        #             # 若当前元素为需要移除元素，则更新当前元素大小为 当前元素值 * -1
        #             if mask_map[row, col] == self.removal_mask:
        #                 mask_map[row, col] = mask_map[row, col] * -1
        #             # 更新当前元素的值




    def image_resizing_without_mask(self, row_number, col_number):
        """
        此函数通过查找和删除/插入低能量像素的接缝，将输入图像的大小调整为指定的输出大小。
        """
        # 记录需要插入的 seam
        # 计算行最优 seam，并修改目标图像 out_image 从 ｍ＊ｎ　到　Ｍ＊ｎ
        self.update_row_or_col(row_number)
        # 图像逆时针旋转 90 度，便于后续查找列的最优 seam
        self.output_image = np.rot90(self.output_image, 1)
        # 计算列最优 seam，并修改目标图像 out_image　从　Ｍ＊ｎ　到　Ｍ＊Ｎ
        self.update_row_or_col(col_number)
        # 将图像顺时针旋转 90 度，恢复图像方向
        self.output_image = np.rot90(self.output_image, -1)
        # 返回图像
        return self.output_image

    def update_row_or_col(self, seam_number):
        """
        此函数通过根据给定 seam 数量,查找并删除或插入 seam 来更新图像。

        Args:
          seam_number: 要从图像中添加或删除的接缝数。正值表示将删除接缝，而负值表示将添加接缝。
        """
        # 原图像行列数
        height, width = self.output_image.shape[: 2]
        # 存储 seam 位置
        seam_paths = np.zeros((abs(seam_number), height))
        # 存储该（x, y）像素是否可以被删除，
        # image_state[x][y] == 0 表示不可删除，image_state[x][y] == 1 表示可以删除
        # image_state = np.zeros((height, width))
        # 待修改的图像
        image_resizing = np.copy(self.output_image)
        # # 待修改的 mask_map
        # mask_map_resizing = np.copy(self.mask_map)
        # 计算列最优 seam，并修改目标图像 out_image　从　m＊ｎ　到　Ｍ＊n 或 m * N
        for dummy in range(abs(seam_number)):
            # 计算能量图
            energy_map = self.energy_map_without_mask(image_resizing)
            # if self.has_mask:
            #     energy_map = self.energy_calculation_with_mask(mask_map_resizing, energy_map)
            seam_found = self.find_seam(energy_map)
            image_resizing = np.copy(self.delete_seams(seam_found, image_resizing))
            # if self.has_mask:
            #     mask_map_resizing = self.delete_seams(seam_found, mask_map_resizing)
            seam_paths[dummy] = seam_found
        # 更新图像
        if seam_number >= 0:
            # 若删除 seam 则直接复制删除后的图像　
            self.output_image = np.copy(image_resizing)
            # self.mask_map = np.copy(mask_map_resizing)
        else:
            # 对每列进行排序
            seam_paths = np.sort(seam_paths, axis=0)
            # 图像插入并拷贝到 output_image
            self.output_image = np.copy(self.insert_seams(seam_paths, self.output_image))
            # if self.has_mask:
                # self.mask_map = self.insert_seams(seam_paths, mask_map_resizing)


    def reachable_map(self, mask_map):
        """
        此函数根据给定的掩码图计算可达图。

        Args:
          mask_map: 代表地图的 2D numpy 数组，其值指示单元格是否被阻塞

        Returns:
          应用 reachable_map 算法后更新的 mask_map。
        """
        for row in range(1, mask_map.shape[0]):
            for col in range(mask_map.shape[1]):
                if mask_map[row, col] != self.removal_mask:
                    if col == 0:
                        mask_map = min(max(mask_map[row - 1, col], 0), max(mask_map[row - 1, col + 1], 0)) + mask_map[row, col]
                    elif col == mask_map.shape[1] - 1:
                        mask_map = min(max(mask_map[row - 1, col], 0), max(mask_map[row - 1, col - 1], 0)) + mask_map[row, col]
                    else:
                        mask_map = min(max(mask_map[row - 1, col], 0), max(mask_map[row - 1, col + 1], 0), max(mask_map[row - 1, col + 1], 0), max(mask_map[row - 1, col - 1], 0)) + mask_map[row, col]
        return mask_map


    def energy_map_without_mask(self, resizing_image):
        """
        此函数在其颜色通道上使用 Scharr 算子计算图像的能量图。

        Args:
          resizing_image: 需要调整大小的输入图像及其能量图需要计算。

        Returns:
          在将输入图像分成蓝色、绿色和红色通道并对每个通道应用 Scharr 算子后，根据输入图像计算的能量图。能量图是每个通道的水平和垂直梯度的绝对值之和。
        """
        b, g, r = cv2.split(resizing_image)
        # energy_b = np.absolute(cv2.Scharr(b, cv2.CV_64F, 1, 0))
        energy_b = np.absolute(cv2.Scharr(b, cv2.CV_64F, 1, 0)) + np.absolute(cv2.Scharr(b, cv2.CV_64F, 0, 1))
        energy_g = np.absolute(cv2.Scharr(b, cv2.CV_64F, 1, 0)) + np.absolute(cv2.Scharr(b, cv2.CV_64F, 0, 1))
        energy_r = np.absolute(cv2.Scharr(b, cv2.CV_64F, 1, 0)) + np.absolute(cv2.Scharr(b, cv2.CV_64F, 0, 1))
        energy_map = energy_r + energy_g + energy_b
        return energy_map

    def find_seam(self, energy_map):
        """
        此函数使用动态规划在能量图中找到最佳接缝。

        Args:
          energy_map: 能量图是一个二维 numpy 数组，表示图像中每个像素的能量。它用于计算从图像中移除的最佳接缝。像
                素的能量是衡量其在图像中重要程度的指标，是根据图像的梯度计算的

        Returns:
          图像的最佳 seam 路径，使用能量图和动态规划计算。
        """
        # 能量累计矩阵长宽
        height, width = energy_map.shape[:2]
        # 用于记录路径
        energy_path = np.zeros(height * width).reshape(height, width)
        # 计算累计矩阵 M(i,j) = e(i,j) + min(M(i - 1, j - 1), M(i - 1, j), M(i - 1, j+1 ))
        for row in range(1, height):
            for col in range(width):
                if col == 0:
                    if energy_map[row - 1, col] > energy_map[row - 1, col + 1]:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col + 1]
                        energy_path[row, col] = self.UPPER_RIGHT
                    else:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col]
                        energy_path[row, col] = self.UP
                elif col == width - 1:
                    if energy_map[row - 1, col] > energy_map[row - 1, col - 1]:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col - 1]
                        energy_path[row, col] = self.UPPER_LEFT
                    else:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col]
                        energy_path[row, col] = self.UP
                else:
                    if energy_map[row - 1, col] <= energy_map[row - 1, col - 1] and energy_map[row - 1, col] <= energy_map[row - 1, col + 1]:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col]
                        energy_path[row, col] = self.UP
                    elif energy_map[row - 1, col - 1] <= energy_map[row - 1, col] and energy_map[row - 1, col - 1] <= energy_map[row - 1, col + 1]:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col - 1]
                        energy_path[row, col] = self.UPPER_LEFT
                    else:
                        energy_map[row, col] = energy_map[row, col] + energy_map[row - 1, col + 1]
                        energy_path[row, col] = self.UPPER_RIGHT
        # 查找最优 seam 路径并返回
        path = self.find_path(energy_map, energy_path)
        return path

    def find_path(self, energy_accumulation_transition, energy_path):
        """
        该函数在给定能量路径的情况下在能量累积转换矩阵中找到最佳 seam 路径。

        Args:
          energy_accumulation_transition: energy_accumulation_transition 是一个二维 numpy
                数组，表示图像的累积能量图。它用于寻找要从图像中移除的最佳 seam 的路径。数组中的值表示图像中每个像素位置的累积能量。
          energy_path: energy_path 参数是一个二维 numpy 数组，表示图像的能量图。它包含有关图像中每
                个像素能量的信息，用于确定要从图像中移除的最佳接缝。 energy_path 数组中的值通常使用梯度计算

        Returns:
          一个 numpy 数组，表示在给定能量累积转换和能量路径中具有最小能量的 seam 路径。
        """
        # 累积能量图长宽
        height, width = energy_accumulation_transition.shape[:2]
        # 记录最小值的 col
        min_energy_col = 0
        min_energy = energy_accumulation_transition[height - 1][0]
        # 记录 seam 路径
        seam_path = np.arange(height)
        # 找到累计能量图的最小的 seam 终点
        for col in range(1, width):
            if energy_accumulation_transition[height - 1][col] < min_energy:
                min_energy = energy_accumulation_transition[height - 1][col]
                min_energy_col = col
        seam_path[height - 1] = min_energy_col
        # 回溯找到最优 seam 路径
        for row in range(height - 1, 0, -1):
            if energy_path[row][min_energy_col] == self.UPPER_LEFT:
                min_energy_col = min_energy_col - 1
            elif energy_path[row][min_energy_col] == self.UPPER_RIGHT:
                min_energy_col = min_energy_col + 1
            seam_path[row - 1] = min_energy_col
        return seam_path

    def insert_seams(self, seams_found, image_resizing):
        """
        此函数通过计算相邻像素的平均值并在适当位置插入新像素来将接缝插入图像。

        Args:
          seams_found: 包含要插入图像中的接缝位置的 numpy 数组。
                seams_found[m][n] 表示需要插入像素的列位置，n 表示插入的行数
                即在 image_resizing[n][seams_found[m][n]] 左边插入像素
          image_resizing: 代表需要调整大小的原始图像的 numpy 数组。

        Returns:
          插入在输入图像中找到的所有接缝后的输出图像。
        """
        height, width, deep = image_resizing.shape[: 3]
        seams_found_height, seams_found_width = seams_found.shape[: 2]
        # 存储插入 seams 后的图像
        output_image_new = np.zeros((height, width + seams_found_height, deep))
        # 插入像素
        for col in range(seams_found_width):
            for piles in range(3):
                image_resizing_row = image_resizing[col, :, piles]
                # 按行插入像素
                for row in range(seams_found_height - 1, -1, -1):
                    # 查找需要插入的像素
                    insert_pixel_position = math.ceil(seams_found[row][col])
                    # 计算需要插入的像素的 values，值为左右邻居像素值相加之和除以二
                    if insert_pixel_position == 0:
                        left_pixel_values = image_resizing[col, insert_pixel_position, piles]
                    else:
                        left_pixel_values = image_resizing[col, insert_pixel_position - 1, piles]
                    # if insert_pixel_position == width - 1:
                    #     right_pixel_values = 0
                    right_pixel_values = image_resizing[col, insert_pixel_position, piles]
                    insert_pixel_values = math.ceil(left_pixel_values / 2 + right_pixel_values / 2)
                    # 插入像素
                    image_resizing_row = np.insert(image_resizing_row, math.ceil(seams_found[row][col]), values=insert_pixel_values)
                # 更新图像行
                output_image_new[col, :, piles] = np.copy(image_resizing_row)
        # 返回插入所有 seam 后的图像
        return output_image_new

    def delete_seams(self, seam_found, image_resizing):
        """
        此函数从图像中删除接缝并返回生成的图像。

        Args:
          seam_found: seam 的路径，是一组一位数组。包含要为图像的每一行删除的像素的列索引。
          image_resizing: 需要通过移除 seam-carving 操作调整大小的原始图像。

        Returns:
          删除在输入图像中找到的 seam 后的输出图像。输出图像是一个 numpy 数组，其高度与输入图像相同，宽度比输入图像小一个像素。
        """
        height, width = image_resizing.shape[: 2]
        # 存储删除 seam 后的图像
        output_image_new = np.zeros((height, width - 1, 3))
        # 按行删除 seam
        for row in range(height):
            col = seam_found[row]
            output_image_new[row, :, 0] = np.delete(image_resizing[row, :, 0], [col])
            output_image_new[row, :, 1] = np.delete(image_resizing[row, :, 1], [col])
            output_image_new[row, :, 2] = np.delete(image_resizing[row, :, 2], [col])
        # 返回删除 seam 后的图像
        return output_image_new

    def save_image(self):
        """
        将处理后的图片 self.output_image 输出到文件 self.output_filename
        """
        cv2.imwrite(self.output_filename, self.output_image)
