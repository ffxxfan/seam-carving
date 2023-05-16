# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/14 21:00
# @Author  : Fengjie Fan
# @File    : seam_carving.py
import seam_carving


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 图像无需标记保护区域或移除区域
    NO_MASK = ""

    # 待处理图像路径
    input_filename = ""
    # 图像处理完的输出路径
    output_filename = ""
    # 图像保护部分路径，如果没有则设置为 NO_MASK
    protected_mask = NO_MASK
    # 图像移除部分路径，如果没有则设置为 NO_MASK
    removal_mask = NO_MASK
    # 图像处理后的长宽
    out_height = 200
    out_width = 200

    # 初始化
    my_seam_carving = seam_carving.SeamCarving(input_filename, output_filename, out_height, out_width, protected_mask, removal_mask)
    # 图像进行 seam-carving 处理并输出图像
    my_seam_carving.image_resizing_without_mask()
    my_seam_carving.save_image()
