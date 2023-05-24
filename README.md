# seam-carving
* [项目背景](#项目背景)
* [安装环境](#安装环境)
* [使用](#使用)
* [贡献](#贡献)
* [参考文献](#参考文献)

## 项目背景

有效地调整图像大小，不仅要使用几何约束，还要考虑图像内容。本项目采用 *seam-carving* 算法，*seam* 是单个图像上从上到下或从左到右的 8 连通的最佳连接路径，通过在一个方向上反复删除或插入 *seam* 的方式，我们可以实现以下功能：

* 改变图像长宽比
* 图片内容放大（先将图像尺寸缩小，再将尺寸放大到原图像大小）
* 物体移除
* 不规则图像扩展为矩形

## 安装环境

* 安装 [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows)
* 安装 [Python3](https://www.python.org/downloads/)，建议 3.7 及以上版本
  * Python 解释器中安装相关依赖包
    * `pip install opencv-python`
    * `pip install matplotlib`

## 使用

> 项目结构如下：
>
> ├─ .idea
> 
> ├─ images 存储图像
> 
> │  ├─ in 存储处理图像
> 
> │  └─ out 存储处理完毕的图像
> 
> ├─ main.py 调用 seam_carving.py 和 add_local.py 实现项目功能
> 
> ├─ seam_carving.py 实现 *改变图像长宽比*、*图片内容放大*，*物体移除* 功能
> 
> ├─ add_local.py 实现 *不规则图像扩展为矩形*

* 改变图像长宽比

  ```python
  # 初始化，自定义相关参数 input_filename, output_filename, out_height, out_width
  my_seam_carving = seam_carving.SeamCarving(input_filename, output_filename, out_height, out_width, protected_mask = '', removal_mask = '')
  # 将图像尺寸长宽分别修改为 out_heigh 和 out_width
  my_seam_carving.image_resizing()
  # 保存处理后的图像
  my_seam_carving.save_image()
  ```

  ![resize.png](https://github.com/ffxxfan/seam-carving/blob/master/images/readme/resize.png)

* 图片内容放大，先将图像尺寸缩小，再将尺寸放大到原图像大小

  ```python
  # 初始化，自定义相关参数 input_filename, output_filename, out_height, out_width
  # 其中 out_height 和 out_width 决定了最终图像内容放大效果
  my_seam_carving = seam_carving.SeamCarving(input_filename, output_filename, out_height, out_width, protected_mask = '', removal_mask = '')
  # 将图像尺寸长宽分别修改为 out_heigh 和 out_width
  my_seam_carving.image_enlarging()
  # 保存处理后的图像
  my_seam_carving.save_image()
  ```

  ![enlarge.png](https://github.com/ffxxfan/seam-carving/blob/master/images/readme/enlarge.png)

* 物体移除

  ```python
  # 初始化，自定义相关参数
  # 其中，protected_mask 表示图像需要保护内容的路径，removal_mask 表示图像需要移除内容的路径
  # protected_mask 和 removal_mask 源的图像必须与 input_filename 源的图像尺寸大小一致
  my_seam_carving = seam_carving.SeamCarving(input_filename, output_filename, out_height, out_width, protected_mask, removal_mask)
  # 将图像尺寸长宽分别修改为 out_heigh 和 out_width
  my_seam_carving.image_resizing()
  # 保存处理后的图像
  my_seam_carving.save_image()
  ```
  > 跑的有点慢，之前跑出来的图片不小心删掉了，下次有空的时候再添加
* 不规则图像扩展为矩形（最好为 *.png* 格式）

  ```python
  # 将非矩形全景图还原为矩形
  my_add_local = add_local.RanctangelWarping(input_filename, output_filename)
  my_add_local.add_local()
  # 保存处理后的图像
  my_add_local.save_image()
  ```

  ![add_local.png](https://github.com/ffxxfan/seam-carving/blob/master/images/readme/add_local.png)

## 贡献

<!-- ALL-CONTRIBUTORS-LIST: START - Do not remove or modify this section -->
ffxxfan
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 参考文献

[1] Avidan S, Shamir A. Seam carving for content-aware image resizing[M]//ACM SIGGRAPH 2007 papers. 2007: 10-es.

[2] He K, Chang H, Sun J. Rectangling panoramic images via warping[J]. ACM Transactions on Graphics (TOG), 2013, 32(4): 1-10.





