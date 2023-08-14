import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk, Image
import os
import glob
# 假设有6张原始图片，保存在image_paths列表中

image_paths = glob.glob(r'train/*.tif')
# image_paths = glob.glob(r'TKS root slice-2023.7.17/37-1/*.tif')
length = len(image_paths)

# 图片处理函数
def process_images():
    average = []
    for i, image_path in enumerate(image_paths):
        # 使用OpenCV打开图像
        original_image_cv = cv2.imread(image_path)
        # 预处理以及二值化处理
        # 分割通道
        b, g, r = cv2.split(original_image_cv)
        # 阈值区域划分（可调整，目前为效果最好）
        lower_range = (0, 0, 0)  # 最低像素值
        upper_range = (210, 210, 210)  # 最高像素值
        # 创建掩膜
        mask = cv2.inRange(original_image_cv, lower_range, upper_range)
        # 将掩膜应用到b通道图像上(b通道效果最优，其它通道可自行修改)
        b_masked = cv2.bitwise_and(b, b, mask=mask)
        # g_masked = cv2.bitwise_and(g, g, mask=mask)
        # r_masked = cv2.bitwise_and(r, r, mask=mask)
        # 掩膜后得到想要保留的主题图片
        image1 = cv2.cvtColor(b_masked, cv2.COLOR_BGR2RGB)
        # 中值滤波处理，滤波核以及循环次数可更改来获得较好效果（目前效果较好）
        for j in range(0, 2):  # 4
            if j == 0:
                filtered_image = cv2.medianBlur(image1, 3)
            filtered_image = cv2.medianBlur(filtered_image, 5)
        # 应用二值化处理
        # 拆分单通道
        filtered_image, filtered_image1, filtered_image2 = cv2.split(filtered_image)
        _, binary_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY)

        # 应用闭运算填补断开的部分
        # 先膨胀运算，再腐蚀运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel1)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel1)

        # 输出
        # 显示二值化图像
        binary_image_1 = cv2.resize(binary_image, (680, 512))
        plt.figure(i)
        plt.imshow(binary_image_1,cmap='gray')
        plt.title(image_path)
        plt.show()
        # 进行图片处理操作轮廓划分
        # 查找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outer = []
        inner = []
        center = []
        for contour in contours:
            # 计算面积
            area = cv2.contourArea(contour)
            # 计算周长
            perimeter = cv2.arcLength(contour, True)
            # 计算中心坐标
            moments = cv2.moments(contour)
            # 零阶矩反映图像灰度的总和，一阶矩描述图像的灰度中心
            # m10表示x轴方向上的一阶矩，m01表示y轴方向上的一阶矩，m00表示零阶矩，即轮廓面积
            if moments["m00"] != 0:
                centers = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            # 检查周长是否为0
            if perimeter == 0:
                circularity = 0
            else:
                # 计算圆形度
                circularity = 4 * 3.1415926 * area / perimeter ** 2 # 图像轮廓完整，没有用到相关计算
            if perimeter > 2000 and area > 4100:    # 周长较长，面积大的为外层
                # 最外层一圈 绿
                outer.append(contour)
            else:
                if area > 1200 and perimeter > 100 and  600< centers[0] <850 and 420< centers[1] <680:
                    # 中心部分 黄
                    center.append(contour)
                else:
                    #  内部红点 蓝
                    inner.append(contour)
        # 计算总面积
        # total_area = cv2.contourArea(outer[0])
        if len(outer) == 1:  # 外层断开的情况
            # 霍夫变换检测圆形
            circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30,
                                       minRadius=100, maxRadius=0)
            # minDist：圆心之间的最小距离。过小会增加圆的误判，过大会丢失存在的圆，param1：Canny检测器的高阈值
            # param2：检测阶段圆心的累加器阈值。越小的话，会增加不存在的圆；越大的话，则检测到的圆就更加接近完美的圆形
            # minRadius：检测的最小圆的半径，maxRadius：检测的最大圆的半径
            if circles is not None:
                # 提取最外圈类圆形
                outer_circle = circles[0][0]
                # 获取圆心坐标和半径
                center_x = int(outer_circle[0])
                center_y = int(outer_circle[1])
                radius = int(outer_circle[2])
                radius = radius + 20  # 半径范围可调
                # 增加的半径范围值：该参数控制着当细胞外层断开时，用来计算整个细胞的圆的面积的大小。
                # 细胞外层完好时，无需考虑更改该参数，该参数的调整可根据最后轮廓图
                # 轨迹圆大于整个细胞时，适当减小该值，轨迹圆小于整个细胞时，适当增大该值
                print(radius)
                # 计算圆形面积
                area = 3.1415916 * radius ** 2
                # 绘制圆心和圆轮廓
                image3 = original_image_cv
                cv2.circle(image3, (center_x, center_y), radius, (255, 0, 0), 2)
                cv2.circle(image3, (center_x, center_y), 2, (255, 255, 255), 3)
                # 创建包含圆心坐标的轮廓
                contour = np.array([[[center_x, center_y]]], dtype=np.int32)
                image3 = cv2.resize(image3, (680, 512))
                # cv2.imshow("Circle Detect", image3)
            else:
                print("未检测到圆形")
            total_area = area  # 断开时，总面积为霍夫变换检测圆面积来计算
            outer_area = cv2.contourArea(outer[0])
        else:  # 外层闭合
            total_area = cv2.contourArea(outer[0])
            outer_area = cv2.contourArea(outer[0]) - cv2.contourArea(outer[1])

        # 计算各部分面积
        # outer_area = cv2.contourArea(outer[0]) - cv2.contourArea(outer[1])
        inner_area = sum([cv2.contourArea(cnt) for cnt in inner])
        center_area = sum([cv2.contourArea(cnt) for cnt in center])

        # 计算比例
        if total_area != 0:
            outer_ratio = outer_area / total_area * 100
            inner_ratio = inner_area / total_area * 100
            center_ratio = center_area / total_area * 100
        else:
            outer_ratio = 0
            inner_ratio = 0
            center_ratio = 0
        # 显示结果
        # 绘制轮廓（b,g,r）
        cv2.drawContours(original_image_cv, outer, -1, (0, 255, 0), 3)#绿
        cv2.drawContours(original_image_cv, inner, -1, (255, 0, 0), 2)#蓝
        cv2.drawContours(original_image_cv, center, -1, (0, 255, 255), 3)#黄色
        processed_image_cv = original_image_cv
        # 创建处理后的图片的Tk图片对象
        processed_image = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB))
        processed_image = processed_image.resize((255, 192))#改变尺寸
        processed_image_tk = ImageTk.PhotoImage(processed_image)
        # 在图像输出界面上根据标签显示轮廓图片
        processed_labels[i].configure(image=processed_image_tk)
        processed_labels[i].image = processed_image_tk
        # 更新变量输出的值
        var1_values[i].set("{:.4f}%".format(outer_ratio))
        var2_values[i].set("{:.4f}%".format(inner_ratio))
        var3_values[i].set("{:.4f}%".format(center_ratio))
        average.append(inner_ratio) # 存储
    average  = sum(average) / length # 计算均值
    average_values.set("{:.4f}%".format(average))   # 传值到输出框


# 创建Tk窗口
root = tk.Tk()
root.title("根部切片分析")

# 创建六个原始图片的标签和标题标签
original_labels = []
original_title_labels = []

for i, image_path in enumerate(image_paths):
    original_image_cv = cv2.imread(image_path)
    original_image = Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))
    # 缩放图片，使其适应窗口大小
    original_image = original_image.resize((255, 192))
    original_image_tk = ImageTk.PhotoImage(original_image)
    #图像对象位置
    original_label = tk.Label(root, image=original_image_tk)
    original_label.grid(row=1, column=i, padx=0, pady=10)
    #图像标题
    original_title_label = tk.Label(root, text=os.path.basename(image_path))
    original_title_label.grid(row=0, column=i, padx=0, pady=10)
    #保存图像对象，标题信息
    original_labels.append(original_label)
    original_labels[i].image = original_image_tk
    original_title_labels.append(original_title_label)

# 创建六个处理后图片的标签,文本框,输出框
processed_labels = []#6张轮廓图标签位置信息
processed2_labels = []#二值化图像信息
var1_values = []
var2_values = []
var3_values = []
average = []

for i in range(length):
    #处理得到轮廓图片的标签位置信息
    processed_label = tk.Label(root)
    processed_label.grid(row=2, column=i, padx=0, pady=10)
    processed_labels.append(processed_label)
    #内容输出00
    var1_label = tk.Label(root, text=f"外层面积占比: ")
    var1_label.grid(row=3, column=i, padx=0, pady=5)
    var1_value = tk.StringVar()
    var1_entry = tk.Entry(root, textvariable=var1_value)
    var1_entry.grid(row=4, column=i, padx=10, pady=5)
    var1_values.append(var1_value)

    var2_label = tk.Label(root, text=f"内层面积占比: ")
    var2_label.grid(row=5, column=i, padx=0, pady=5)
    var2_value = tk.StringVar()
    var2_entry = tk.Entry(root, textvariable=var2_value)
    var2_entry.grid(row=6, column=i, padx=10, pady=5)
    var2_values.append(var2_value)

    var3_label = tk.Label(root, text=f"中心面积占比: ")
    var3_label.grid(row=7, column=i, padx=0, pady=5)
    var3_value = tk.StringVar()
    var3_entry = tk.Entry(root, textvariable=var3_value)
    var3_entry.grid(row=8, column=i, padx=10, pady=5)
    var3_values.append(var3_value)

# 创建按钮
button = tk.Button(root, text="轮廓划分", command=process_images)
button.grid(row=9, column=3, columnspan=2, padx=20, pady=20)

#平均值位置输出
average_values_label = tk.Label(root, text=f"内部染色区域平均面积占比: ")
average_values_label.grid(row=9, column=0,columnspan=2, padx=0, pady=0)
average_values = tk.StringVar()
average_values_entry = tk.Entry(root, textvariable=average_values)
average_values_entry.grid(row=9, column=1,columnspan=2, padx=10, pady=0)

# 运行Tk主循环
root.mainloop()
