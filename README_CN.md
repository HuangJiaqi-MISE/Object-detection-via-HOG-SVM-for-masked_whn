# 基于HOG-SVM的口罩检测Demo
这是一个使用定向梯度直方图（HOG）作为特征和支持向量机（SVM）作为分类器的口罩检测应用，使用模式识别课程上提供的Masked-Dataset数据集进行目标检测训练。

程序使用Python编程, 需要以下依赖库才可运行程序:
1. Scikit-learn (For implementing SVM)
2. Scikit-image (For HOG feature extraction)
3. OpenCV (for testing)
4. PIL (Image processing library)
5. Numpy (matrix multiplication)
6. Imutils for Non-maximum suppression

训练集应包括以下内容：
1. 正样本图像：这些图像只包含你试图检测的物体，如：口罩。
2. 负样本图像：这些图像可以包含任何东西，除了你要检测的对象之外。


数据集需要需要分为下列两部分:
nag & pos，其对应程序根目录中的负样本图像存储文件夹与正样本图像文件夹。

Train_HOG_SVM.py：用于HOG_SVM模型的训练。

visualise_HOGdescriptors.py: 可视化计算的梯度在给定的图像（由用户指定）上的样子。

testing_HOG_SVM.py：导入一张图片，使用训练后的model_name.npy模型文件对口罩进行检测

这里给出了两张实际测试时程序运行的截图，pos_result.png显示的是检测有口罩的情况（有绿色的方框标记，并且能看到置信度分数），而nag_result.png显示的检测不戴口罩的情况。

```bash
Author：Huang Jiaqi
Created：2022-05-16
Last updated：2022-05-27
Function：Target detection task for masks using the HOG-SVM model for the Masked-Face-Dataset dataset.
```
