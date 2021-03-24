# VideoShot
人脸+身份证检测

# Prerequisite
TensorFlow=1.15.0\
Opencv=3.4.2\
Numpy=1.19.2\
Pandas=1.2.3\
Pillow=8.1.2\
[ID Card Detector](https://github.com/AmosChenZixuan/id-card-detector)\
[Lightwight Face Dectector](https://github.com/hpc203/10kinds-light-face-detector-align-recognition)

# Change Log
## 2021/3/22
1.在crop image的步骤中省略了draw box，提升了每一帧的处理速度 \
2.之前每一次crop image都需要get tensor，现在提前加载好，稍微提升了运行速度

## 2021/3/23
1. 解决了部分测试样本中检测不到人脸的问题 \ 

## 2021/3/24
1. 将所有可调的变量加入到了config.py中 \ 
2. 为代码添加了注释

# TODO
TBD
