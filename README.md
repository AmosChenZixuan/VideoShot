# VideoShot

## 2021/3/22
1.在crop image的步骤中省略了draw box，提升了每一帧的处理速度 \
2.之前每一次crop image都需要get tensor，现在提前加载好，稍微提升了运行速度

# TODO
1.现在当没有检测到face or id card时会报错，需要改成跳过并继续执行任务
