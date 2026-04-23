Runway Detection Project Based on YOLOv8 and Ultra Fast Lane Detection v2 and PNP algorithm 

2026.3.15 更新

新增Qt可视化功能，支持单张跑道图片的姿态解算可视化（视频模式还在开发中）

2026.4.12 更新

新增基于XTdrone开源无人机仿真平台的跑道检测 -> 姿态解算 -> 自动着陆引导程序。

XTdrone链接: https://github.com/robin-shaun/XTDrone 

UFLDv2链接:  https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

2026.4.23更新 

新增项目原理，仿真结果与误差分析等

项目原理如下：

<img width="1612" height="906" alt="image" src="https://github.com/user-attachments/assets/35c60933-8f56-41e7-991c-254bef3275a3" />

关于yolo26seg部分，目前完成了网络训练与姿态解算部分，但是在仿真中为了提高系统运行速度所以直接提取的跑道块四个顶点进行的pnp解算，没有尽心真正的yolo26分割

<img width="1604" height="891" alt="image" src="https://github.com/user-attachments/assets/df99b426-a9c9-4ad9-a457-758c91677a74" />

<img width="1626" height="906" alt="image" src="https://github.com/user-attachments/assets/bac61631-0f9e-47ce-89ce-4a05f6b3d63f" />

仿真流程及结果如下：

<img width="1613" height="895" alt="image" src="https://github.com/user-attachments/assets/20a74d1e-3f7b-4019-92d8-75153424f20b" />

<img width="1627" height="904" alt="image" src="https://github.com/user-attachments/assets/0ab36344-1a1b-4a0d-b03a-f142ac307f2c" />



