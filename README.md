Runway Detection Project Based on YOLOv8 and Ultra Fast Lane Detection v2 and PNP algorithm 

2026.3.15 更新

新增Qt可视化功能，支持单张跑道图片的姿态解算可视化（视频模式还在开发中）

2026.4.12 更新

新增基于XTdrone开源无人机仿真平台的跑道检测 -> 姿态解算 -> 自动着陆引导程序。

XTdrone链接: https://github.com/robin-shaun/XTDrone 

UFLDv2链接:  https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

2026.4.23更新 

新增项目原理，仿真结果与误差分析等

仿真运行流程如下：

首先开启gazebo环境

'''
roslaunch px4 outdoor2.launch
'''

项目原理如下：    

<img width="1612" height="906" alt="image" src="https://github.com/user-attachments/assets/35c60933-8f56-41e7-991c-254bef3275a3" />

关于yolo26seg部分，目前完成了网络训练与姿态解算，但是在仿真中为了提高系统运行速度所以直接提取的跑道块四个顶点在HSV空间内按照阈值提取顶点进行的pnp解算，没有进行真正的yolo26分割
需要进一步完善

<img width="1604" height="891" alt="image" src="https://github.com/user-attachments/assets/df99b426-a9c9-4ad9-a457-758c91677a74" />

<img width="1626" height="906" alt="image" src="https://github.com/user-attachments/assets/bac61631-0f9e-47ce-89ce-4a05f6b3d63f" />

仿真流程及结果如下：

<img width="1612" height="890" alt="image" src="https://github.com/user-attachments/assets/924adf8c-fe98-46d4-8093-4723134f57b2" />

可以看到目前高度方向，减去静差后误差可以控制在2m内较为精确，侧偏方向误差较大，需要对解算方法进行一定改进

<img width="1617" height="910" alt="image" src="https://github.com/user-attachments/assets/c9d823c1-c094-49f8-abc6-5aa85f5c3f3c" />









