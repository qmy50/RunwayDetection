# Runway Detection Project Based on YOLO， UFLDv2 and PNP

# 2026.3.15 更新

新增Qt可视化功能，支持单张跑道图片的姿态解算可视化

# 2026.4.12 更新

新增基于XTdrone开源无人机仿真平台的跑道检测 -> 姿态解算 -> 自动着陆引导程序。

XTdrone链接: https://github.com/robin-shaun/XTDrone 

UFLDv2链接:  https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

# 2026.4.23更新 

新增项目原理，仿真结果与误差分析等

## 仿真运行流程如下：

首先需要线安装XTdrone开源无人机仿真平台，安装链接及文档为： https://www.yuque.com/woshihenyouxiude/lwkpvm/bl2h1k3sh6dnru0y

开启gazebo环境：

```
roslaunch px4 outdoor2.launch
```

开启飞机通信节点与键盘控制节点
```
python plane_communication.py 0
```
```
python plane_keyboard_control.py 1
```
开启位姿真值获取
```
python get_local_pose.py plane 1
```
在键盘控制节点终端输入v,t,等待飞机起飞.随后按b切换为offboard模式并=退出键盘控制节点

运行自动控制节点，首先控制飞机大致对准机场，随后按v开启视觉自动对准模式，并开启视觉处理节点
```
python plane_keyboard_simple.py
```

```
python detect_img_ros_test.py ../UFLDv2/configs/tusimple_res18.py
```

## 项目原理如下：    

<img width="1612" height="906" alt="image" src="https://github.com/user-attachments/assets/35c60933-8f56-41e7-991c-254bef3275a3" />

<br>
关于yolo26seg部分，目前完成了网络训练与姿态解算，但是在仿真中为了提高系统运行速度所以直接提取的跑道块四个顶点在HSV空间内按照阈值提取顶点进行的pnp解算，没有进行真正的yolo26分割
需要进一步完善
<br>

<img width="1604" height="891" alt="image" src="https://github.com/user-attachments/assets/df99b426-a9c9-4ad9-a457-758c91677a74" />

<img width="1626" height="906" alt="image" src="https://github.com/user-attachments/assets/bac61631-0f9e-47ce-89ce-4a05f6b3d63f" />

## 仿真流程及结果如下：

<img width="1612" height="890" alt="image" src="https://github.com/user-attachments/assets/924adf8c-fe98-46d4-8093-4723134f57b2" />

<br>
可以看到目前高度方向，减去静差后误差可以控制在2m内较为精确，侧偏方向误差较大，需要对解算方法进行一定改进

<img width="1617" height="910" alt="image" src="https://github.com/user-attachments/assets/c9d823c1-c094-49f8-abc6-5aa85f5c3f3c" />

## 项目参考文献及资料:

[1] 孙宇鑫.基于视觉引导的无人机自主导航与着陆方法研究[D].西安电子科技大学,2024.DOI:10.27389/d.cnki.gxadu.2024.000415.

[2] 程晋前.基于视觉辅助的固定翼无人机跑道着陆导航与控制技术研究[D].电子科技大学,2025.DOI:10.27005/d.cnki.gdzku.2025.004039.

[3] Qin, Zequn,Zhang, Pengyi,Li, Xi.Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification[J].IEEE Transactions on Pattern Analysis and Machine Intelligence,2024,46,(5):2555-2568.DOI:10.1109/TPAMI.2022.3182097.

[4] Qin, Zequn,Wang, Huanyu,Li, Xi.Ultra Fast Structure-Aware Deep Lane Detection[C].//16th European Conference on Computer Vision-ECCV-Biennial.2020:276-291.

[5] Ouyang, Daliang,He, Su,Zhang, Guozhong , et al.EFFICIENT MULTI-SCALE ATTENTION MODULE WITH CROSS-SPATIAL LEARNING[C].//IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).2023:1-5.









