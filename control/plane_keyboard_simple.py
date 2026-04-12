#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys, select, tty, termios
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32,Bool
import numpy as np
import math

# 速度参数（固定翼专用）
V_FORWARD = 8.0    # 前进速度
V_YAW = 2.0        # 转弯速度（偏航率）
V_UP = 2.0         # 升降速度
KP_yaw = 3.0
KP_alt = 3.0
KP_pnp = 0.05

# 当前速度
manul_vel_x = 0.0
manul_vel_z = 0.0
yaw_rate = 0.0
manul_vel_y = 0.0
yaw = 0.0
current_yaw = 0.0
target_offset_x = 0.0
target_offset_y = 0.0
target_offset_pnp = 0.0
land_flag = False

msg = """
--- 固定翼 速度控制 ---
w : 前进
a : 左转
d : 右转
i : 上升
k : 下降
v : 模式切换
CTRL-C 退出
"""
def pose_callback(msg):
    global current_yaw
    q = msg.pose.orientation
    # 四元数转 yaw 角
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    current_yaw = math.atan2(siny_cosp, cosy_cosp)

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def detection_callback(msg):
    global target_offset_x
    target_offset_x = msg.data
    if abs(target_offset_x) < 0.05:
        target_offset_x = 0.0

def detection_callback_vertical(msg):
    global target_offset_y
    target_offset_y = msg.data
    if abs(target_offset_y) < 0.05:
        target_offset_y = 0.0

def detection_callback_pnp(msg):
    global target_offset_pnp
    target_offset_pnp = msg.data
    if abs(target_offset_pnp) < 0.05:
        target_offset_pnp = 0.0

def detection_callback_land(msg):
    global land_flag
    if not land_flag:
            land_flag = msg.data

if __name__ == "__main__":
    auto_mode = False
    settings = termios.tcgetattr(sys.stdin)
    plane_id = "0" if len(sys.argv) < 2 else sys.argv[1]
    
    rospy.init_node('plane_keyboard_vel')
    
    # ✅ 正确： unstamped → 用 Twist
    rospy.Subscriber(f'plane_{plane_id}/mavros/local_position/pose',PoseStamped,pose_callback)
    rospy.Subscriber(f'runway_offset',Float32,detection_callback,queue_size=10)
    rospy.Subscriber(f'runway_offset_vertical',Float32,detection_callback_vertical,queue_size=10)
    rospy.Subscriber('runway_offset_pnp',Float32,detection_callback_pnp,queue_size=10)
    rospy.Subscriber('runway_land',Bool,detection_callback_land,queue_size=10)
    pub_vel = rospy.Publisher(f'plane_{plane_id}/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)
    
    twist = Twist()
    print(msg)
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        key = getKey()
        if key == 'w':
            manul_vel_x = V_FORWARD
            manul_vel_y = 0
            manul_vel_z = 0
            print("→ 前进")
        elif key == 'a':
            manul_vel_y = V_YAW
            print("↩ 左转")
        elif key == 'd':
            manul_vel_y = -V_YAW
            print("↪ 右转")
        elif key == 'i':
            manul_vel_z = -V_UP
            print(f"↑ 上升")
        elif key == 'k':
            manul_vel_z = V_UP
            print(f"↓ 下降")
        elif key == '\x03':
            break
        elif key == 'v':
            auto_mode = not auto_mode
            rospy.logwarn("自动模式开启"if auto_mode else "手动模式开启")
            manul_vel_x = V_FORWARD
            manul_vel_y = 0.0
            manul_vel_z = 0.0
        
        if auto_mode:
            if land_flag:
                rospy.loginfo("进入降落模式")
                vx_body = 0.05*V_FORWARD
                vy_body = 0.0
                vz_ned = 0.5
            else:
                vx_body = V_FORWARD
                rospy.loginfo(f"收到x,y方向偏差:{target_offset_x,target_offset_y}")
                vy_body =  - KP_yaw * target_offset_x
                vy_body -= KP_pnp * target_offset_pnp 
                vz_ned =  KP_alt * target_offset_y
        else:
            vx_body = manul_vel_x
            vy_body = manul_vel_y
            vz_ned = manul_vel_z
        yaw = current_yaw

        vx_ned = vx_body*np.cos(yaw) - vy_body*np.sin(yaw)
        vy_ned = vx_body*np.sin(yaw) + vy_body*np.cos(yaw)

        # ✅ 正确：Twist 没有 .twist
        twist.linear.x = vx_ned      # 机体前向
        # twist.linear.y = manul_vel_y
        twist.linear.y = vy_ned     # 上升/下降
        twist.linear.z = vz_ned
        # twist.angular.z = yaw_rate  # 转弯

        pub_vel.publish(twist)
        rate.sleep()

    termios.tcsetattr(sys.stdin, settings)
