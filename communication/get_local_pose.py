import rospy
from geometry_msgs.msg import PoseStamped, Vector3Stamped,TransformStamped
import sys
from gazebo_msgs.msg import ModelStates
import tf2_ros

vehicle_type = sys.argv[1]
vehicle_num = int(sys.argv[2])
multi_pose_pub = [None]*vehicle_num
multi_speed_pub = [None]*vehicle_num
multi_local_pose = [PoseStamped() for i in range(vehicle_num)]
multi_speed = [Vector3Stamped() for i in range(vehicle_num)]

def gazebo_model_state_callback(msg):
    for vehicle_id in range(vehicle_num):
        id = msg.name.index(vehicle_type+'_'+str(vehicle_id))
        multi_local_pose[vehicle_id].header.stamp = rospy.Time().now()
        multi_local_pose[vehicle_id].header.frame_id = 'map'
        multi_local_pose[vehicle_id].pose = msg.pose[id]
        multi_speed[vehicle_id].header.stamp = rospy.Time().now()
        multi_speed[vehicle_id].header.frame_id = 'map'
        multi_speed[vehicle_id].vector = msg.twist[id]

if __name__ == '__main__':
    rospy.init_node(vehicle_type+'_get_pose_groundtruth')
    gazebo_model_state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, gazebo_model_state_callback,queue_size=1)
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    for i in range(vehicle_num):
        multi_pose_pub[i] = rospy.Publisher(vehicle_type+'_'+str(i)+'/mavros/vision_pose/pose', PoseStamped, queue_size=1)
        multi_speed_pub[i] = rospy.Publisher(vehicle_type+'_'+str(i)+'/mavros/vision_speed/speed', Vector3Stamped, queue_size=1)
        print("Get " + vehicle_type + "_" + str(i) + " groundtruth pose")
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        for i in range(vehicle_num):
            multi_pose_pub[i].publish(multi_local_pose[i])
            multi_speed_pub[i].publish(multi_speed[i])

            t = TransformStamped()
            t.header.stamp = multi_local_pose[i].header.stamp
            t.header.frame_id = 'map'
            t.child_frame_id = vehicle_type + '_' + str(i)+'/base_link'
            t.transform.translation.x = multi_local_pose[i].pose.position.x
            t.transform.translation.y = multi_local_pose[i].pose.position.y
            t.transform.translation.z = multi_local_pose[i].pose.position.z
            t.transform.rotation = multi_local_pose[i].pose.orientation
            tf_broadcaster.sendTransform(t)

        try:
            rate.sleep()
        except:
            continue

