import rosbag
import rospy

bag = rosbag.Bag('nothing-changed.bag')

topic = '/crobat/motor_pwms'

# UNIX時間（秒）で指定
start_time = rospy.Time.from_sec(1768507730.0)  # 例
end_time   = rospy.Time.from_sec(1768507740.0)  # 例

values = []

for topic, msg, t in bag.read_messages(
        topics=[topic],
        start_time=start_time,
        end_time=end_time):

    # メッセージ型に応じて調整
    values.append(msg.motor_value[0])

bag.close()

if values:
    avg = sum(values) / len(values)
    print("平均:", avg)
else:
    print("指定範囲にデータがありません")
