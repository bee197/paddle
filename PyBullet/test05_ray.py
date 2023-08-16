import pybullet as p
import pybullet_data
import time
from pprint import pprint
import math

# ---------------------------------------------------------#
#   连接物理引擎
# ---------------------------------------------------------#

use_gui = True
if use_gui:
    serve_id = p.connect(p.GUI)
else:
    serve_id = p.connect(p.DIRECT)

# ---------------------------------------------------------#
#   添加资源路径
# ---------------------------------------------------------#

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# ---------------------------------------------------------#
#   渲染设置
# ---------------------------------------------------------#

# 创建过程中先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# 不展示GUI的套件
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# 禁用 tinyrenderer
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

# ---------------------------------------------------------#
#   环境操控
# ---------------------------------------------------------#

# 设置重力，加载模型
p.setGravity(0, 0, -10)
# 加载URDF模型，此处是加载蓝白相间的陆地,没有机器人会掉下去
_ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)

# ---------------------------------------------------------#
#   加载机器人
# ---------------------------------------------------------#

# 设置加载的机器人的位置
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
# 加载机器人
robot_id = p.loadURDF("r2d2.urdf", useMaximalCoordinates=True)

# ---------------------------------------------------------#
#   加入碰撞物
# ---------------------------------------------------------#

# 创建视觉模型和碰撞箱模型时共用的两个参数
shift = [0, -0.02, 0]
scale = [1, 1, 1]

# 创建一面墙
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[60, 5, 5]
)

collison_box_id = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[60, 5, 5]
)

wall_id = p.createMultiBody(
    baseMass=10000,
    baseCollisionShapeIndex=collison_box_id,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0, 10, 5],
    useMaximalCoordinates=True
)

# ---------------------------------------------------------#
#   关节操控
# ---------------------------------------------------------#

available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if
                            p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
pprint([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])
# 获取轮子的关节索引
wheel_joints_indexes = [i for i in available_joints_indexes if "wheel" in str(p.getJointInfo(robot_id, i)[1])]
pprint([p.getJointInfo(robot_id, i)[1] for i in wheel_joints_indexes])

# 预备工作结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# 禁用自然时间流逝
p.setRealTimeSimulation(0)

# ---------------------------------------------------------#
#   运动参数
# ---------------------------------------------------------#

target_v = 20  # 电机达到的预定角速度（rad/s）
max_force = 100  # 电机能够提供的力，这个值决定了机器人运动时的加速度，太快会翻车哟，单位N

# ---------------------------------------------------------#
#   激光参数
# ---------------------------------------------------------#

# 如果不需要将激光可视化出来，置为False
useDebugLine = True
hitRayColor = [0, 1, 0]
missRayColor = [1, 0, 0]

rayLength = 15          # 激光长度
rayNum = 1              # 激光数量

# ---------------------------------------------------------#
#   开始录制
# ---------------------------------------------------------#

# 录制到../log/robotmove.mp4
# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "../log/robotmove.mp4")

# 开始一千次迭代，也就是一千次交，每次交互后停顿1/240
for i in range(1000):
    p.stepSimulation()

    # ---------------------------------------------------------#
    #   机器人运动
    # ---------------------------------------------------------#

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,  # 机器人编号
        jointIndices=wheel_joints_indexes,  # 运动的关节数
        controlMode=p.VELOCITY_CONTROL,  # 控制方式为速度控制
        targetVelocities=[target_v for _ in wheel_joints_indexes],  # 每一个关节的速度
        forces=[max_force for _ in wheel_joints_indexes]  # 每一个关节的电机能够提供的力
    )

    # ---------------------------------------------------------#
    #   在机器人位置设置相机
    # ---------------------------------------------------------#

    # 机器人获取位置
    location, _ = p.getBasePositionAndOrientation(robot_id)
    # 在机器人位置设置相机
    p.resetDebugVisualizerCamera(
        cameraDistance=3,  # 与相机距离
        cameraYaw=30,  # 左右视角
        cameraPitch=-30,  # 上下视角
        cameraTargetPosition=location  # 位置
    )

    # ---------------------------------------------------------#
    #   碰撞检测
    # ---------------------------------------------------------#

    # 获取需要发射得rayNum激光的起点与终点
    begins, _ = p.getBasePositionAndOrientation(robot_id)
    rayFroms = [begins for _ in range(rayNum)]
    rayTos = [
        [
            begins[0] + rayLength * math.cos(2 * math.pi * float(i) / rayNum),
            begins[1] + rayLength * math.sin(2 * math.pi * float(i) / rayNum),
            begins[2]
        ]
        for i in range(rayNum)]

    # 调用激光探测函数
    results = p.rayTestBatch(rayFroms, rayTos)

    # 染色前清楚标记
    p.removeAllUserDebugItems()

    # 根据results结果给激光染色
    for index, result in enumerate(results):
        if result[0] == -1:
            p.addUserDebugLine(rayFroms[index], rayTos[index], missRayColor)
        else:
            p.addUserDebugLine(rayFroms[index], rayTos[index], hitRayColor)

    time.sleep(1 / 240)
#     结束录制
# p.stopStateLogging(log_id)