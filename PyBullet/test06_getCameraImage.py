import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# ---------------------------------------------------------#
#   连接物理引擎
# ---------------------------------------------------------#

use_gui = True
if use_gui:
    cid = p.connect(p.GUI)
else:
    cid = p.connect(p.DIRECT)

# ---------------------------------------------------------#
#   添加资源(必须先添加模型,在tinyrenderer之前,不然相机里看不见,或者tinyrenderer设为1)
# ---------------------------------------------------------#

p.setAdditionalSearchPath(pybullet_data.getDataPath())
# 加载机器人
robot_id = p.loadURDF("./miniBox.urdf", basePosition=[0, 0, 0.5])
# 加载URDF模型，此处是加载蓝白相间的陆地,没有机器人会掉下去
_ = p.loadURDF("plane.urdf")

# ---------------------------------------------------------#
#   渲染设置
# ---------------------------------------------------------#

# 创建过程中先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# 不展示GUI的套件
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# 禁用 tinyrenderer
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

# ---------------------------------------------------------#
#   环境操控
# ---------------------------------------------------------#

# 设置重力，加载模型
p.setGravity(0, 0, -10)

# ---------------------------------------------------------#
#   绘制直线
# ---------------------------------------------------------#

# 绘制直线
froms = [[1, 1, 0], [-1, 1, 0], [-1, 1, 3], [1, 1, 3]]
tos = [[-1, 1, 0], [-1, 1, 3], [1, 1, 3], [1, 1, 0]]
for f, t in zip(froms, tos):
    p.addUserDebugLine(
        lineFromXYZ=f,
        lineToXYZ=t,
        lineColorRGB=[0, 1, 0],
        lineWidth=2
    )

# ---------------------------------------------------------#
#   添加文字
# ---------------------------------------------------------#

p.addUserDebugText(
    text="Destination",
    textPosition=[0, 1, 3],
    textColorRGB=[0, 1, 0],
    textSize=1.2,
)

p.addUserDebugText(
    text="I'm R2D2",
    textPosition=[0, 0, 1.2],
    textColorRGB=[0, 0, 1],
    textSize=1.2
)

# ---------------------------------------------------------#
#   添加按钮控件,获得相机图像
# ---------------------------------------------------------#

# 添加按钮控件
btn = p.addUserDebugParameter(
    paramName="getCameraImage",
    rangeMin=1,
    rangeMax=0,
    startValue=0
)

previous_btn_value = p.readUserDebugParameter(btn)

# ---------------------------------------------------------#
#   其他
# ---------------------------------------------------------#

# 预备工作结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# 禁用自然时间流逝
p.setRealTimeSimulation(1)

# ---------------------------------------------------------#
#   开始迭代
# ---------------------------------------------------------#

# 开始一千次迭代，也就是一千次交，每次交互后停顿1/240
while True:
    p.stepSimulation()

    # ---------------------------------------------------------#
    #   按钮
    # ---------------------------------------------------------#

    # 如果按钮的值发生变化了，说明clicked了
    if p.readUserDebugParameter(btn) != previous_btn_value:
        # 传入画面
        w, h, rgbPixels, depthPixels, segPixels = p.getCameraImage(800, 600)

        # 转化np格式
        rgbPixels = np.array(rgbPixels)
        depthPixels = np.array(depthPixels)
        segPixels = np.array(segPixels)

        print("rgb", rgbPixels.shape)
        print("depth", depthPixels.shape)
        print("seg", segPixels.shape)

        # plt画出图像
        plt.figure(figsize=[12, 9])
        plt.subplot(2, 2, 1)
        plt.imshow(rgbPixels)
        plt.title("rgbPixels")
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.imshow(depthPixels, cmap=plt.cm.gray)
        plt.title("depthPixels")
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.imshow(segPixels)
        plt.title("segmentationMaskBuffer")
        plt.axis("off")
        plt.show()

        previous_btn_value = p.readUserDebugParameter(btn)
