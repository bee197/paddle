import collections
import math
from random import randint

import cv2
import gym
import numpy as np
import paddle
import pybullet_data
from gym import spaces
import pybullet as p

STACK_SIZE = 3
STEP_MAX = 300
BASE_RADIUS = 0.5
BASE_THICKNESS = 0.2
TIME_STEP = 0.02
BASE_POS = [0., 0., 0.2]
LOCATION = [0., 0., 0.2]
NOISE = 0.5


class RobotEnv(gym.Env):

    # ---------------------------------------------------------#
    #   __init__
    # ---------------------------------------------------------#

    def __init__(self, render: bool = False):
        self.hit_num = None
        self.angle_prev2 = None
        self.distance_prev2 = None
        self.angle_prev = None
        self.distance_prev = None
        self.h_fov = None
        self.coord2 = None
        self.coord1 = None
        self.soccer = None
        self.collision = None
        self.plane = None
        self.robot = None
        self._render = render
        self.stacked_frames = collections.deque([np.zeros((84, 84), dtype=np.float32) for i in range(STACK_SIZE)],
                                                maxlen=3)

        # 定义动作空间
        self.action_space = spaces.Discrete(3)

        # 定义状态空间的最小值和最大值(二值化后只有0与1)
        min_value = 0
        max_value = 1

        # 定义状态空间
        self.observation_space = spaces.Box(
            low=min_value,
            high=max_value,
            shape=(4, 84, 84),
            dtype=np.float32
        )

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)

        # 计数器
        self.step_num = 0

    # ---------------------------------------------------------#
    #   reset
    # ---------------------------------------------------------#

    def reset(self):
        # 在机器人位置设置相机
        p.resetDebugVisualizerCamera(
            cameraDistance=8,  # 与相机距离
            cameraYaw=-90,  # 左右视角
            cameraPitch=-89,  # 上下视角
            cameraTargetPosition=LOCATION  # 位置
        )
        # 初始化计数器
        self.step_num = 0
        self.hit_num = 0
        # 设置一步的时间,默认是1/240
        p.setTimeStep(TIME_STEP)
        # 初始化模拟器
        p.resetSimulation(physicsClientId=self._physics_client_id)
        # 初始化重力
        p.setGravity(0, 0, -9.8)
        # 初始化随机坐标
        seed1, seed2 = self.seed()
        # 初始化障碍物1 TODO:
        self.soccer = self.__create_coll(seed1, obj="soccerball.obj", is_current_folder=False, scale=[0.7, 0.7, 0.7])
        # 初始化障碍物2
        self.collision = self.__create_coll(seed2, obj="Bottle.obj", is_current_folder=True,
                                            scale=[0.005, 0.005, 0.005])
        # 初始化机器人
        self.robot = p.loadURDF("./miniBox.urdf", basePosition=BASE_POS, physicsClientId=self._physics_client_id)
        # 设置文件路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 初始化地板
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        p.changeVisualShape(self.plane, -1, rgbaColor=[0, 1, 0, 1])
        # 初始化白线
        # whiteColor = [1, 1, 1, 1]
        # lineWidth = 0.2
        # lineLength = 100
        # visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
        #                                     halfExtents=[lineLength / 2, lineWidth / 2, 0.01],
        #                                     rgbaColor=whiteColor)
        # lineId = p.createMultiBody(baseVisualShapeIndex=visualShapeId)
        # p.resetBasePositionAndOrientation(lineId, [0, 0, 0.01], [0, 0, 0, 1])

        # 获得视野
        fov = 50.0
        aspect = 1.0
        fov_rad = math.radians(fov)
        self.h_fov = math.atan(math.tan(fov_rad / 2) * aspect) * 2
        # 上一步的距离
        self.distance_prev, distance_ori, distance_coll, robot_pos, coll_pos, robot_ori, coll_ori = self.__get_distance(self.soccer)
        self.distance_prev2, distance_ori, distance_coll, robot_pos, coll_pos2, robot_ori2, coll_ori = self.__get_distance(
            self.soccer)
        robot_angle = math.atan2(robot_ori[0], robot_ori[1])
        coll_angle = math.atan2(coll_pos[0], coll_pos[1])
        robot_angle2 = math.atan2(robot_ori2[0], robot_ori2[1])
        coll_angle2 = math.atan2(coll_pos2[0], coll_pos2[1])

        self.angle_prev = np.abs(coll_angle - robot_angle)
        self.angle_prev2 = np.abs(coll_angle2 - robot_angle2)

        # 获得图片
        obs = self.__get_observation()

        distance = paddle.to_tensor([self.distance_prev, self.distance_prev2])
        angle = paddle.to_tensor([self.angle_prev, self.angle_prev2])
        obs = paddle.to_tensor(obs, dtype='float32')

        obs = [(obs, distance, angle)]

        return obs

    # ---------------------------------------------------------#
    #   step
    # ---------------------------------------------------------#

    def step(self, action):
        self.step_num += 1
        # 调整速度
        self.__apply_action(action)
        # 步进
        p.stepSimulation(physicsClientId=self._physics_client_id)

        # 获得距离 TODO:
        distance, distance_ori, distance_coll, robot_pos, coll_pos, robot_ori, coll_ori = self.__get_distance(
            self.soccer)
        distance2, distance_ori2, distance_coll2, robot_pos2, coll_pos2, robot_ori2, coll_ori2 = self.__get_distance(
            self.collision)

        # 获得angle TODO:
        # 障碍物与机器人向量
        coll_robot = np.array(coll_pos) - np.array(robot_pos)
        robot_angle = math.atan2(robot_ori[0], robot_ori[1])
        coll_angle = math.atan2(coll_robot[0], coll_robot[1])

        coll_robot2 = np.array(coll_pos2) - np.array(robot_pos2)
        robot_angle2 = math.atan2(robot_ori2[0], robot_ori2[1])
        coll_angle2 = math.atan2(coll_robot2[0], coll_robot2[1])

        # TODO:
        angle = np.abs(coll_angle - robot_angle)
        angle2 = np.abs(coll_angle2 - robot_angle2)

        # 设置奖励:足球 TODO:
        # TODO:改进人工势场算法 Uatt = 0.5 * ξ * ||qgoal - q||^2
        video_in = angle < self.h_fov / 2
        video_in2 = angle2 < self.h_fov / 2

        # 沿着白线有奖励
        reward_line = 0
        # if -0.2 < robot_pos[1] < 0.2:
        #     reward_line += 0.5

        # 沿着足球有奖励
        reward_goal = 0
        if video_in:
            reward_goal += 1 / (distance + 1)
        # if video_in and distance <= 8:
        #     reward_goal += 1
        # if angle2 < 0.5:
        #     close_angle = self.angle_prev - angle
        #     close_dis = self.distance_prev - distance
        #     reward_goal += 10 * close_angle + 5 * close_dis

        # 障碍物
        reward_coll = 0
        # if video_in2:
        #     reward_coll -= 0.25
        # if angle2 < 0.5:
        #     close_angle2 = self.angle_prev2 - angle2
        #     close_dis2 = self.distance_prev2 - distance2
        #     reward_coll -= 10 * close_angle2 + 5 * close_dis2

        # 奖励
        reward = reward_line + reward_goal + reward_coll
        # print(f"reward:{reward}")
        info = False
        # 只有距离小于2时才判断是否相撞,减少运算
        # 距离大于2,肯定没相撞 TODO:
        if distance > 2:
            done = False
        # 距离小于2,且相撞
        elif self.__is_collision(self.soccer):
            print(f"!!!足球!!!")
            reward += 10
            info = True
            done = True
        # 距离小于2,又没有相撞
        else:
            done = False

        # 与障碍物相撞
        if self.__is_collision(self.collision):
            reward -= 10
            self.hit_num += 1
            if self.hit_num >= 20:
                done = True
                print(f"!")
        # 步数超过限制
        if self.step_num > STEP_MAX:
            print("-")
            done = True

        # TODO:上一次的距离角度
        self.distance_prev = distance
        self.angle_prev = angle
        self.distance_prev2 = distance2
        self.angle_prev2 = angle2

        # 获得图片
        obs = self.__get_observation()

        distance = paddle.to_tensor([self.distance_prev, self.distance_prev2])
        angle = paddle.to_tensor([self.angle_prev, self.angle_prev2])
        obs = paddle.to_tensor(obs, dtype='float32')

        obs = [(obs, distance, angle)]

        return obs, reward, done, info

    def seed(self):
        # 随机足球的坐标
        x1 = np.random.uniform(6, 7)
        # x1 = 6
        # y1 = np.random.uniform(-2, 2)
        y1 = np.random.choice([-2, 2], size=1)
        # x2 = np.random.uniform(3, 4)
        x2 = 4
        # y2 = np.random.uniform(-1, 1)
        y2 = 0
        # y2 = np.clip(np.random.normal(y1, NOISE), y1 - 0.5, y1 + 0.5)

        self.coord1 = [x1, y1, 0.4]
        self.coord2 = [x2, y2, 0.]

        return self.coord1, self.coord2

    def render(self, mode='human'):
        pass

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

    # ---------------------------------------------------------#
    #   设置速度
    # ---------------------------------------------------------#

    def __apply_action(self, action):
        left_v = 10.
        right_v = 10.
        # print("action : ", action)
        if action == 0:
            left_v = 5.
        elif action == 2:
            right_v = 5.

        # 设置速度
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[3, 2],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[10., 10.],
            physicsClientId=self._physics_client_id
        )

        # ---------------------------------------------------------#
        #   获得状态空间
        # ---------------------------------------------------------#

    def __get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        # 传入图片
        w, h, rgbPixels, depthPixels, segPixels = self.__setCameraPicAndGetPic(self.robot)
        # 灰度化,归一化
        obs = self.__norm(rgbPixels)

        stacked_frames = collections.deque([np.zeros((84, 84), dtype=np.float32) for i in range(STACK_SIZE)],
                                           maxlen=3)
        stacked_frames.append(obs)
        stacked_frames.append(depthPixels)
        stacked_frames.append(segPixels)

        obs = np.stack(stacked_frames, axis=0)
        return obs

    # ---------------------------------------------------------#
    #   堆栈4张归一化的图片
    # ---------------------------------------------------------#

    def __stack_pic(self, pic, stacked_frames, is_done):
        # 如果新的开始
        if is_done:
            # Clear our stacked_frames
            stacked_frames = collections.deque([np.zeros((84, 84), dtype=np.float32) for i in range(STACK_SIZE)],
                                               maxlen=3)

            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(pic)
            stacked_frames.append(pic)
            stacked_frames.append(pic)

            # Stack the frames
            stacked_pic = np.stack(stacked_frames, axis=0)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(pic)

            # Build the stacked state (first dimension specifies different frames)
            stacked_pic = np.stack(stacked_frames, axis=0)

        return stacked_pic, stacked_frames

    # ---------------------------------------------------------#
    #   获得距离
    # ---------------------------------------------------------#

    def __get_distance(self, id, bias=[0, 0]):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        # 获得机器人位置
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        # 碰撞物位置
        collPos, collOri = p.getBasePositionAndOrientation(id, physicsClientId=self._physics_client_id)

        # 添加y轴偏移,tuple不能改变元素,换list
        collList = list(collPos)
        collList[0] += bias[0]
        collList[1] += bias[1]
        collPos = tuple(collList)

        dis = np.array(collPos) - np.array(basePos)
        dis_Ori = np.array(basePos) - np.array(BASE_POS)
        dis_coll = np.array(collPos) - np.array(BASE_POS)
        # 机器人与障碍物的距离
        distance = np.linalg.norm(dis)
        # 机器人与原点的距离
        distance_Ori = np.linalg.norm(dis_Ori)
        # 障碍物与原点的距离
        distance_coll = np.linalg.norm(dis_coll)
        # 再获取夹角
        matrix = p.getMatrixFromQuaternion(baseOri, physicsClientId=self._physics_client_id)
        baseOri = np.array([matrix[0], matrix[3], matrix[6]])

        # print("distance : ", distance)

        return distance, distance_Ori, distance_coll, basePos, collPos, baseOri, collOri

    # ---------------------------------------------------------#
    #   是否碰撞
    # ---------------------------------------------------------#

    def __is_collision(self, id):
        P_min, P_max = p.getAABB(self.robot)
        id_tuple = p.getOverlappingObjects(P_min, P_max)
        if len(id_tuple) > 1:
            for ID, _ in id_tuple:
                if ID == id:
                    return True
                else:
                    continue

        return False

    # ---------------------------------------------------------#
    #   合成相机
    # ---------------------------------------------------------#

    def __setCameraPicAndGetPic(self, robot_id: int, width: int = 84, height: int = 84, physicsClientId: int = 0):
        """
            给合成摄像头设置图像并返回robot_id对应的图像
            摄像头的位置为miniBox前头的位置
        """
        basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
        # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴

        basePos = np.array(basePos)
        # 摄像头的位置
        # BASE_RADIUS 为 0.5，是机器人底盘的半径。BASE_THICKNESS 为 0.2 是机器人底盘的厚度。
        # 别问我为啥不写成全局参数，因为我忘了我当时为什么这么写的。
        cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
        targetPos = cameraPos + 1 * tx_vec

        # 相机的空间位置
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec,
            physicsClientId=physicsClientId
        )

        # 相机镜头的光学属性，比如相机能看多远，能看多近
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=50.0,  # 摄像头的视线夹角
            aspect=1.0,
            nearVal=0.01,  # 摄像头焦距下限
            farVal=20,  # 摄像头能看上限
            physicsClientId=physicsClientId
        )

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=width, height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            physicsClientId=physicsClientId
        )

        return width, height, rgbImg, depthImg, segImg

    # ---------------------------------------------------------#
    #   创造足球
    # ---------------------------------------------------------#

    def __create_coll(self, coord, obj, is_current_folder, scale):
        # 创建视觉模型和碰撞箱模型时共用的两个参数
        shift = [0, 0, 0]
        if not is_current_folder:
            # 添加资源路径
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 创建视觉形状
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=shift,
            meshScale=scale
        )

        # 创建碰撞箱模型
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=obj,
            collisionFramePosition=shift,
            meshScale=scale
        )

        # 使用创建的视觉形状和碰撞箱形状使用createMultiBody将两者结合在一起
        p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=coord,
            baseOrientation=p.getQuaternionFromEuler([3.14/2, 0, 0]),
            useMaximalCoordinates=True
        )

        # 碰撞id
        return collision_shape_id

    # ---------------------------------------------------------#
    #   归一化
    # ---------------------------------------------------------#

    def __norm(self, image):
        # cv灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图
        # 归一化
        complete = np.zeros(gray.shape, dtype=np.float32)
        result = cv2.normalize(gray, complete, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return result

# # test
# env = RobotEnv(render=True)
# obs = env.reset()
# p.setRealTimeSimulation(1)
# while True:
#     action = np.random.uniform(-10, 10, size=(2,))
#     obs, reward, done, _ = env.step(action)
#     if done:
#         break
#
#     print(f"obs : {obs} reward : {reward}")
