import paddle
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import torch
from torch import nn
from x2paddle.convert import pytorch2paddle


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 这个网络是原版Atari的网络架构
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 3)
        self.fc3 = nn.Linear(512, 1)


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)

# Example: model = PPO("MlpPolicy", "Pendulum-v1")
model = PPO.load("../ppo/train_log/ppo_robot_1080000_steps")
model.policy.to("cpu")
model.policy.eval()
onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)
# model.policy.observation_space.shape[0]
dummy_input = torch.randn(3136, 512)
torch.onnx.export(onnxable_model, dummy_input, "my_ppo_model.onnx", opset_version=11)
