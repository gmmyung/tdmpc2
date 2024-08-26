import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QProgressBar, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
import torch
from raisimGymTorch.env.envs.rsg_anymal.tdmpc2.envs import make_env
from tdmpc2 import TDMPC2
from common.parser import parse_cfg
import hydra
import os

class SimpleApp(QWidget):
    def __init__(self, reset, step, setCommand):
        super().__init__()

        # Set up the window
        self.setWindowTitle('Raibo Controller')
        self.setGeometry(100, 100, 300, 200)

        # Create a vertical layout
        layout = QVBoxLayout()

        self.resetButton = QPushButton('Reset', self)
        layout.addWidget(self.resetButton)
        self.resetButton.clicked.connect(reset)

        self.reward = QHBoxLayout()
        self.rewardLabel = QLabel('Reward:', self)
        self.reward.addWidget(self.rewardLabel)
        self.rewardBar = QProgressBar(self)
        self.reward.addWidget(self.rewardBar)
        layout.addLayout(self.reward)

        self.reward_info = QHBoxLayout()
        self.command_tracking_reward_coeff = QLabel('command_tracking_reward_coeff:', self)
        self.body_contact_reward_coeff = QLabel('body_contact_reward_coeff:', self)
        self.torque_reward_coeff = QLabel('torque_reward_coeff:', self)
        self.pitch_reward_coeff = QLabel('pitch_reward_coeff:', self)
        self.reward_info.addWidget(self.command_tracking_reward_coeff)
        self.reward_info.addWidget(self.body_contact_reward_coeff)
        self.reward_info.addWidget(self.torque_reward_coeff)
        self.reward_info.addWidget(self.pitch_reward_coeff)
        layout.addLayout(self.reward_info)

        self.FPS = QHBoxLayout()
        self.FPSLabel = QLabel('FPS:', self)
        self.FPS.addWidget(self.FPSLabel)
        self.FPSBar = QProgressBar(self)
        self.FPS.addWidget(self.FPSBar)
        layout.addLayout(self.FPS)

        # Timer steps every 0.05 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(step)
        self.timer.start(50)

        self.command = [0.0, 0.0, 0.0]        
        self.setCommand = setCommand

        self.setLayout(layout)
        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.command[0] =0.8
        elif event.key() == Qt.Key_S:
            self.command[0] =0.0
        elif event.key() == Qt.Key_A:
            self.command[1] = -0.2
        elif event.key() == Qt.Key_D:
            self.command[1] = 0.2
        elif event.key() == Qt.Key_Q:
            self.command[2] = -0.5
        elif event.key() == Qt.Key_E:
            self.command[2] = 0.5
        else:
            super().keyPressEvent(event)
            return
        self.setCommand(self.command)
        print(self.command)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_W:
            self.command[0] = 0.0
        elif event.key() == Qt.Key_A:
            self.command[1] = 0.0
        elif event.key() == Qt.Key_D:
            self.command[1] = 0.0
        elif event.key() == Qt.Key_Q:
            self.command[2] = 0.0
        elif event.key() == Qt.Key_E:
            self.command[2] = 0.0
        else:
            super().keyReleaseEvent(event)
            return
        self.setCommand(self.command)
        print(self.command)

    def updateReward(self, reward):
        self.rewardBar.setValue(int(abs(reward[0]) * 100))
        self.rewardLabel.setText(f'Reward: {reward[0]:.2f}')

    def updateFPS(self, fps):
        self.FPSBar.setValue(int(fps * 100))
        self.FPSLabel.setText(f'FPS: {fps:.2f}')

class Controller:
    def __init__(self, agent, env):
        self.app = QApplication(sys.argv)
        self.agent = agent
        self.env = env
        self.window = SimpleApp(self.reset, self.step, self.setCommand)
        self.command = [0.0, 0.0, 0.0]
        self.obs = None
        self.action = None
        self.t = 0

    def reset(self):
        self.obs = None

    def step(self):
        if self.obs is None:
            self.obs = self.env.reset()
        start = time.time()
        self.obs['state'][0, -3:] = torch.tensor(self.command)
        # self.obs[0, -3:] = torch.tensor(self.command)
        action = self.agent.act(self.obs, t0 = self.t == 0, eval_mode=True)
        self.obs, self.reward, _, info = self.env.step(action)
        elapsed = time.time() - start

        self.window.updateReward(self.reward)
        self.window.updateFPS(1/elapsed)
        self.t += 1
        for key in info.keys():
            print(key, info[key])


    def setCommand(self, command):
        self.command = command

    def run(self):
        self.window.show()
        self.app.exec_()

@hydra.main(config_name="config", config_path=".")
def run(cfg: dict):
    cfg["num_envs"] = 1
    cfg["raisim_config"]["num_envs"] = 1
    cfg["raisim_config"]["num_threads"] = 1
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)
    controller = Controller(agent, env)
    controller.run()

if __name__ == "__main__":
    run()
