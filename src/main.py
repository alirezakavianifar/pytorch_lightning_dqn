from deep_q_learning import DeepQLearning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from envirnment import DeltaIotEnv
from reward_helper import RewardMcThree

if __name__ == "__main__":
    env = DeltaIotEnv(n_actions=216, 
                      n_obs_space=3, 
                      reward_type=RewardMcThree,
                      data_dir=r'E:\projects\RL-DeltaIoT\data\DeltaIoTv1\train',
                      energy_thresh=12.90)
    algo = DeepQLearning(env=env)

    trainer = Trainer(max_epochs=10_000,
                      callbacks=EarlyStopping(monitor='episode/Return', mode='max', patience=500))
    trainer.fit(algo)
    
