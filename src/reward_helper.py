from abc import ABCMeta, abstractmethod


class IRewardMCOne(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None):
        pass


class RewardMcOne(IRewardMCOne):
    def __init__(self) -> None:
        self.ut = 9999

    def get_reward(self, util=None, desired_util=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        self.ut = desired_util
        if util <= desired_util:
            reward = 1.0
        else:
            reward = -0.02

        return reward


class RewardMcTwo(IRewardMCOne):
    def __init__(self) -> None:
        self.ut = 9999

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ut <= self.ut:
            reward = 1.0
            self.ut = ut
        else:
            reward = -0.02

        return reward


class RewardMcThree(IRewardMCOne):
    def __init__(self) -> None:
        pass

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ((packet_loss < packet_thresh) and
                (latency < latency_thresh) and
                ((energy_thresh - 0.1) < energy_consumption) and
                (energy_consumption < energy_thresh)):
            reward = 1.0
        else:
            reward = -0.02

        return reward


class RewardMcFour(IRewardMCOne):
    def __init__(self) -> None:
        pass

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ((packet_loss < packet_thresh) and
                (latency < latency_thresh)):
            reward = 1.0
        else:
            reward = -0.02

        return reward


class RewardMcFive(IRewardMCOne):
    def __init__(self) -> None:
        self.energy_consumption = 9999

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ((packet_loss < packet_thresh) and
                (latency < latency_thresh) and
                (energy_consumption < self.energy_consumption)):
            self.energy_consumption = energy_consumption
            reward = 1.0
        else:
            reward = -0.02

        return reward




