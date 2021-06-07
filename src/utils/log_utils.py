from timeit import default_timer

import numpy as np

from flatland.envs.rail_env import RailEnvActions
from torch.utils.tensorboard import SummaryWriter

class Timer(object):
    """
    Utility to measure times.
    """

    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        self.start_time = default_timer()

    def end(self):
        self.total_time += default_timer() - self.start_time

    def get(self):
        return self.total_time

    def get_current(self):
        return default_timer() - self.start_time

    def reset(self):
        self.__init__()

    def __repr__(self):
        return self.get()


class TBLogger:
    """
    Class to handle Tensorboard logging.
    """

    def __init__(self, tensorboard_path, env_params=None, train_params=None):
        """
        :param tensorboard_path: the path where logs are saved
        :param env_params: environment parameters
        :param train_params: training parameters
        """
        self.writer = SummaryWriter(tensorboard_path)

        if train_params is not None:
            self.writer.add_hparams(train_params, {})
        if env_params is not None:
            self.writer.add_hparams(env_params, {})

    def write(self, env, train_params, timers, step):
        """
        Save logs to Tensorboard
        :param env: the environment used to extract some statistics
        :param train_params: a dictionary of training statistics to record
        :param timers: a dictionary of timers to record
        """
        # Environment parameters
        self.writer.add_scalar("metrics/score", env.normalized_score, step)
        self.writer.add_scalar("metrics/accumulated_score", np.mean(env.normalized_score_history), step)
        self.writer.add_scalar("metrics/completion", env.completion, step)
        self.writer.add_scalar("metrics/accumulated_completion", np.mean(env.completion_history), step)
        self.writer.add_histogram("actions/distribution", np.array(env.action_probs), step)
        self.writer.add_scalar("actions/nothing", env.action_probs[RailEnvActions.DO_NOTHING], step)
        self.writer.add_scalar("actions/left", env.action_probs[RailEnvActions.MOVE_LEFT], step)
        self.writer.add_scalar("actions/forward", env.action_probs[RailEnvActions.MOVE_FORWARD], step)
        self.writer.add_scalar("actions/right", env.action_probs[RailEnvActions.MOVE_RIGHT], step)
        self.writer.add_scalar("actions/stop", env.action_probs[RailEnvActions.STOP_MOVING], step)

        # Training parameters
        for param_name, param in train_params.__dict__.items():
            assert type(param_name) is str, "Parameters names must be strings!"
            self.writer.add_scalar("training/" + param_name, param, step)

        # Timers
        for timer_name, timer in timers.items():
            assert type(timer_name) is str and type(timer) is Timer, "A Timer object and its name (string) must be passed!"
            self.writer.add_scalar("timer/" + timer_name, timer.get(), step)