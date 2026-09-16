#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Registry of the research environments, keyed by `cfg.env.name`."""
from itps.envs.base_env import Environment


def get_env(name: str) -> Environment:
    """Return the Environment registered under `name`, or raise if the name is not one of ours."""
    if name == "gmm":
        from itps.envs.gmm.gmm_env import GMMEnv

        return GMMEnv()
    elif name == "maze2d":
        from itps.envs.maze.maze_env import MazeEnv

        return MazeEnv()
    elif name == "robosuite":
        from itps.envs.robosuite.robosuite_env import RobosuiteEnv

        return RobosuiteEnv()
    raise NotImplementedError(f"No environment registered under {name!r}.")


def is_registered(name: str) -> bool:
    """Whether `name` is one of our environments (as opposed to a gym env LeRobot rolls out)."""
    return name in ("gmm", "maze2d", "robosuite")
