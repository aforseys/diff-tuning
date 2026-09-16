# MIT License

# Copyright (c) 2024 Yanwei Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Some of this software is derived from LeRobot, which is subject to the following copyright notice:

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# Tony Z. Zhao
# and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys, os
import numpy as np
import pygame
import torch
import argparse
import matplotlib.pyplot as plt
import einops
from pathlib import Path
from huggingface_hub import snapshot_download
from common.utils.utils import seeded_context, init_hydra_config
from common.utils.maze_maps import MAZE_MAPS
from common.utils.maze_scoring import check_maze_collision
from common.policies.factory import make_policy
from common.datasets.factory import make_dataset
from scipy.special import softmax
import time
import json

class MazeEnv:
    def __init__(self, maze_type):
        # GUI x coord 0 -> gui_size[0] #1200
        # GUI y coord 0 
        #         |
        #         v
        #       gui_size[1] #900
        # xy is in the same coordinate system as the background
        # bkg y coord 0 -> maze_shape[1] #12
        # bkg x coord 0
        #         |
        #         v
        #       maze_shape[0] #9
        if maze_type not in MAZE_MAPS:
            raise NotImplementedError("Maze type does not exist!")
        self.maze = MAZE_MAPS[maze_type]

        self.gui_size = (1200, 900)
        self.fps = 10
        self.batch_size = 32        
        self.offset = 0.5 # Offset to put object in the center of the cell

        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        self.agent_color = self.RED
        self.goal_color = self.GREEN

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.gui_size)
        pygame.display.set_caption("Maze")
        self.clock = pygame.time.Clock()                                              
        self.agent_gui_pos = np.array([0, 0]) # Initialize the position of the red dot
        self.running = True

        self._sample_times=[]

    def check_collision(self, xy_traj):
        assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
        # Single shared definition (maze_scoring.check_maze_collision): NaN steps
        # count as collisions, consistent with eval_maze and preference selection.
        return check_maze_collision(xy_traj, self.maze)

    def blend_with_white(self, color, factor=0.5):
        white = np.array([255, 255, 255])
        blended_color = (1 - factor) * np.array(color) + factor * white
        return blended_color.astype(int)

    def report_collision_percentage(self, collisions):
        num_trajectories = collisions.shape[0]
        num_collisions = np.sum(collisions)
        collision_percentage = (num_collisions / num_trajectories) * 100
        print(f"{num_collisions}/{num_trajectories} trajectories are in collision ({collision_percentage:.2f}%).")
        return collision_percentage

    def xy2gui(self, xy):
        xy = xy + self.offset # Adjust normalization as necessary
        x = xy[0] * self.gui_size[1] / (self.maze.shape[0])
        y = xy[1] * self.gui_size[0] / (self.maze.shape[1])
        return np.array([y, x], dtype=float)

    def gui2xy(self, gui):
        x = gui[1] / self.gui_size[1] * self.maze.shape[0] - self.offset
        y = gui[0] / self.gui_size[0] * self.maze.shape[1] - self.offset
        return np.array([x, y], dtype=float)

    def generate_time_color_map(self, num_steps):
        cmap = plt.get_cmap('rainbow')
        values = np.linspace(0, 1, num_steps)
        colors = cmap(values)
        return colors

    def draw_maze_background(self):
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(self.maze[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, self.gui_size)
        self.screen.blit(surface, (0, 0))

    def update_screen(self, xy_pred=None, collisions=None, scores=None, keep_drawing=False, traj_in_gui_space=False, goal=None):
        self.draw_maze_background()

        if goal is not None:
            pygame.draw.circle(self.screen, self.goal_color, 
                             (int(goal[0]), int(goal[1])), 15)
        if xy_pred is not None:
            time_colors = self.generate_time_color_map(xy_pred.shape[1])
            if collisions is None:
                collisions = self.check_collision(xy_pred)
            # self.report_collision_percentage(collisions)
            for idx, pred in enumerate(xy_pred):
                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)

                    # visualize constraint violations (collisions) by tinting trajectories white
                    whiteness_factor = 0.8 if collisions[idx] else 0.0
                    color = self.blend_with_white(color, whiteness_factor)
                    if scores is None:
                        circle_size = 5 if collisions[idx] else 5
                    else: # when similarity scores are provided, visualizing them by changing the trajectory size
                        circle_size = int(3 + 20 * scores[idx])
                    if traj_in_gui_space:
                        start_pos = pred[step_idx]
                        end_pos = pred[step_idx + 1]
                    else:
                        start_pos = self.xy2gui(pred[step_idx])
                        end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        if keep_drawing: # visualize the human drawing input
            if len(self.draw_traj) == 1:
                pt = self.draw_traj[0]
                pygame.draw.circle(self.screen, self.GRAY, (int(pt[0]), int(pt[1])), 12)
            else:
                for i in range(len(self.draw_traj) - 1):
                    pygame.draw.line(self.screen, self.GRAY, self.draw_traj[i], self.draw_traj[i + 1], 10)
  
        pygame.display.flip()

    def similarity_score(self, samples, guide=None):
        # samples: (B, pred_horizon, action_dim)
        # guide: (guide_horizon, action_dim)
        if guide is None:
            return samples, None
        assert samples.shape[2] == 2 and guide.shape[1] == 2
        indices = np.linspace(0, guide.shape[0]-1, samples.shape[1], dtype=int)
        guide = np.expand_dims(guide[indices], axis=0) # (1, pred_horizon, action_dim)
        guide = np.tile(guide, (samples.shape[0], 1, 1)) # (B, pred_horizon, action_dim)
        scores = np.linalg.norm(samples[:, :] - guide[:, :], axis=2, ord=2).mean(axis=1) # (B,)
        scores = 1 - scores / (scores.max() + 1e-6) # normalize
        temperature = 20
        scores = softmax(scores*temperature)
        # normalize the score to be between 0 and 1
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        # sort the predictions based on scores, from smallest to largest, so that larger scores will be drawn on top
        sort_idx = np.argsort(scores)
        samples = samples[sort_idx]
        scores = scores[sort_idx]  
        return samples, scores

class UnconditionalMaze(MazeEnv):

    # for dragging the agent around to explore motion manifold
    def __init__(self, policy, policy_tag=None, vis_energy=False, maze_type="large", obs_list=None, opt_params=None, ddim=False, sample_seed=0):
        super().__init__(maze_type=maze_type)
        self.mouse_pos = None
        self.agent_in_collision = False
        self.agent_history_xy = []
        self.policy = policy
        self.policy_tag = policy_tag
        self.vis_energy = vis_energy
        self.obs_list = obs_list
        self.opt_params = opt_params
        self.sampling_methods = ['ddim'] if ddim else ["ired"]
        # Seeds the initial diffusion noise on every infer_target call -- see there for
        # what that implies. Default 0 is what every dataset collected before this was
        # configurable used, so the default regenerates them exactly.
        self.sample_seed = sample_seed

    def infer_target(self, return_energy=False, goal_pos=None):
        agent_hist_xy = self.agent_history_xy[-1] 
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        if self.policy_tag == 'dp':
            agent_hist_xy = agent_hist_xy.repeat(2, axis=0)

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
        )

        if goal_pos is not None:
            goal = np.array(goal_pos).reshape(1,2)
            obs_batch["episode_goal"] = einops.repeat(
                torch.from_numpy(goal).float().cuda(), "t d -> b t d", b=self.batch_size
            )

        start = time.perf_counter()
        # seeded_context is re-entered on EVERY call, and batch_size is constant, so the
        # initial noise is identical for every observation: one collection run explores
        # the same `batch_size` draws from the prior throughout. That makes a run exactly
        # reproducible; pass --sample-seed to collect a differently-seeded candidate pool
        # (default 0 == every dataset collected before this flag existed).
        with torch.autocast(device_type="cuda"), seeded_context(self.sample_seed):
            if return_energy:
                # Not wired up. `run_inference` no longer returns energies -- scoring now
                # goes through `policy.get_energy(...)`, which makes the timestep, the
                # noise settings and the window explicit. This branch was already dead
                # before that removal: it unpacked `run_inference`'s list return as a bare
                # tensor, so `.detach()` raised AttributeError and `--vis_energy` has never
                # run. Rebuild it as, on the same trajectory that gets drawn:
                #     energy = self.policy.get_energy(
                #         action_batch={'action': actions}, t=0,
                #         observation_batch=obs_batch, deterministic=True)
                raise NotImplementedError(
                    "Energy visualization (--vis_energy) is not set up. Score the drawn "
                    "trajectory with policy.get_energy(...) instead; see the comment above."
                )
            else:
                actions = self.policy.run_inference(obs_batch, opt_params=self.opt_params, methods=self.sampling_methods)[0].cpu().numpy()
        torch.cuda.synchronize()  # important — ensures GPU work is complete before stopping the clock
        elapsed = time.perf_counter() - start

        self._sample_times.append(elapsed)
        print(f"Sample time: {elapsed*1000:.1f}ms | Avg: {np.mean(self._sample_times)*1000:.1f}ms")
        
        return actions
    
    def update_mouse_pos(self):
        self.mouse_pos = np.array(pygame.mouse.get_pos())

    def update_agent_pos(self, new_agent_pos, history_len=1):
        self.agent_gui_pos = np.array(new_agent_pos)
        agent_xy_pos = self.gui2xy(self.agent_gui_pos)
        self.agent_in_collision = self.check_collision(agent_xy_pos.reshape(1, 1, 2))[0]
        if self.agent_in_collision:
            self.agent_color = self.blend_with_white(self.RED, 0.8)
        else:
            self.agent_color = self.RED        
        self.agent_history_xy.append(agent_xy_pos)
        self.agent_history_xy = self.agent_history_xy[-history_len:]
    
    def update_screen_goal_set(self):

        self.draw_maze_background()
        # show where mouse is
        pygame.draw.circle(self.screen, self.BLUE, (int(self.mouse_pos[0]), int(self.mouse_pos[1])), 20)
        pygame.display.flip()

    def run(self):
        trial_idx = 0

        if self.obs_list is not None:
            self.update_agent_pos(self.xy2gui(np.array(self.obs_list[trial_idx])))


        while self.running:
            self.update_mouse_pos()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n and self.obs_list is not None:
                        trial_idx += 1
                        if trial_idx >= len(self.obs_list):
                            print("All observations complete.")
                            self.running = False
                            break
                        print(f"Moving to obs {trial_idx}: {self.obs_list[trial_idx]}")
                        self.update_agent_pos(self.xy2gui(np.array(self.obs_list[trial_idx])))

            if self.obs_list is None:
                self.update_agent_pos(self.mouse_pos.copy())
                
            if self.policy is not None:
                if self.vis_energy: 
                    xy_pred, energy = self.infer_target(return_energy=True)
                    self.update_screen(xy_pred, scores=energy)
                else: 
                    xy_pred = self.infer_target()
                    self.update_screen(xy_pred)
            else:
                self.update_screen()
            self.clock.tick(30)

        pygame.quit()

    def run_gc(self):
        self.goal_set_mode = True
        obs_idx = 0
        goal_pos = None
        goal_gui_pos = None

        while self.running:
            self.update_mouse_pos()

            if self.obs_list is not None:
                if obs_idx >= len(self.obs_list):
                    print("All observations complete.")
                    self.running = False
                    break
                current_obs = self.obs_list[obs_idx]
                start_xy = np.array(current_obs[:2])
                goal_xy = np.array(current_obs[2:]) if len(current_obs) == 4 else None
                self.update_agent_pos(self.xy2gui(start_xy))
                if goal_xy is not None and self.goal_set_mode:
                    goal_gui_pos = self.xy2gui(goal_xy)
                    goal_pos = goal_xy
                    self.goal_set_mode = False

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.goal_set_mode:
                        goal_gui_pos = self.mouse_pos.copy()
                        goal_pos = self.gui2xy(goal_gui_pos)
                        print(f"Goal set at GUI: {goal_gui_pos}, XY: {goal_pos}")
                        self.goal_set_mode = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n and self.obs_list is not None:
                        obs_idx += 1
                        self.goal_set_mode = True
                        goal_pos = None
                        goal_gui_pos = None

            if self.goal_set_mode:
                self.update_screen_goal_set()
            else:
                if self.obs_list is None:
                    self.update_agent_pos(self.mouse_pos.copy())
                if self.vis_energy:
                    xy_pred, energy = self.infer_target(goal_pos=goal_pos, return_energy=True)
                    self.update_screen(xy_pred, scores=energy, goal=goal_gui_pos)
                else:
                    xy_pred = self.infer_target(goal_pos=goal_pos)
                    self.update_screen(xy_pred, goal=goal_gui_pos)
                self.clock.tick(30)

        pygame.quit()

class ConditionalMaze(UnconditionalMaze):
    # for interactive guidance dataset collection
    def __init__(self, policy, savepath=None, policy_tag=None,  maze_type='large', obs_list=None, opt_params=None, ddim=False, point_guide=False, sample_seed=0):
        super().__init__(policy, policy_tag=policy_tag,  maze_type=maze_type, obs_list=obs_list, opt_params=opt_params, ddim=ddim, sample_seed=sample_seed)
        self.drawing = False
        self.keep_drawing = False
        self.savepath = savepath
        self.draw_traj = [] # gui coordinates
        self.xy_pred = None # numpy array
        self.collisions = None # boolean array
        self.scores = None # numpy array
        self.point_guide = point_guide

    def run(self):
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
        self.trial_idx = 0

        while self.running:
            self.update_mouse_pos()

            if self.obs_list is not None:
                if self.trial_idx >= len(self.obs_list):
                    print("All observations complete.")
                    break
                self.update_agent_pos(self.xy2gui(np.array(self.obs_list[self.trial_idx])))

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if self.point_guide:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.draw_traj = [self.mouse_pos.copy()]
                        goal_xy = self.gui2xy(self.mouse_pos)
                        print(f"Guide point set at GUI: {self.mouse_pos.tolist()}, XY: {goal_xy.tolist()}")
                        self.drawing = False
                        self.keep_drawing = True
                else:
                    if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                        if not self.drawing:
                            self.drawing = True
                            self.draw_traj = []
                        self.draw_traj.append(self.mouse_pos)
                    else: # mouse released
                        if self.drawing:
                            self.drawing = False # finish drawing action
                            self.keep_drawing = True # keep visualizing the drawing
                if event.type == pygame.KEYDOWN:
                    # press s to save the trial
                    if event.key == pygame.K_s and self.savefile is not None:
                        self.save_trials()

            if self.keep_drawing and not self.point_guide: # visualize the human drawing input
                # Check if mouse returns to the agent's location
                if np.linalg.norm(self.mouse_pos - self.agent_gui_pos) < 20:  # Threshold distance to reactivate the agent
                    self.keep_drawing = False # delete the drawing
                    self.draw_traj = []

            if not self.drawing: # inference mode
                if not self.keep_drawing and self.obs_list is None:
                    self.update_agent_pos(self.mouse_pos.copy())
                if len(self.draw_traj) > 0:
                    guide = np.array([self.gui2xy(point) for point in self.draw_traj])
                else:
                    guide = None
                self.xy_pred = self.infer_target()
                self.scores = None
                if guide is not None:
                    xy_pred, scores = self.similarity_score(self.xy_pred, guide)
                    self.xy_pred = xy_pred
                    self.scores = scores
                self.collisions = self.check_collision(self.xy_pred)

            self.update_screen(self.xy_pred, self.collisions, self.scores, (self.keep_drawing or self.drawing))
            self.clock.tick(30)

        pygame.quit()

    def run_gc(self):
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
            self.trial_idx = 0

        goal_pos = None
        goal_gui_pos = None
        self.goal_set_mode = True
        self.obs_idx = 0

        while self.running:
            self.update_mouse_pos()

            # Fix agent pos from obs_list if provided
            if self.obs_list is not None:
                if self.obs_idx >= len(self.obs_list):
                    print("All observations complete.")
                    break
                current_obs = self.obs_list[self.obs_idx]
                start_xy = np.array(current_obs[:2])
                goal_xy = np.array(current_obs[2:]) if len(current_obs)==4 else None
                self.update_agent_pos(self.xy2gui(start_xy))
                # If goal provided in obs_list, skip interactive goal setting
                if goal_xy is not None and self.goal_set_mode:
                    goal_gui_pos = self.xy2gui(goal_xy)
                    goal_pos = goal_xy
                    self.goal_set_mode = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.goal_set_mode:
                        goal_gui_pos = self.mouse_pos.copy()
                        goal_pos = self.gui2xy(goal_gui_pos)
                        print(f"Goal set at GUI: {goal_gui_pos}, XY: {goal_pos}")
                        self.goal_set_mode = False
                if any(pygame.mouse.get_pressed()):
                    if not self.drawing:
                        self.drawing = True
                        self.draw_traj = []
                    self.draw_traj.append(self.mouse_pos)
                else:
                    if self.drawing:
                        self.drawing = False
                        self.keep_drawing = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_g:  # press G to reset goal
                        self.goal_set_mode = True
                    if event.key == pygame.K_s and self.savefile is not None:
                        self.save_trials()
                        if self.obs_list is not None:
                            self.obs_idx += 1
                            self.goal_set_mode = True  # ask for new goal for next obs

            if self.goal_set_mode:
                self.update_screen_goal_set()
                continue

            if self.keep_drawing:
                if np.linalg.norm(self.mouse_pos - self.agent_gui_pos) < 20:
                    self.keep_drawing = False
                    self.draw_traj = []

            if not self.drawing:
                if not self.keep_drawing and self.obs_list is None:
                    self.update_agent_pos(self.mouse_pos.copy())
                self.xy_pred = self.infer_target(goal_pos=goal_pos)
                self.scores = None
                self.collisions = self.check_collision(self.xy_pred)

            self.update_screen(self.xy_pred, self.collisions, self.scores,
                            self.keep_drawing or self.drawing, goal=goal_gui_pos)
            self.clock.tick(30)

        pygame.quit()

    def save_trials(self):
        b, t, _ = self.xy_pred.shape
        xy_pred = self.xy_pred.reshape(b*t, 2)
        pred_gui_traj = [self.xy2gui(xy) for xy in xy_pred]
        pred_gui_traj = np.array(pred_gui_traj).reshape(b, t, 2)
        entry = {
            "trial_idx": self.trial_idx,
            "agent_pos": self.agent_gui_pos.tolist(),
            "guide": np.array(self.draw_traj).tolist(),
            "pred_traj": pred_gui_traj.astype(int).tolist(),
            "collisions": self.collisions.tolist(), 
        }
        if self.obs_list is not None:
            entry['obs']= self.obs_list[self.trial_idx]

        self.savefile.write(json.dumps(entry) + "\n")
        print(f"Trial {self.trial_idx} saved to {self.savepath}.")
        self.trial_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, default=None,  help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', default=None, type=str, help="Policy name")
    parser.add_argument('-u', '--unconditional', action='store_true', help="Unconditional Maze")
    parser.add_argument('-s', '--savepath', type=str, default=None, help="Filename to save the drawing")
    parser.add_argument('-e', '--vis_energy', action='store_true', help="Visualize energy")
    parser.add_argument('--sample-seed', type=int, default=0,
                        help="Seed for the initial diffusion noise, re-applied on every "
                             "inference call (default 0). Because it is re-applied per call "
                             "with a constant batch size, one run explores the same batch_size "
                             "draws from the prior for every observation. Change it to collect "
                             "a different candidate pool; keep 0 to reproduce any dataset "
                             "collected before this flag existed.")
    parser.add_argument('-mt',  '--maze_type', default="large", type=str, help="Maze Type")
    parser.add_argument('-gc', '--goal_conditioned', action='store_true', help="Condition on goal")
    parser.add_argument('-of', '--obs-file', default=None, help="Path to loaded observations")
    parser.add_argument('-d', '--ddim', action='store_true')
    parser.add_argument('--opt_steps', type=int, default=None, help="IRED only (required unless --ddim): gradient steps on the energy per timestep")
    parser.add_argument('--t_subset', default=None, help="IRED only (required unless --ddim): optimize only the last K timesteps (an integer), or 'all'")
    parser.add_argument('--denoise', action='store_true', help="IRED only: re-predict the sample from its denoised estimate before each timestep")
    parser.add_argument('--point-guide', action='store_true', help="Use a single clicked point as guide instead of drawn line")
    args = parser.parse_args()
    if args.policy is not None and not args.ddim and (args.opt_steps is None or args.t_subset is None):
        parser.error("IRED sampling requires --opt_steps and --t_subset (an integer or 'all'); or pass --ddim")

    # Create and load the policy
    device = torch.device("cuda")

    # Load input observations
    if args.obs_file is not None:
        with open(args.obs_file) as f:
            obs_list = json.load(f)
    else:
        obs_list = None

    # Set policy type 
    if args.policy in ["diffusion", "dp"]:
        if args.checkpoint is not None:
             checkpoint_path = args.checkpoint
        else:
             checkpoint_path = 'weights_dp_energy'
    else:
        policy = None
        #raise NotImplementedError(f"Policy with name {args.policy} is not implemented.")

    # Load pretrained policy  
    if args.policy is not None:
        # Load policy
        pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))

    # Set policy parameters
    if args.policy in ["diffusion", "dp"]:
        policy_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), ["device=cuda"])
        policy = make_policy(policy_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        policy_tag = 'dp'
        policy.cuda()
        policy.eval()
    else:
        policy = None
        policy_tag = None

    # Set sampling specific parameters
    opt_params = None
    if not args.ddim and args.opt_steps is not None:
        opt_params = [{'n_opt': args.opt_steps,
                       't_subset': None if args.t_subset == 'all' else int(args.t_subset),
                       'denoise': args.denoise}]
    ddim = args.ddim
    
    if args.unconditional:
        interactiveMaze = UnconditionalMaze(policy, policy_tag=policy_tag, vis_energy=args.vis_energy, maze_type=args.maze_type, obs_list=obs_list, opt_params=opt_params, ddim=ddim, sample_seed=args.sample_seed)
    else:
        interactiveMaze = ConditionalMaze(policy, savepath=args.savepath, policy_tag=policy_tag, maze_type=args.maze_type, obs_list=obs_list, opt_params=opt_params, ddim=ddim, point_guide=args.point_guide, sample_seed=args.sample_seed)
    if args.goal_conditioned:
        interactiveMaze.run_gc()
    else:
        interactiveMaze.run()
