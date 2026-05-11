import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.pick_place import PickPlace

class BinPlacingTask(PickPlace):
    """
    Extend PickPlace for bin placing task.
    Object starts in gripper, task is to place in specified bin.
    """
    
    AVAILABLE_OBJECTS = {
        "cube": "Cube",
        "cylinder": "Cylinder",
        "sphere": "Sphere",
        "mug": "Mug",
        "teapot": "Teapot",
        "can": "Can",
        "bottle": "Bottle",
    }
    
    def __init__(self, objects=None, randomize_object=False, **kwargs):
        if objects is None:
            objects = ["cube"]
        self.object_list = objects
        self.randomize_object = randomize_object
        self.current_object = objects[0]
        self._next_object = None
        self._next_bin = None
        
        self.bin_positions = np.array([
            [-0.25, -0.25, 0.82],
            [-0.25,  0.25, 0.82],
            [ 0.25, -0.25, 0.82],
            [ 0.25,  0.25, 0.82],
        ])
        self.bin_radius = 0.08
        self.current_goal_bin = None
        
        super().__init__(**kwargs)
    
    def _setup_references(self):
        super()._setup_references()
    
    def _reset_arena(self):
        """Reset with object selection and placement in gripper"""
        # Determine object for this episode
        if self._next_object is not None:
            self.current_object = self._next_object
            self._next_object = None
        elif self.randomize_object:
            self.current_object = np.random.choice(self.object_list)
        
        # Remove old object if exists
        if hasattr(self, "object_id") and self.object_id is not None:
            mujoco_object = self.mujoco_objects[0]
            self.model.delete_body(mujoco_object.root_body)
        
        # Add new object
        mujoco_object_class = suite.models.OBJECTS[self.AVAILABLE_OBJECTS[self.current_object]]
        mujoco_object = mujoco_object_class()
        self.mujoco_objects = [mujoco_object]
        self.model.add_object(mujoco_object)
        
        # Get references
        self.object_id = self.model.body_name2id(mujoco_object.root_body)
        self.obj_body_id = self.sim.model.body_name2id(mujoco_object.root_body)
        
        # Place object near gripper
        random_ee_pos = self.random_start()
        object_pos = random_ee_pos + np.array([0, 0, 0.05])
        self.sim.data.set_body_xpos(self.obj_body_id, object_pos)
        
        # Close gripper
        gripper_joint_indices = self.robot.gripper.joint_indexes
        self.sim.data.qpos[gripper_joint_indices] = self.robot.gripper.init_qpos
        
        # Move robot EE to random start
        self.robot.set_eef_site_pos(random_ee_pos)
        
        # Set goal bin
        if self._next_bin is not None:
            self.current_goal_bin = self._next_bin
            self._next_bin = None
        else:
            self.current_goal_bin = np.random.randint(0, 4)
    
    def random_start(self):
        """Sample random EE start position above the bin area"""
        x_range = [-0.35, 0.35]
        y_range = [-0.35, 0.35]
        z_height = 0.45
        
        ee_x = np.random.uniform(x_range[0], x_range[1])
        ee_y = np.random.uniform(y_range[0], y_range[1])
        ee_z = z_height
        
        return np.array([ee_x, ee_y, ee_z])
    
    def goal(self, bin_idx):
        """Sample random goal location within specified bin"""
        if bin_idx < 0 or bin_idx >= 4:
            raise ValueError(f"bin_idx must be 0-3, got {bin_idx}")
        
        bin_center = self.bin_positions[bin_idx]
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, self.bin_radius * 0.8)
        
        goal_x = bin_center[0] + radius * np.cos(angle)
        goal_y = bin_center[1] + radius * np.sin(angle)
        goal_z = bin_center[2] - 0.02
        
        return np.array([goal_x, goal_y, goal_z])
    
    def set_next_object(self, object_name):
        """Set object for next reset"""
        if object_name not in self.object_list:
            raise ValueError(f"Object {object_name} not in available list: {self.object_list}")
        self._next_object = object_name
    
    def set_next_bin(self, bin_idx):
        """Set bin for next reset"""
        if bin_idx < 0 or bin_idx >= 4:
            raise ValueError(f"bin must be 0-3, got {bin_idx}")
        self._next_bin = bin_idx
    
    def reward(self, action):
        """Simple placement reward"""
        object_pos = self.sim.data.body_xpos[self.obj_body_id]
        goal_pos = self.bin_positions[self.current_goal_bin]
        distance = np.linalg.norm(object_pos - goal_pos)
        reward = -distance
        
        if distance < self.bin_radius * 0.5:
            reward += 10.0
        
        return reward
    
    def _get_observation(self):
        """Return observation"""
        obs = super()._get_observation()
        obs["goal_bin"] = np.array([self.current_goal_bin])
        obs["bin_positions"] = self.bin_positions.flatten()
        
        object_one_hot = np.zeros(len(self.object_list))
        object_one_hot[self.object_list.index(self.current_object)] = 1.0
        obs["object_type"] = object_one_hot
        
        return obs


def test_environment():
    """Test the environment"""
    
    object_list = ["cube", "cylinder", "mug"]
    
    env = suite.make(
        env_name="BinPlacingTask",
        robots="Panda",
        has_renderer=True,
        render_camera="frontview",
        use_camera_obs=False,
        objects=object_list,
        randomize_object=False,
    )
    
    print("Environment created!")
    
    # Test 1: Specific object and bin
    print("\n--- Test 1: Specific object and bin ---")
    env.set_next_object("cylinder")
    env.set_next_bin(2)
    obs = env.reset()
    print(f"Current object: {env.current_object}")
    print(f"Current goal bin: {env.current_goal_bin}")
    
    for step in range(50):
        action = np.random.randn(7) * 0.1
        obs, reward, done, info = env.step(action)
        env.render()
    
    env.close()
    print("✓ Test complete!")


if __name__ == "__main__":
    test_environment()