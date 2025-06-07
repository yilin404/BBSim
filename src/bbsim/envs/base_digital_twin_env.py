# Sapien
import sapien
from sapien.utils.viewer import Viewer

import numpy as np

import random

from ..utils.create_utils import create_box
from ..utils.random_utils import random_pose, set_random_background_texture
from ..utils.sapien_utils import look_at

class BaseDigitalTwinEnv:
    # Wall Params
    WALL_WIDTH = 15.0
    WALL_THICKNESS = 0.2
    WALL_HEIGHT = 2.5

    # Cube Params
    MAX_NUM_CUBE = 10
    CUBE_HALF_SIZE = [0.02, 0.02, 0.02]  # Half size of the cube

    # Messy objects Params
    MAX_NUM_MESSY = 10

    # Sensor Camera Params
    SENSOR_CAMERA_FOVY = 42 # RealSense D435i
    SENSOR_CAMERA_NEAR = 0.01
    SENSOR_CAMERA_FAR = 100

    def __init__(self, render: bool = True, sensor_camera_width: int = 640, sensor_camera_height: int = 480):
        self._render = render

        self._sensor_camera_width = sensor_camera_width
        self._sensor_camera_height = sensor_camera_height

    def _setup_scene(
        self, 
        timestep: float = 1 / 150, 
        ground_height: float = 0.,
        static_friction: float = 0.5, 
        dynamic_friction: float = 0.5, 
        restitution: float = 0.,
        shadow: bool = True,
        direction_lights: list[list[float]] = [[[0, 0.5, -1], [0.5, 0.5, 0.5]]],
        point_lights: list = [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]],
        viewer_camera_xyz: list[float] = [0.4, 0.22, 1.5], 
        viewer_camera_rpy: list[float] = [0, -0.8, 2.45],
    ):
        '''
        Set the scene
            - Set up the basic scene: light source, viewer.
        '''
        sapien.render.set_camera_shader_dir("rt")
        # sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(4)
        sapien.render.set_ray_tracing_path_depth(3)
        sapien.render.set_ray_tracing_denoiser("optix")

        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config
        set_global_config(max_num_materials = 50000, max_num_textures = 50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.scene.set_timestep(timestep)
        # add ground to scene
        self.ground = self.scene.add_ground(ground_height, render_half_size=[10, 10])
        self.ground = set_random_background_texture(self.ground,) # Domain Randomization: random texture
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            static_friction,
            dynamic_friction,
            restitution,
        )
        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light([
            np.random.uniform(0.2, 0.6), 
            np.random.uniform(0.2, 0.6), 
            np.random.uniform(0.2, 0.6)
        ]) # Domain Randomization: random ambient light
        # default spotlight angle and intensity
        for direction_light in direction_lights:
            self.scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        # default point lights position and intensity
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

        # initialize viewer with camera position and orientation
        if self._render:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(
                x=viewer_camera_xyz[0],
                y=viewer_camera_xyz[1],
                z=viewer_camera_xyz[2],
            )
            self.viewer.set_camera_rpy(
                r=viewer_camera_rpy[0],
                p=viewer_camera_rpy[1],
                y=viewer_camera_rpy[2],
            )
    
    def _create_wall(self):
        width = self.WALL_WIDTH + np.random.uniform(-2.5, 2.5) # Domain Randomization: random wall width
        height = self.WALL_HEIGHT + np.random.uniform(-0.5, 0.5) # Domain Randomization: random wall height

        # creat wall
        self.forward_wall = create_box(
            self.scene,
            sapien.Pose(p=[0, (width + self.WALL_THICKNESS) / 2, height / 2]),
            half_size=[width / 2, self.WALL_THICKNESS / 2, height / 2],
            color=(1, 0.9, 0.9), 
            name="forward_wall",
            is_static=True
        )
        self.backward_wall = create_box(
            self.scene,
            sapien.Pose(p=[0, -(width + self.WALL_THICKNESS) / 2, height / 2]),
            half_size=[width / 2, self.WALL_THICKNESS / 2, height / 2],
            color=(1, 0.9, 0.9),
            name="backward_wall",
            is_static=True
        )
        self.left_wall = create_box(
            self.scene,
            sapien.Pose(p=[-(width + self.WALL_THICKNESS) / 2, 0, height / 2]),
            half_size=[self.WALL_THICKNESS / 2, width / 2, height / 2],
            color=(1, 0.9, 0.9), 
            name="left_wall",
            is_static=True
        )
        self.right_wall = create_box(
            self.scene,
            sapien.Pose(p=[(width + self.WALL_THICKNESS) / 2, 0., height / 2]),
            half_size=[self.WALL_THICKNESS / 2, width / 2, height / 2],
            color=(1, 0.9, 0.9), 
            name="right_wall",
            is_static=True
        )
        self.forward_wall, self.backward_wall, self.left_wall, self.right_wall = \
            set_random_background_texture(
                [self.forward_wall, self.backward_wall, self.left_wall, self.right_wall], 
            ) # Domain Randomization: random texture
    
    def _load_entities(self):
        self.rigid_objects: dict[str, sapien.Entity] = dict()
        self.articulations: dict[str, sapien.physx.PhysxArticulation] = dict()

        for i in range(np.random.randint(1, self.MAX_NUM_CUBE + 1)):
            cube = create_box(
                self.scene,
                random_pose(
                    position_reference=[0., 0., 0.5,],
                    x_noise_range=[-1., 0.], y_noise_range=[-1., 0.], z_noise_range=[0.],
                    rotation_reference=[0., 0., 0.],
                    euler_angles_noise_range=[np.pi, np.pi, np.pi],
                ),
                half_size=self.CUBE_HALF_SIZE,
                color=[np.random.uniform(0., 1.), np.random.uniform(0., 1.), np.random.uniform(0., 1.)],
                is_static=False,
            )

            self.rigid_objects["cube_{}".format(i)] = cube

        for i in range(np.random.randint(1, self.MAX_NUM_MESSY + 1)):
            # TODO: add messy objects
            pass

    def _load_camera(self):
        self.camera = self.scene.add_camera(
            name="camera",
            width=self._sensor_camera_width,
            height=self._sensor_camera_height,
            fovy=self.SENSOR_CAMERA_FOVY,
            near=self.SENSOR_CAMERA_NEAR,
            far=self.SENSOR_CAMERA_FAR,
        )

        self.camera.set_pose(
            look_at(
                eye=np.array([-1., -1., np.random.uniform(0.25, 0.5)]),
                target=np.array([0., 0., 0.]),
                up=np.array([0., 0., 1.]),
            )
        )

    def _reconfigure(self, seed: int = 0):
        print("=> Set Seed")
        print("Seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed=seed)

        print("=> Setup Scene")
        self._setup_scene()

        print("=> Create Wall")
        self._create_wall()

        print("=> Load Entities")
        self._load_entities()

        print("=> Load Camera")
        self._load_camera()

    def reset(self, seed: int = 0):
        '''
        Reset the environment.
        '''
        if self._render and hasattr(self, 'viewer'):
            self.viewer.close()

        self._reconfigure(seed=seed)

    def step(self):
        '''
        Step the environment.
        '''
        self.scene.step()
        self.scene.update_render()
        if self._render:
            self.viewer.render()

    def get_obs(self):
        '''
        Get the observation.
        '''
        self.camera.take_picture()

        # Get the RGB image
        rgba = self.camera.get_picture("Color") # [H, W, 4]
        color_img = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)

        # Get the bounding box
        entity_segmentation = self.camera.get_picture("Segmentation")[..., 1] # [H, W]

        cubes_id = [obj.per_scene_id for obj_name, obj in self.rigid_objects.items() if obj_name.startswith("cube_")]
        bboxes = []
        for id in cubes_id:
            mask = np.isin(entity_segmentation, [id])  # [H, W]
            ys, xs = np.where(mask)
            if ys.size > 0:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                bboxes.append([x_min, y_min, x_max, y_max]) # xyxy format

        return dict(
            rgb=color_img,
            bboxes=np.array(bboxes, dtype=np.float32),
        )
