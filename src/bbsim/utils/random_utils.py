# Sapien
import sapien

import numpy as np
import transforms3d as t3d

from ..common.constants import ASSET_DIR

def random_pose(
    position_reference: np.ndarray | list[float], # xyz
    x_noise_range: np.ndarray | list[float], # x坐标变化范围
    y_noise_range: np.ndarray | list[float], # y坐标变化范围
    z_noise_range: np.ndarray | list[float], # z坐标变化范围
    rotation_reference: np.ndarray | list[float], # euler angle, radians
    euler_angles_noise_range: np.ndarray | list[float], # 欧拉角变化范围, radians
) -> sapien.Pose:
    def _sanitize_value_range(value_range: np.ndarray) -> np.ndarray:
        if len(value_range) < 2 or value_range[1] < value_range[0]:
            return np.array([value_range[0], value_range[0]])
        return np.array(value_range)
    x_noise_range = _sanitize_value_range(x_noise_range)
    y_noise_range = _sanitize_value_range(y_noise_range)
    z_noise_range = _sanitize_value_range(z_noise_range)

    noised_euler_angles = [0, 0, 0]
    for i in range(3):
        noised_euler_angles[i] = np.random.uniform(
            -euler_angles_noise_range[i], 
            euler_angles_noise_range[i]
    )
    if len(rotation_reference) == 3:
        quaternion_reference = t3d.euler.euler2quat(rotation_reference[0], rotation_reference[1], rotation_reference[2])
    elif len(rotation_reference) == 4:
        quaternion_reference = np.asarray(rotation_reference)
    else:
        raise ValueError('rotation_reference should be a list of 3 or 4 elements!')
    quaternion = t3d.quaternions.qmult(
        quaternion_reference, 
        t3d.euler.euler2quat(
            noised_euler_angles[0], 
            noised_euler_angles[1], 
            noised_euler_angles[2]
        )
    )
    
    noised_x = np.random.uniform(x_noise_range[0], x_noise_range[1])
    noised_y = np.random.uniform(y_noise_range[0], y_noise_range[1])
    noised_z = np.random.uniform(z_noise_range[0], z_noise_range[1])
    position = np.array(position_reference) + t3d.quaternions.rotate_vector(np.array([noised_x, noised_y, noised_z]), quaternion_reference)
    
    return sapien.Pose(position, quaternion)


# Texture Augmentation
def get_texture_files(texture_name: str, texture_formats: list[str] = [".jpg", ".png", ".jpeg"]) -> list:
    texture_dir = ASSET_DIR / f"textures/{texture_name}"
    if not texture_dir.exists():
        raise FileNotFoundError(f"Texture directory {texture_dir} does not exist.")
    texture_files = []
    for fmt in texture_formats:
        texture_files.extend(list(texture_dir.glob(f"*{fmt}")))
    if not texture_files:
        raise FileNotFoundError(f"No texture files found in {texture_dir}.")
    return sorted(texture_files)

def get_texture_count(texture_name: str) -> int:
    return len(get_texture_files(texture_name))

def get_texture_path(texture_name: str, texture_id: int = 0) -> str:
    texture_files = get_texture_files(texture_name)
    if texture_id < 0 or texture_id >= len(texture_files):
        raise IndexError(f"Texture id {texture_id} out of range.")
    return str(texture_files[texture_id])

def set_random_background_texture(
    entity: sapien.Entity | list[sapien.Entity],
):
    if not isinstance(entity, list):
        entity = [entity]
    
    def _random_background():
        subfolders = [f for f in (ASSET_DIR / "textures/backgrounds").iterdir() if f.is_dir()]
        if not subfolders:
            raise FileNotFoundError("No background textures found.")
        
        counts = [get_texture_count(str(f.relative_to(ASSET_DIR / "textures"))) for f in subfolders]
        total = sum(counts)
        if total == 0:
            raise FileNotFoundError("Found background folders but none contain textures.")
        probs = [c / total for c in counts]

        selected_subfolder = np.random.choice(subfolders, p=probs)
        return str(selected_subfolder.relative_to(ASSET_DIR / "textures"))
    texture_name = _random_background()
    texture_count = get_texture_count(texture_name)
    if np.random.uniform(0., 1.) > 1. / (texture_count + 1):
        texture_id = np.random.randint(0, texture_count)
        texture_path = get_texture_path(texture_name, texture_id)
        texture = sapien.render.RenderTexture2D(
            filename=texture_path,
        )

        for e in entity:
            render_body_component = e.find_component_by_type(sapien.render.RenderBodyComponent)
            for render_shape in render_body_component.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color_texture(texture)
                    part.material.base_color = [1, 1, 1, 1]
                    part.material.metallic = 0.1
                    part.material.roughness = 0.3
    
    if len(entity) == 1:
        entity = entity[0]
    
    return entity