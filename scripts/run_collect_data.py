from bbsim.envs import BaseDigitalTwinEnv

import numpy as np
import cv2
import supervision as sv

from pathlib import Path

import argparse

def visualize_detections(image: np.ndarray, bboxes: np.ndarray) -> None:
    """
    Visualize detections on the image.
    """
    if len(bboxes) > 0:
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        detections = sv.Detections(
            xyxy=bboxes,
            confidence=np.asarray([1.] * len(bboxes)),
            class_id=np.asarray([0] * len(bboxes)),
        )

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # 添加标签（可自定义）
        labels = [f"Class {class_id} {confidence:.2f}" 
                for class_id, confidence in zip(detections.class_id, detections.confidence)]

        # 叠加边界框和标签到图像上
        image = bounding_box_annotator.annotate(scene=image, detections=detections)
        image = label_annotator.annotate(scene=image, detections=detections, labels=labels)

        # 显示图像
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)
    else:
        print("No bounding boxes to visualize.")

class DetectionDatasetWriter:
    def __init__(self, path: str, camera_width: int, camera_height: int, dataset_size: int):
        self.path = path
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.dataset_size = dataset_size

        self.train_dataset_path = Path(self.path) / "train"
        self.val_dataset_path = Path(self.path) / "val"

    def write(self, dataset_cnt: int, obs: dict):
        if dataset_cnt >= int(0.7 * self.dataset_size):
            print("Switching to validation dataset.")
            dataset_path = self.val_dataset_path
        else:
            dataset_path = self.train_dataset_path

        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dataset_path / "images" / f"{dataset_cnt:06d}.png"), obs["rgb"])

        labels_path = dataset_path / "labels" / f"{dataset_cnt:06d}.txt"
        with open(labels_path, "w") as f:
            for bbox in obs["bboxes"]:
                x_min, y_min, w, h = self._xxyy_to_xywh(bbox)
                
                x_center = (x_min + w / 2) / self.camera_width
                y_center = (y_min + h / 2) / self.camera_height
                w_norm = w / self.camera_width
                h_norm = h / self.camera_height

                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    def _xxyy_to_xywh(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box from xyxy format to xywh format.
        """
        x_min, y_min, x_max, y_max = bbox
        return np.array([x_min, y_min, x_max - x_min, y_max - y_min])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BaseDigitalTwinEnv to collect data.")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to collect.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--camera_width", type=int, default=640, help="Width of the camera image.")
    parser.add_argument("--camera_height", type=int, default=480, help="Height of the camera image.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the detections.")
    parser.add_argument("--record", action="store_true", help="Record the Observation.")
    parser.add_argument("--record_path", type=str, default="data/recordings", help="Path to save the recorded data.")
    args = parser.parse_args()

    env = BaseDigitalTwinEnv(
        render=args.render, 
        sensor_camera_width=args.camera_width, 
        sensor_camera_height=args.camera_height
    )
    writer = DetectionDatasetWriter(
        path=args.record_path, 
        camera_width=args.camera_width, 
        camera_height=args.camera_height, 
        dataset_size=args.num_images
    )
    
    seed = args.seed
    env.reset(seed)

    num_steps = 0
    dataset_cnt = 0
    while True:
        env.step()
        num_steps += 1

        if (num_steps + 1) % 200 == 0:
            num_steps = 0

            obs = env.get_obs()
            if args.visualize:
                visualize_detections(obs["rgb"], obs["bboxes"])

            if args.record:
                writer.write(dataset_cnt, obs)
            dataset_cnt += 1
            print("Collected data: {}/{}".format(dataset_cnt, args.num_images))
            if dataset_cnt >= args.num_images:
                break

            seed += 1
            env.reset(seed)