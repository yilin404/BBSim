### BBSim

This project is built on Sapien and is designed to generate object detection datasets for detecting blocks in various colors.

#### Installing
```
conda create -n bbsim python=3.10

conda activate bbsim
pip3 install sapien==3.0.0b1
pip3 install supervision==0.25.1

cd bbsim
pip3 install -e .
```

#### Using
To collect data, run:
```bash
python3 scripts/run_collect_data.py --record
```

To render and inspect your collected dataset, run:
```bash
python3 scripts/run_collect_data.py --render --visualize
```
Note: We use OpenCV for visualization and call cv2.waitKey(0) to prevent the image window from closing immediately. Press any key in the detection result window to proceed to the next frame.

To record a custom number of images, add the `--num_images` flag:
```bash
python3 scripts/run_collect_data.py --record --num_images <number_of_images>
```

Dataset Format:
[Ultralytics YOLO](https://docs.ultralytics.com/zh/datasets/detect/)
