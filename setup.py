from setuptools import setup, find_packages

setup(
    name='bbsim', # 你项目的名字，可自定义
    version='0.1.0',
    description='A twin framework for object detection',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "sapien==3.0.0b1",
        "numpy==1.26.4",
        "supervision=0.25.1"
    ],
    python_requires='>=3.10',
)