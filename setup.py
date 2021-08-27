from setuptools import setup, find_packages
import pkg_resources
import pathlib


setup(
  name = 'visualdl',
  packages = find_packages(),
  version = '0.0.3',
  install_requires=[
'segmentation_models_pytorch @ git+https://github.com/PhilippMarquardt/segmentation_models.pytorch.git@master'
    ,
    'timm'
,
'pytorch-gradcam'
,
'albumentations'
,
'scikit-image'
,
'grad-cam'
,
'torchmetrics'
,
'tensorboard'
,
'ttach'
,
'tqdm'
,
'einops'
,
'uformer-pytorch'
,
'matplotlib'
,
'numpy',
'Pillow', 'PyYAML', 'scipy', 'seaborn>=0.11.0', 'pandas'


  ],
  license='MIT',
  description = 'VisualDL - pytorch',
  author = 'Philipp Marquardt',
  author_email = 'p.marquardt15@googlemail.com',
  url = 'https://github.com/PhilippMarquardt/VisualDL',
)
