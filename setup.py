from setuptools import setup, find_packages
import pkg_resources
import pathlib


setup(
  name = 'visualdl',
  packages = find_packages(),
  version = '0.0.2',
  install_requires=[
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

  ],
  dependency_links = ['git+https://github.com/PhilippMarquardt/segmentation_models.pytorch.git'],
  license='MIT',
  description = 'VisualDL - pytorch',
  author = 'Philipp Marquardt',
  author_email = 'p.marquardt15@googlemail.com',
  url = 'https://github.com/PhilippMarquardt/VisualDL',
)
