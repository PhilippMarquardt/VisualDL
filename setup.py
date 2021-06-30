from setuptools import setup, find_packages
import pkg_resources
import pathlib

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
  name = 'visualdl',
  packages = find_packages(),
  version = '0.0.2',
  install_requires=install_requires,
  license='MIT',
  description = 'VisualDL - pytorch',
  author = 'Philipp Marquardt',
  author_email = 'p.marquardt15@googlemail.com',
  url = 'https://github.com/PhilippMarquardt/VisualDL',
)
