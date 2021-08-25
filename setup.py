from setuptools import setup, find_packages

setup(
  name = 'simple-diffusion-model',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='BSD 3-Clause',
  description = 'Simple Diffusion Model',
  author = 'Charles Foster',
  author_email = 'cfoster0@alumni.stanford.edu',
  url = 'https://github.com/cfoster0/simple-diffusion-model',
  keywords = [
    'artificial intelligence',
  ],
  install_requires=[
    'einops',
    'numpy',
    'torch',
    'torch-fidelity',
    'torchvision',
    'wandb'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
