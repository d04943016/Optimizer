from setuptools import setup
from Optimizer import __version__

setup(name='optimizer',
      version=__version__,
      license='MIT License',
      install_requires=['numpy','scipy'],
      description='Optimizer',
      author='Wei-Kai Lee',
      author_email='d04943016@ntu.edu.tw',
      url='',
      packages=['Optimzer'],
      )