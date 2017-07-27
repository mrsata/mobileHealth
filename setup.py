from setuptools import setup
from setuptools import find_packages


setup(name='mobileHealth',
      version='0.1.0',
      description='Deep Reinforcement Learning for Mobile Health',
      author='Han Liu',
      author_email='liuhanrick@gmail.com',
      url='https://github.com/mrsata/mobileHealth',
      download_url='https://github.com/mrsata/mobileHealth/archive/master.zip',
      license='MIT',
      install_requires=['keras', 'tensorflow'],
      packages=find_packages())
