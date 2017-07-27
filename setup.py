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
      install_requires=[
         'keras>=2.0.4',
         'tensorflow>=1.2.1',
         'numpy>=1.12.1',
         'scipy>=0.19.0'
      ],
      packages=find_packages())
