from setuptools import setup

setup(
    name='rl_tutorial',
    version='0.1',
    description='Reinforcement learning tutorial package',
    author='Kishansingh Rajput',
    author_email='kishan@jlab.org',
    packages=['rl_tutorial'],
    install_requires=['gym==0.21.0', 'numpy==1.21.6', 'tqdm', 'seaborn', 'tensorflow==2.8']
)
