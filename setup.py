from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    readme = f.read()

setup(
    name='Pet Lovers Center',
    version='0.1.0',
    description='predict the Pawpularity of pet photos',
    packages=find_packages()
)
