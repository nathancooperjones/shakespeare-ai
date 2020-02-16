from setuptools import find_packages, setup

with open('shakespeare_ai/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='shakespeare_ai',
    version=__version__,
    description="It's no Shakespeare, taken to a whole new level.",
    long_description=readme,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
    ]
)
