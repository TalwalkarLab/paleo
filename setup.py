import re
import ast
from setuptools import setup
from setuptools import find_packages

_version_re = re.compile(r'__version__\s+=\s+\'(.*)\'')

with open('paleo/__init__.py', 'r') as f:
    _version = str(ast.literal_eval(_version_re.search(f.read()).group(1)))

setup(
    name='Paleo',
    version=_version,
    description=('An analytical model to estimate the scalability and '
                 'performance of deep learning systems.'),
    author='Hang Qi',
    author_email='hangqi@ucla.edu',
    url='https://github.com/TalwalkarLab/paleo/',
    license='Apache 2.0',
    install_requires=['click', 'numpy', 'six'],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'paleo=paleo.profiler:cli'
        ]
    })
