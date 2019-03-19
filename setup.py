from setuptools import setup
from setuptools.extension import Extension

import numpy
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


# extract version from __init__.py
with open('openpifpaf/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


EXTENSIONS = [Extension('openpifpaf.functional',
                        ['openpifpaf/functional.pyx'],
                        include_dirs=[numpy.get_include()])]
if cythonize is not None:
    EXTENSIONS = cythonize(EXTENSIONS,
                           annotate=True,
                           compiler_directives={'language_level': 3})


setup(
    name='openpifpaf',
    version=VERSION,
    packages=[
        'openpifpaf',
        'openpifpaf.decoder',
        'openpifpaf.encoder',
        'openpifpaf.network',
    ],
    license='GNU AGPLv3',
    description='PifPaf: Composite Fields for Human Pose Estimation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf',
    ext_modules=EXTENSIONS,
    zip_safe=False,

    install_requires=[
        'matplotlib',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'scipy',
        'torch>=1.0.0',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
        'train': [
            'pycocotools',  # pre-install cython
        ],
    },
)
