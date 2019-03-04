from setuptools import setup, Extension

# extract version from __init__.py
with open('openpifpaf/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


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
    description='PifPaf: Association Fields for Human Pose Estimation',
    long_description=open('README.rst').read(),
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf',
    ext_modules=[
        Extension('openpifpaf.functional',
                  sources=['openpifpaf/functional.pyx']),
    ],

    setup_requires=[
        'numpy',
        'cython',
        'setuptools>=40.0.0',
    ],
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
