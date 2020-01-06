from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
try:
    import numpy
except ImportError:
    numpy = None


# extract version from __init__.py
with open('openpifpaf/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


class NumpyIncludePath(object):
    """Lazy import of numpy to get include path."""
    @staticmethod
    def __str__():
        import numpy
        return numpy.get_include()


if cythonize is not None and numpy is not None:
    EXTENSIONS = cythonize([Extension('openpifpaf.functional',
                                      ['openpifpaf/functional.pyx'],
                                      include_dirs=[numpy.get_include()]),
                           ],
                           annotate=True,
                           compiler_directives={'language_level': 3})
else:
    EXTENSIONS = [Extension('openpifpaf.functional',
                            ['openpifpaf/functional.pyx'],
                            include_dirs=[NumpyIncludePath()])]


setup(
    name='openpifpaf',
    version=VERSION,
    packages=[
        'openpifpaf',
        'openpifpaf.decoder',
        'openpifpaf.decoder.generator',
        'openpifpaf.encoder',
        'openpifpaf.network',
        'openpifpaf.transforms',
    ],
    license='GNU AGPLv3',
    description='PifPaf: Composite Fields for Human Pose Estimation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf',
    ext_modules=EXTENSIONS,
    zip_safe=False,

    install_requires=[
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'scipy',
        'torch>=1.1.0',
        'torchvision>=0.3',
        'pillow<7',  # temporary compat requirement for torchvision
    ],
    extras_require={
        'onnx': [
            'onnx',
            'onnx-simplifier',
        ],
        'test': [
            'pylint',
            'pytest',
            'opencv-python',
        ],
        'train': [
            'matplotlib',
            'pycocotools',  # pre-install cython
            'torch>=1.3.0',
            'torchvision>=0.4',
        ],
    },
)
