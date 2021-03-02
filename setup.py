from setuptools import setup, find_packages
from setuptools.extension import Extension


# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.
import os, sys
sys.path.append(os.path.dirname(__file__))
import versioneer


try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
try:
    import numpy
except ImportError as e:
    print('install numpy first')
    raise e


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
                            include_dirs=[numpy.get_include()])]


setup(
    name='openpifpaf',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    license='GNU AGPLv3',
    description='PifPaf: Composite Fields for Human Pose Estimation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf',
    ext_modules=EXTENSIONS,
    zip_safe=False,

    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'scipy',
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'pillow',
        'dataclasses; python_version<"3.7"',
    ],
    extras_require={
        'dev': [
            'cython',
            'flameprof',
            'jupyter-book>=0.9.1',
            'matplotlib>=3.3',
            'nbdime',
            'nbstripout',
            'sphinx-book-theme',
            'wheel',
        ],
        'onnx': [
            'onnx',
            'onnxruntime',
            'onnx-simplifier>=0.2.9',
        ],
        'coreml': [
            'coremltools>=4.1',
        ],
        'test': [
            'nbconvert',
            'nbval',
            'onnx',
            'onnxruntime',
            'onnx-simplifier>=0.2.9',
            'pylint',
            'pycodestyle',
            'pytest',
            'opencv-python',
            'thop',
        ],
        'train': [
            'matplotlib>=3.3',  # required by pycocotools
            'pycocotools>=2.0.1',  # pre-install cython (currently incompatible with numpy 1.18 or above)
        ],
    },
)
