from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
try:
    import numpy
except ImportError:
    numpy = None

import versioneer


class NumpyIncludePath():
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
        'torch>=1.7',
        'torchvision>=0.8.1',
        'pillow',
        'dataclasses; python_version<"3.7"',
    ],
    extras_require={
        'dev': [
            'flameprof',
            'jupyter-book>=0.8.0',
            'sphinxcontrib-bibtex<2.0.0',
            'matplotlib',
            'nbdime',
            'nbstripout',
        ],
        'onnx': [
            'onnx',
            'onnxruntime',
            'onnx-simplifier>=0.2.9',
        ],
        'coreml': [
            'coremltools==4.0b3',
        ],
        'test': [
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
            'matplotlib',  # required by pycocotools
            'pycocotools>=2.0.1',  # pre-install cython (currently incompatible with numpy 1.18 or above)
        ],
    },
)
