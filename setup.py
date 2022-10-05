import glob
import os
import setuptools
import sys
import torch.utils.cpp_extension


# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.
sys.path.append(os.path.dirname(__file__))
import versioneer


EXTENSIONS = []
CMD_CLASS = {}


def add_cpp_extension():
    extra_compile_args = [
        '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17',
    ]
    extra_link_args = []
    define_macros = [
        ('_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS', None),  # mostly for the pytorch codebase
    ]

    if sys.platform.startswith('win'):
        extra_compile_args += ['/permissive']
        define_macros += [('OPENPIFPAF_DLLEXPORT', None)]

    if os.getenv('DEBUG', '0') == '1':
        print('DEBUG mode')
        if sys.platform.startswith('linux'):
            extra_compile_args += ['-g', '-Og']
            extra_compile_args += [
                '-Wuninitialized',
                # '-Werror',  # fails in pytorch code, but would be nice to have in CI
            ]
        define_macros += [('DEBUG', None)]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join('src', 'openpifpaf', 'csrc')
    EXTENSIONS.append(
        torch.utils.cpp_extension.CppExtension(
            'openpifpaf._cpp',
            glob.glob(os.path.join(csrc_dir, 'src', '**', '*.cpp'), recursive=True),
            depends=glob.glob(os.path.join(this_dir, csrc_dir, 'include', '**', '*.hpp'), recursive=True),
            include_dirs=[os.path.join(this_dir, csrc_dir, 'include')],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
    assert 'build_ext' not in CMD_CLASS, f"build_ext in CMD_CLASS: {CMD_CLASS}"
    CMD_CLASS['build_ext'] = torch.utils.cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)


add_cpp_extension()
CMD_CLASS = versioneer.get_cmdclass(CMD_CLASS)
setuptools.setup(
    name='openpifpaf',
    version=versioneer.get_version(),
    license='GNU AGPLv3',
    description='PifPaf: Composite Fields for Human Pose Estimation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/openpifpaf/openpifpaf',

    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        'openpifpaf': ['*.dll', '*.dylib', '*.so'],
    },
    cmdclass=CMD_CLASS,
    ext_modules=EXTENSIONS,
    zip_safe=False,

    python_requires='>=3.7',
    install_requires=[
        'importlib_metadata!=3.8.0',  # temporary for pytest
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'torch==1.12.1',
        'torchvision==0.13.1',
        'pillow!=8.3.0',  # exclusion torchvision 0.10.0 compatibility
        'dataclasses; python_version<"3.7"',
    ],
    extras_require={
        'dev': [
            'flameprof',
            'ipython<8',  # temporarily added to avoid broken output cells in jupyter-book
            'jupyter-book>=0.9.1',
            'matplotlib>=3.3',
            'nbdime',
            'nbstripout',
            'scipy',
            'sphinx-book-theme',
            'wheel',
        ],
        'extras': [
            'openpifpaf-extras>=0.0.3',
        ],
        'onnx': [
            'onnx',
            'onnxruntime',
            'onnx-simplifier>=0.2.9; python_version<"3.9"',  # Python 3.9 not supported yet
            'protobuf<4',  # temporary explicit dependency until tests pass again
        ],
        'coreml': [
            'coremltools>=5.0b3',
        ],
        'test': [
            'cpplint',
            'nbconvert<7',
            'nbstripout',
            'nbval',
            'opencv-python',
            'pylint<2.9.4',  # avoid 2.9.4 and up for time.perf_counter deprecation warnings
            'pycodestyle',
            'pytest',
            'thop; python_version<"3.10"',
        ],
        'train': [
            'matplotlib>=3.3',  # required by pycocotools
            'pycocotools>=2.0.1,!=2.0.5',  # pre-install cython (currently incompatible with numpy 1.18 or above)
            'scipy',
            'xtcocotools>=1.5; sys_platform == "linux"',  # required for wholebody eval, only wheels and only for linux on pypi
        ],
    },
)
