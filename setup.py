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
CMD_CLASS = versioneer.get_cmdclass()


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
    EXTENSIONS.append(
        torch.utils.cpp_extension.CppExtension(
            'openpifpaf._cpp',
            glob.glob(os.path.join(this_dir, 'openpifpaf', 'csrc', 'src', '**', '*.cpp'), recursive=True),
            depends=glob.glob(os.path.join(this_dir, 'openpifpaf', 'csrc', 'include', '**', '*.hpp'), recursive=True),
            include_dirs=[os.path.join(this_dir, 'openpifpaf', 'csrc', 'include')],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
    assert 'build_ext' not in CMD_CLASS
    CMD_CLASS['build_ext'] = torch.utils.cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)


add_cpp_extension()
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

    packages=setuptools.find_packages(),
    package_data={
        'openpifpaf': ['*.dll', '*.dylib', '*.so'],
    },
    cmdclass=CMD_CLASS,
    ext_modules=EXTENSIONS,
    zip_safe=False,

    python_requires='>=3.6',
    install_requires=[
        'importlib_metadata!=3.8.0',  # temporary for pytest
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pillow!=8.3.0',  # exclusion torchvision 0.10.0 compatibility
        'dataclasses; python_version<"3.7"',
    ],
    extras_require={
        'backbones': [
            'timm>=0.4.9',  # For Swin Transformer and XCiT
            'einops>=0.3',  # required for BotNet
        ],
        'dev': [
            'flameprof',
            'jupyter-book>=0.9.1',
            'matplotlib>=3.3',
            'nbdime',
            'nbstripout',
            'scipy',
            'sphinx-book-theme',
            'wheel',
        ],
        'onnx': [
            'onnx',
            'onnxruntime',
            'onnx-simplifier>=0.2.9; python_version<"3.9"',  # Python 3.9 not supported yet
        ],
        'coreml': [
            'coremltools>=5.0b3',
        ],
        'test': [
            'cpplint',
            'nbconvert',
            'nbstripout',
            'nbval',
            'onnx',
            'onnxruntime',
            'onnx-simplifier>=0.2.9; python_version<"3.9"',  # Python 3.9 not supported yet
            'pylint<2.9.4',  # avoid 2.9.4 and up for time.perf_counter deprecation warnings
            'pycodestyle',
            'pytest',
            'opencv-python',
            'thop',
        ],
        'train': [
            'matplotlib>=3.3',  # required by pycocotools
            'pycocotools>=2.0.1',  # pre-install cython (currently incompatible with numpy 1.18 or above)
            'scipy',
            'xtcocotools>=1.5; sys_platform == "linux"',  # required for wholebody eval, only wheels and only for linux on pypi
        ],
    },
)
