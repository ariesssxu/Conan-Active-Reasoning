import setuptools
import pathlib


setuptools.setup(
    name='conan',
    version='1.0.0',
    description='Open world survival game for reinforcement learning.',
    url='',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['conan'],
    package_data={},
    entry_points={'console_scripts': ['conan=conan.run_gui:main']},
    install_requires=[
        'numpy', 'imageio', 'astar', 'pygame', 'tqdm', 'pillow', 'opensimplex', 'ruamel.yaml', 'gym',
        # Numba is an optional dependency but we want it installed by default
        # because it speeds up world generation by ~5x.
        'numba',
    ],
    extras_require={'gui': ['pygame'], 'training': ['stable_baselines3', 'transformers', 'torch', 'torchvision', 'datasets']},
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)