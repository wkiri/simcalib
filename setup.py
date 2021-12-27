import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'simcalib',
    version = '1.0.0',
    author = 'Kiri Wagstaff',
    author_email = 'kiri.wagstaff@oregonstate.edu',
    description = 'Similarity-based calibration methods',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/wkiri/simcalib',
    packages = setuptools.find_packages(),
    install_requires = ['numpy',
                        'scipy',
                        'matplotlib',
                        'sklearn',
                        'progressbar',
                        'uncertainty-calibration'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)
