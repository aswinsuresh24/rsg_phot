from setuptools import setup, find_packages

setup(
    name='rsg_phot',
    version='0.1', 
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'astropy',
        'matplotlib',
        'tqdm',
        'dustmaps',
        'dust-extinction',
        'sbi',
        'torch',
        'emcee',
        'progressbar2',
        'synphot',
        'scipy',
        'corner',        
    ],
    author='Aswin Suresh',

    description='A package for photometric modeling of red supergiants using JWST data',
    url='https://github.com/aswinsuresh24/rsg_phot',
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)   