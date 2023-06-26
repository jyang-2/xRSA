from setuptools import setup

setup(
        name='xRSA',
        version='2023.06',
        packages=['expt', 'xrsa', 'xrsa.vis', 'stimuli', 'external', 'external.suite2p',
                  'ryeutils'],
        package_dir={'': 'src'},
        url='https://github.com/jyang-2/xRSA',
        license='GNU GPLv3',
        author='remy',
        author_email='jyang2@caltech.edu',
        description='Functions for processing neural timeseries data and performing RSA analysis'
        )
