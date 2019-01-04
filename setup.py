from setuptools import setup

#def readme():
#    with open('README.rst') as readme_file:
#        return readme_file.read()

configuration = {
    'name' : 'mnistvisualizationutils',
    'version': '0.0.1',
    'description' : 'Visualizing MNIST handwritten digits',
#    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    'keywords' : '',
    'url' : 'https://github.com/mrperkett/mnist-digits-visualization',
    'maintainer' : 'Matthew Perkett',
#    'maintainer_email' : '',
#    'license' : 'BSD',
    'packages' : ['utils'],
    'install_requires': [],
#    'install_requires': ['numpy >= 1.13',
#                         'scikit-learn >= 0.16',
#                          'scipy >= 0.19',
#                         'numba >= 0.37'],
    'ext_modules' : [],
    'cmdclass' : {},
#    'test_suite' : 'nose.collector',
    'test_suite' : '',
#    'tests_require' : ['nose'],
    'tests_require' : [],
    'data_files' : ()
    }

setup(**configuration)
