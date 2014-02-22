# Import sys and add the path so we can check that the package imports
# correctly, and so that we can grab the version from the package.
import __main__, os, sys
sys.path.insert(0, os.path.join(
    # Try to get the name of this file. If we can't, go with the CWD.
    os.path.dirname(os.path.abspath(__main__.__file__))
    if hasattr(__main__, '__file__') else
    os.getcwd(), 
    'src'
))
import qinfer

from distutils.core import setup

setup(
    name='QInfer',
    version=qinfer.__version__,
    url='https://github.com/csferrie/python-qinfer',
    author='Chris Ferrie and Chris Granade',
    author_email='csferrie@gmail.com',
    package_dir={'': 'src'},
    packages=[
        'qinfer',
        'qinfer._lib',
        'qinfer.examples',
        'qinfer.ui',
    ]
)
