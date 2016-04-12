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

try:
    with open('README.rst', 'r') as readme:
        long_description = readme.read()
except:
    long_description = ''

setup(
    name='QInfer',
    version=qinfer.__version__,
    url='https://github.com/QInfer/python-qinfer',
    download_url='https://github.com/QInfer/python-qinfer/archive/v1.0b1.tar.gz',
    author='Chris Granade and Chris Ferrie',
    author_email='cgranade@cgranade.com',
    maintainer='Chris Granade and Chris Ferrie',
    maintainer_email='cgranade@cgranade.com',
    package_dir={'': 'src'},
    packages=[
        'qinfer',
        'qinfer._lib',
        'qinfer.examples',
        'qinfer.ui',
        'qinfer.experimental',
        'qinfer.tomography'
    ],
    keywords=['quantum', 'Bayesian', 'estimation'],
    description=
        'Bayesian particle filtering for parameter estimation in quantum '
        'information applications.',
    long_description=long_description,
    license='http://www.gnu.org/licenses/agpl-3.0.en.html',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    platforms=['any']
)
