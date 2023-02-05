import os
import re
import sys
import platform
import subprocess
import setuptools
from setuptools import setup, Extension, find_packages
from shutil import copyfile, copymode, move
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from distutils.command.install_data import install_data

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the egg-info
    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """

    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """

        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated

        self.outfiles = self.distribution.data_files

class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles
    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step

        bin_dir = self.distribution.bin_dir
        lib_dir = self.distribution.lib_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run

        libs = []

        for root, _, files in os.walk(bin_dir):
            for _lib in files:
                if os.path.isfile(os.path.join(root, _lib)) and _lib.startswith("libcaffe"):
                    libs.append(os.path.join(root, _lib))

        for root, _, files in os.walk(lib_dir):
            for _lib in files:
                if os.path.isfile(os.path.join(root, _lib)) and (_lib.startswith("libcaffe") or _lib.startswith("_caffe")) and os.path.isfile(os.path.join(root, _lib)):
                    libs.append(os.path.join(root, _lib))

        for lib in libs:
            move(lib, os.path.join(self.build_dir, 'caffe', os.path.basename(lib)))

        # Mark the libs for installation, adding them to
        # distribution.data_files seems to ensure that setuptools' record
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list,
        # but that never seemed to add them to the installed-files.txt in the
        # egg-info, and the online recommendation seems to be adding libraries
        # into eager_resources in the call to setup(), which I think puts them
        # in data_files anyways.
        #
        # What is the best way?

        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code

        self.distribution.data_files = [os.path.join(self.install_dir, 'caffe', os.path.basename(lib))
                                        for lib in libs]
        # Must be forced to run after adding the libs to data_files

        self.distribution.run_command("install_data")

        super().run()

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['conda', '--version'])      
        except OSError:
            raise RuntimeError(
                "conda must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if sys.version_info.major == 3:
            if sys.version_info.minor > 9 or sys.version_info.minor < 7: 
                raise RuntimeError(
                    "python version should be within py3.7->py3.9, but got: python{}.{}".format(*sys.version_info))
        else:
            raise RuntimeError("python 3 is required, but got python2")

        if platform.system() == "Windows":
            raise RuntimeError("Windows is not supported")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)    
            
        extpath = os.path.abspath(self.get_ext_fullpath(ext.name))
        
        bin_dir = os.path.abspath(os.path.join(self.build_temp, 'install'))
        lib_dir = os.path.abspath(os.path.join(self.build_temp, 'lib'))

        self.distribution.bin_dir = bin_dir
        self.distribution.lib_dir = lib_dir
        install_command = ["conda", "install", "cmake==3.18.2", "boost", "openblas", "gflags", "glog", "lmdb", "leveldb",
                            "h5py", "hdf5", "scikit-image", "protobuf==3.19.1", "six"]
        if subprocess.call(install_command) != 0:
            sys.exit(-1)
        
        python_version = 'python{}.{}'.format(*sys.version_info)
        cmake_command = ['cmake', '-DANACONDA_HOME='+os.environ['CONDA_PREFIX'], '-DPYTHON_VERSION='+python_version, "-B", self.build_temp]
        if subprocess.call(cmake_command) != 0:
            sys.exit(-1)            

        cmake_command = ['cmake', "--build", self.build_temp, '-j4']
        if subprocess.call(cmake_command) != 0:
            sys.exit(-1)      

        cmake_command = ['cmake', "--install", self.build_temp, '--prefix', bin_dir]
        if subprocess.call(cmake_command) != 0:
            sys.exit(-1)   

        print()  # Add an empty line for cleaner output

setup(
    name="brocolli-caffe",
    version="7.0.0",
    author="desmond",
    author_email="desmond.yao@buaa.edu.cn",
    description="official caffe, commit id 9b891540183ddc834a02b2bd81b31afae71b2153",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inisis/caffe",
    project_urls={
        "Bug Tracker": "https://github.com/inisis/caffe/issues",
    },
    cmdclass=dict(build_ext=CMakeBuild,
                  install_data=InstallCMakeLibsData,
                  install_lib=InstallCMakeLibs),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=[CMakeExtension('caffe._caffe', sourcedir='.')],    
    packages=['caffe', 'caffe.proto'],
    package_dir={'': 'python'},
    python_requires=">=3.7",
    keywords="machine-learning, caffe, framework",
    zip_safe = False
)
