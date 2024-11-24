import os
import sys
import subprocess
import platform
from skbuild import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # Define paths
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={os.path.abspath(sys.executable)}",
        ]
        
        # Add build-specific arguments
        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=" + ext_dir]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=Release"]

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        # Run CMake
        subprocess.check_call(["cmake", os.path.abspath(".")] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)

def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="SciencePy",
    version="0.1.0",
    description="A Python library for rendering triangles.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Benjamin Chislett",
    author_email="chislett.ben@gmail.com",
    url="https://github.com/benchislett/Science",
    license="MIT",
    packages=[""],
    cmake_install_dir="SciencePy",
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
        ext_modules=[Extension("SciencePy", sources=[])],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
)
