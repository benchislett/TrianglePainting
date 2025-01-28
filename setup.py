import glob
import os
import platform

from setuptools import find_packages, setup
from setuptools.dist import Distribution

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def get_version() -> str:
    version_path = os.path.join(PROJECT_DIR, "polypaint", "version.py")
    if not os.path.exists(version_path) or not os.path.isfile(version_path):
        msg = f"Version file not found: {version_path}"
        raise RuntimeError(msg)
    with open(version_path) as f:
        code = compile(f.read(), version_path, "exec")
    loc = {"__file__": version_path}
    exec(code, loc)
    if "__version__" not in loc:
        msg = "Version info is not found in polypaint/version.py"
        raise RuntimeError(msg)
    return loc["__version__"]


def parse_requirements(filename: os.PathLike) -> list[str]:
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != "-", line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith(("#", "-r")):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            else:
                deps.append(line)
    return deps, extra_URLs


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self) -> bool:
        """Return True for binary distribution."""
        return True

    def is_pure(self) -> bool:
        """Return False for binary distribution."""
        return False


def get_polypaint_lib() -> str:
    if platform.system() == "Windows":
        lib_glob = "polypaint.*.pyd"
    else:
        lib_glob = "polypaint.*.so"
    lib_glob = os.path.join(PROJECT_DIR, "polypaint", lib_glob)

    lib_paths = glob.glob(lib_glob)
    if len(lib_paths) == 0 or not os.path.isfile(lib_paths[0]):
        msg = (
            "Cannot find polypaint bindings library. Please build the library first. Search path: "
            f"{lib_glob}"
        )
        raise RuntimeError(msg)
    if len(lib_paths) > 1:
        msg = (
            f"Found multiple polypaint bindings libraries: {lib_paths}. "
            "Please remove the extra ones."
        )
        raise RuntimeError(msg)

    return lib_paths[0]


def main() -> None:
    polypaint_lib_path = get_polypaint_lib()

    setup(
        name="polypaint",
        version=get_version(),
        author={"name": "Benjamin Chislett", "email": "chislett.ben@gmail.com"},
        description="Efficient, Flexible and Portable Structured Generation",
        long_description=open(os.path.join(PROJECT_DIR, "README.md")).read(),
        long_description_content_type="text/markdown",
        licence="MIT",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: C++",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX",
        ],
        keywords="polygon geometry",
        packages=find_packages(),
        package_data={"polypaint": [polypaint_lib_path]},
        zip_safe=False,
        install_requires=parse_requirements("requirements.txt")[0],
        python_requires=">=3.10, <4",
        url="https://github.com/benchislett/TrianglePainting",
        distclass=BinaryDistribution,
    )

main()