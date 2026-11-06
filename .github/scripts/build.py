"""Configure the same hosted CPU/CUDA build for PR validation and SHA packages."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ARCHITECTURES = "75-virtual;80-virtual;86-real;89-real;90-virtual;120"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--shared", choices=("ON", "OFF"), default="ON")
    parser.add_argument("--sanitizer", choices=("ADDRESS", "UNDEFINED", "THREAD"))
    parser.add_argument("--cuda-architectures", default=ARCHITECTURES)
    args = parser.parse_args()
    cuda = args.backend == "cuda"
    options = ["cmake", "-S", ".", "-B", str(args.build), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=" + ("Debug" if args.sanitizer else "Release"),
               "-DBUILD_SHARED_LIBS=" + args.shared, "-DGGML_NATIVE=OFF",
               "-DGGML_BACKEND_DL=" + ("ON" if cuda and args.shared == "ON" else "OFF"),
               "-DLLAMA_FATAL_WARNINGS=ON", "-DGGML_FATAL_WARNINGS=ON",
               "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_APP=OFF",
               "-DLLAMA_BUILD_TESTS=" + ("OFF" if cuda else "ON"),
               "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_BUILD_UI=OFF",
               "-DGGML_CUDA=" + ("ON" if cuda else "OFF")]
    if sys.platform != "win32":
        options += ["-DCMAKE_C_COMPILER=gcc-13", "-DCMAKE_CXX_COMPILER=g++-13",
                    "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                    "-DCMAKE_INSTALL_RPATH=$ORIGIN"]
    if sys.platform == "win32":
        options.append("-DLLAMA_BUILD_BORINGSSL=ON")
    if cuda:
        root = Path(os.environ["CUDA_PATH"])
        options += ["-DCUDAToolkit_ROOT=" + str(root),
                    "-DCMAKE_CUDA_COMPILER=" + str(root / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")),
                    "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_architectures,
                    "-DGGML_CUDA_FA_QUANTS=all", "-DGGML_CUDA_CCCL_VERSION=v3.4.3"]
    if args.sanitizer:
        options += ["-DLLAMA_SANITIZE_" + args.sanitizer + "=ON", "-DGGML_SANITIZE_" + args.sanitizer + "=ON"]
    subprocess.run(options, check=True)
    subprocess.run(["cmake", "--build", str(args.build), "--parallel", "1" if cuda else "2"], check=True)


if __name__ == "__main__":
    main()
