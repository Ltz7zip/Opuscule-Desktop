# -*- coding: utf-8 -*-
"""fws_native —— Fantasy World Simulator C++ 原生加速扩展。

构建: /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 setup.py build_ext --inplace
"""
import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

# OpenMP (Apple clang 需 -Xpreprocessor)
# 运行时 libomp 必须与 torch 共用一个实例 (torch 自带 libomp.dylib),
# 否则 OMP Error #15 双 runtime 冲突段错误
LIBOMP = "/opt/homebrew/opt/libomp"
extra_compile_args = ["-O3", "-Wall", "-Wextra", "-Wno-unused-parameter",
                      "-mcpu=apple-m1"]
extra_link_args = []


def _torch_libomp():
    try:
        import torch
        p = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
        return p if os.path.exists(p) else None
    except Exception:
        return None


_omp_lib = _torch_libomp()
if _omp_lib:
    omp_dir = os.path.dirname(_omp_lib)
    extra_compile_args += ["-Xpreprocessor", "-fopenmp",
                           f"-I{LIBOMP}/include"]
    # -fopenmp 会隐式追加 -lomp; 把 torch 的 lib 目录放最前, 使 -lomp
    # 解析到 torch 自带 libomp.dylib (与 torch 共用同一 runtime, 避免
    # OMP Error #15 双 runtime 冲突)
    extra_link_args += [f"-L{omp_dir}",
                        f"-Wl,-rpath,{omp_dir}", "-lomp"]
elif os.path.isdir(os.path.join(LIBOMP, "include")):
    extra_compile_args += ["-Xpreprocessor", "-fopenmp",
                           f"-I{LIBOMP}/include"]
    extra_link_args += [f"-L{LIBOMP}/lib", "-lomp"]

ext_modules = [
    Pybind11Extension(
        "fws_native",
        sources=[
            os.path.join(SRC, "bindings.cpp"),
            os.path.join(SRC, "climate.cpp"),
            os.path.join(SRC, "hydro.cpp"),
            os.path.join(SRC, "hydro_adv.cpp"),
            os.path.join(SRC, "temperature.cpp"),
        ],
        include_dirs=[SRC],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="fws_native",
    version="0.1.0",
    packages=["fws_native"],  # 占位; 实际产物为内联 .so
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)