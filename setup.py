import setuptools
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import os
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
cwd = os.getcwd()

cython_module = cythonize(Extension(
                           "maxflow._maxflow",                                
                           sources=[os.path.join(cwd, 'maxflow', '_maxflow.pyx')], 
                           language="c++",                        
                           extra_compile_args = ["-O3", "-std=c++17", "-fopenmp", "-w"],
                           extra_link_args = ["-fopenmp"],
                           ),
                            compiler_directives={'language_level' : "3"},
                          )

setup (name = 'maxflow',
        version = '0.0.2',
        author = 'Touqir Sajed',
        author_email = 'shuhash6@gmail.com',
        description = 'MaxFlow - a high performance Python library for computing the maximum flow of a graph',
        license = 'MIT',
        url = 'https://github.com/touqir14/MaxFlow',
        packages = ['maxflow'],
        ext_modules = cython_module,
        install_requires=['scipy>=0.17.0', 'numpy>=1.13.0'],
        )