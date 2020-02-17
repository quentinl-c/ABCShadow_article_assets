# # coding: utf-8
#
# Copyright (c) 2019 quentinl-c.
#
# This file is part of ABCShadow 
# (see https://github.com/quentinl-c/ABCShadow_article_assets).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from distutils.core import setup

import numpy

import Cython.Compiler.Options
from Cython.Build import cythonize

Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
Cython.Compiler.Options.get_directive_defaults()['binding'] = True
setup(name='ABC Shadow',
      ext_modules=cythonize([
        "./abc_shadow/*.pyx", "./abc_shadow/graph/graph_wrapper.pyx"],
                            annotate=True),
      include_dirs=[numpy.get_include()])
