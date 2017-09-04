Generator Systems Engineering (GeneratorSE)

GeneratorSE is a set of analytical tools for sizing variable speed wind turbine Generators. The analytical framework involves electromagnetic, structural, and basic thermal design that are integrated to provide the optimal generator design dimensions using conventional magnetic circuit laws. 
The tool is structured on OpenMDAO and mainly considers available torque, mechanical power, normal and shear stresses, material properties, and costs to optimize designs of variable-speed wind turbine generators by satisfying specific design criteria. 

Authors: [Latha Sethuraman and Katherine Dykes](nrel.wisdem+GeneratorSE@gmail.com)

## Version

This software is a beta version 0.1.1.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/GeneratorSE/>

## Prerequisites

General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

## Dependencies:

Wind Plant Framework: [FUSED-Wind](http://fusedwind.org) (Framework for Unified Systems Engineering and Design of Wind Plants)

Other interfaces: DriveSE

Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

## Installation

First, clone the [repository](https://github.com/WISDEM/GeneratorSE)
or download the releases and uncompress/unpack (GeneratorSE.py-|release|.tar.gz or GeneratorSE.py-|release|.zip) from the website link at the bottom the [GeneratorSE site](http://nwtc.nrel.gov/GeneratorSE).

Install GeneratorSE within an activated OpenMDAO environment:

	$ plugin install

It is not recommended to install the software outside of OpenMDAO.

## Run Unit Tests

To check if installation was successful try to import the module within an activated OpenMDAO environment.

	$ python
	> import XXX

You may also run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

	$ python src/test/test_GeneratorSE.py

For software issues please use <https://github.com/WISDEM/GeneratorSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
