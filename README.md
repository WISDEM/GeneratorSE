# DEPRECATED
------------

**THIS REPOSITORY IS DEPRECATED AND WILL BE ARCHIVED (READ-ONLY) IN NOVEMBER 2019.**

WISDEM has moved to a single, integrated repository at https://github.com/wisdem/wisdem

---------------
# GeneratorSE

GeneratorSE is a set of analytical tools for sizing variable speed wind turbine Generators. The analytical framework involves electromagnetic, structural, and basic thermal design that are integrated to provide the optimal generator design dimensions using conventional magnetic circuit laws. 

The tool is structured on OpenMDAO and mainly considers available torque, mechanical power, normal and shear stresses, material properties, and costs to optimize designs of variable-speed wind turbine generators by satisfying specific design criteria. 

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Documentation

See local documentation in the `docs`-directory or access the online version at <http://wisdem.github.io/GeneratorSE/>

## Installation

For detailed installation instructions of WISDEM modules see <https://github.com/WISDEM/WISDEM> or to install GeneratorSE by itself do:

    $ python setup.py install

## Run Unit Tests

To check if installation was successful try to import the package.

	$ python
	> import XXX

You may also run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

	$ python src/test/test_GeneratorSE.py

For software issues please use <https://github.com/WISDEM/GeneratorSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
