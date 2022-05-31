---------------
# GeneratorSE

GeneratorSE is a set of numerical tools for sizing two technologies of variable speed wind turbine generators:
	- Interior permanent magnet synchronous generators (PMSG)
	- Low temperature superconducting generators (LTS)

This repository contains the codes based on the open-source library pyFEMM. The purely analytical tools to size PMSG machines have instead been integrated within [WISDEM](https://github.com/wisdem/wisdem)

The codes adopt the optimization library [OpenMDAO](https://openmdao.org) and mainly considers available torque, mechanical power, normal and shear stresses, material properties, and costs to optimize designs by satisfying specific design criteria. 

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Documentation

See local documentation in the `docs`-directory or access the online version at <http://wisdem.github.io/GeneratorSE/>

## Installation

To run the code, follow these steps

1. Download and install pyfemm from https://www.femm.info/wiki/pyFEMM

2. In an anaconda environment, type 

        pip install pyfemm
        conda install openmdao

3. In your preferred folder, clone and compile the repository

		git clone git@github.com:WISDEM/GeneratorSE.git # or git clone https://github.com/WISDEM/GeneratorSE.git
		pip install -e .


For functionality, theory, or software issues please use <https://github.com/WISDEM/GeneratorSE/issues>

## Acknowledgments

The technical support of David Meeker, Ph.D., author of the Femm library is gratefully acknowledged