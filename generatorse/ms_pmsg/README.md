---------------
# MS-PMSG

MS-PMSG is a set of codes for sizing medium speed permanent magnet synchronous generators:

This folder contains the codes wrtten to estimate basic generator dimensions, mass and costs based on both analytical models as well as performance predictions

based on FEA using the open-source library pyFEMM available through a user-defined flag. These tools have been integrated within [WISDEM](https://github.com/wisdem/wisdem)

The codes adopt the optimization library [OpenMDAO](https://openmdao.org) and mainly considers available torque, mechanical power, normal and shear stresses, material properties, and costs to optimize designs by satisfying specific design criteria. 

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Documentation

The WISDEM team is working on a relevant publication. The link to it will be provided here.

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



