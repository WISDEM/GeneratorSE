.. _documentation-label:

.. currentmodule:: GeneratorSE

Documentation
--------------

.. only:: latex

    An HTML version of this documentation is available which is better formatted for reading the code documentation and contains hyperlinks to the source code.


Sizing models for the different generator modules are described along with mass, cost and efficiency as objective functions .

Documentation for GeneratorSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for GeneratorSE :

.. literalinclude:: ../src/generatorse/drive.py
    :language: python
    :start-after: Drive3pt(Assembly)
    :end-before: def configure(self)
    :prepend: class Drive3pt(Assembly):

.. module:: drivese.drive
.. class:: Drive3pt

The following inputs and outputs are defined for GeneratorSE using the four-point suspension configuration:

.. literalinclude:: ../src/generatorse/drive.py
    :language: python
    :start-after: Drive4pt(Assembly)
    :end-before: def configure(self)
    :prepend: class Drive4pt(Assembly):

.. module:: drivese.drive
.. class:: Drive4pt

Implemented Base Model
=========================
.. module:: drivewpact.drive
.. class:: NacelleBase

Referenced Sub-System Modules 
==============================
.. module:: drivese.drivese_components
.. class:: LowSpeedShaft_drive
.. class:: LowSpeedShaft_drive4pt
.. class:: LowSpeedShaft_drive3pt
.. class:: MainBearing_drive
.. class:: SecondBearing_drive
.. class:: Gearbox_drive
.. class:: HighSpeedSide_drive
.. class:: Generator_drive
.. class:: Bedplate_drive
.. class:: AboveYawMassAdder_drive
.. class:: YawSystem_drive
.. class:: NacelleSystemAdder_drive





