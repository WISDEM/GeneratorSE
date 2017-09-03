.. _tutorial-label:

.. currentmodule:: masstocost.docs.source.examples.example


Tutorial
--------

As an example, let us size a direct-drive Permanent Magnet synchronous Generator (PMSG) for the NREL 5MW Reference Model :cite:`FAST2009`.  

The first step is to import the relevant files.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

We will start with the radial flux PMSG with spoke arm arrangement.  The PMSG module relies on inputs from the rotor such as rated speed, 
power rating and rated torque.  It also requires specification of shear stress, material properties( densities), specific costs and target design efficiency.
and initialization of electromagnetic and structural design variables necessary to calculate basic design. Specification of the optimisation 
objective(Costs,Mass, Efficiency or Aspect ratio) and driver determines the final design. The designs are generated in compliance with the user-specified constraints on generator 
terminal voltage and constraints imposed on the dimensions and electrical,magnetic and structural deformations. The excitation requirement (e.g., magnet dimensions,pole pairs) 
is determined in accordance with the required voltage at no-load. 

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


We now run the PMSG_arms module.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


The resulting system and component properties can then be printed.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The results should appear as below:

>>>                               Parameters       Values       Limit      Units
>>>0                                  Rating     5.000000                     MW
>>>1                             Stator Arms     5.000000                   unit
>>>2              Stator Axial arm dimension   308.694191                     mm
>>>3    Stator Circumferential arm dimension   484.042558     540.354         mm
>>>4                    Stator arm Thickness    65.128033                     mm
>>>5                              Rotor arms     5.000000                     mm
>>>6               Rotor Axial arm dimension   700.586761                     mm
>>>7     Rotor Circumferential arm dimension   519.687520     540.354				 mm
>>>8                     Rotor arm Thickness    64.784072                     mm
>>>9                Stator Radial deflection     0.295831    0.338212         mm
>>>10                Stator Axial deflection     1.200257     32.1335         mm
>>>11               Stator circum deflection     2.762541     2.95146         mm
>>>12                Rotor Radial deflection     0.289673    0.321078         mm
>>>13                 Rotor Axial deflection     0.205148     32.1335         mm
>>>14                Rotor circum deflection     2.518227     2.80194         mm
>>>15                       Air gap diameter     6.546696                      m
>>>16                 Overall Outer diameter     6.769071                      m
>>>17                          Stator length     1.606676                      m
>>>18                              l/d ratio     0.245418  (0.2-0.27)
>>>19                      Slot_aspect_ratio     4.312481      (4-10)
>>>20                             Pole pitch    92.859636                     mm
>>>21                     Stator slot height    59.929070                     mm
>>>22                       Stator slotwidth    13.896657                     mm
>>>23                     Stator tooth width    16.984803                     mm
>>>24                     Stator yoke height    97.687186                     mm
>>>25                      Rotor yoke height    89.961327                     mm
>>>26                          Magnet height    11.035806                     mm
>>>27                           Magnet width    65.001745                     mm
>>>28  Peak air gap flux density fundamental     0.831741                      T
>>>29          Peak stator yoke flux density     0.308723          <2          T
>>>30           Peak rotor yoke flux density     0.301712          <2          T
>>>31              Flux density above magnet     0.733157          <2          T
>>>32            Maximum Stator flux density     0.096619    0.831741          T
>>>33             Maximum tooth flux density     1.512256                      T
>>>34                             Pole pairs   111.000000                      -
>>>35             Generator output frequency    22.385000                     Hz
>>>36         Generator output phase voltage  1853.819124                      V
>>>37         Generator Output phase current   926.443803        >500          A
>>>38                      Stator resistance     0.089215              ohm/phase
>>>39                 Synchronous inductance     0.009194                    p.u
>>>40                           Stator slots   666.000000                 A/mm^2
>>>41                           Stator turns   222.000000                  slots
>>>42                Conductor cross-section   248.082396           5      turns
>>>43                Stator Current density      3.734420         3-6       mm^2
>>>44               Specific current loading    60.000000          60       kA/m
>>>45                  Generator Efficiency     93.000545        >93%          %
>>>46                              Iron mass    57.805067                   tons
>>>47                            Magnet mass     2.161781                   tons
>>>48                            Copper mass     5.817571                   tons
>>>49                           Mass of Arms    34.841671                   tons
>>>50                             Total Mass   100.626089                   tons
>>>51                            Stator Mass    60.522128                    ton
>>>52                             Rotor Mass    40.103961                    ton
>>>53                    Total Material Cost   282.820957                     k$


Secondly, we will design a gear driven Doubly fed Induction generator.  The DFIG design 
relies on inputs such as machine rating, target overall drivetrain efficiency, gearbox efficiency,
high speed shaft speed( or gear ratio).It also requires specification of shear stress, material properties( densities), specific material costs. 
The main design variables are initialized with an objective and driver. The designs are computed analytically and checked against predefined constraints 
to meet objective functions.The optimized design dimensions are printed on screen and available in an output file in a Microsoft Excel format.

.. literalinclude:: examples/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

We now instantiate the DFIG object which automatically updates the mass, costs, efficiency and performance variables based on the supplied inputs.  
In addition, calculations of mass properties are also made.

.. literalinclude:: examples/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


The resulting system and component properties can then be printed.

.. literalinclude:: examples/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---

The results should appear as below:

>>> Estimate of Nacelle Component Sizes for the NREL 5 MW Reference Turbine
>>> Low speed shaft:  18564.9 kg
>>> Main bearings:   5845.4 kg
>>> Gearbox:  55658.3 kg
>>> High speed shaft & brakes:   2414.7 kg
>>> Generator:  16699.9 kg
>>> Variable speed electronics:      0.0 kg
>>> Overall mainframe: 60785.2 kg
>>>      Bedplate:  51364.7 kg
>>> Electrical connections:      0.0 kg
>>> HVAC system:    400.0 kg
>>> Nacelle cover:   4577.4 kg
>>> Yaw system:   6044.7 kg
>>> Overall nacelle: 170990.5 kg
>>>     cm  -0.44   0.00   0.26 [m, m, m]
>>>     I 77320965.7 1033725.7 935773.7 [kg*m^2, kg*m^2, kg*m^2]

