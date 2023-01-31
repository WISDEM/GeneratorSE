import setuptools

metadata = dict(
    name="generatorse",
    url='https://github.com/WISDEM/GneeratorSE',
    version="0.0.1",
    description="Design model for DD-IPM, DD-LTS, and MS-IPM wind turbine generators",
    author="Latha Sethuraman, Garrett Barter and Pietro Bortolotti",
    packages=setuptools.find_packages(exclude=["test", "examples"]),
    python_requires=">=3.7",
    zip_safe=True,
    install_requires=['pyfemm','numpy','openmdao','pandas','nlopt'],
    )

setuptools.setup(**metadata)
