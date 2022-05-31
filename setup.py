import setuptools

metadata = dict(
    name="lts",
    url='https://github.com/WISDEM/LTS',
    version="0.0.1",
    description="Design model for low-temperature superconducting generators",
    author="Latha Sethuraman",
    packages=setuptools.find_packages(exclude=["test", "examples"]),
    python_requires=">=3.7",
    zip_safe=True,
    install_requires=['pyfemm','numpy','openmdao','pandas','nlopt'],
    )

setuptools.setup(**metadata)
