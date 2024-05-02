from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A package for implementing Chow Test trough a window function'

# Setting up
setup(
    name="completechowtest",
    version=VERSION,
    author="Hercroce (Marcos Hernan)",
    author_email="<hercroce@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['scipy', 'pandas', 'scikit-learn', 'numpy', 'statsmodels', 'matplotlib'],
    classifiers=[
        "Development Status :: 2 - Developing",
        "Intended Audience :: Economists, statisticians and data professionals",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

