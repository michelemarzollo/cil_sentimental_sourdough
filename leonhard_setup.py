from setuptools import setup, find_packages
"""Setup module for project. To run when you enter leonhard cluster, in order to
install dependencies"""

setup(
        packages=find_packages(exclude=[]),
        python_requires='>=3.7',
        install_requires=[
                # Add external libraries here.
                'tensorflow-gpu==2.2.0',
                'numpy',
                'matplotlib',
                'sklearn',
                'pandas',
        ],
)
