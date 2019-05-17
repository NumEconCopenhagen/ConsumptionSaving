#!/usr/bin/env python
import distutils
import os
import setuptools
import shlex
import shutil
import subprocess
import time


class BlackCommand(distutils.cmd.Command):
    """A custom command to format python code"""

    description = "run Black on Python source files"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        command = f"python -mblack setup.py {self.distribution.get_name()} tests"

        self.announce(f"Running command: {command}", level=distutils.log.INFO)
        subprocess.check_call(shlex.split(command))


class DocsCommand(distutils.cmd.Command):
    """A custom command to format python code"""

    description = "generate documentation"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        command = "make html"

        if os.path.exists("docs/_build") and os.path.isdir("docs/_build"):
            self.announce(f"Removing build directory", level=distutils.log.INFO)
            shutil.rmtree("docs/_build")

        self.announce(f"Running command: {command}", level=distutils.log.INFO)
        if os.name == "nt":
            subprocess.check_call(shlex.split(command), cwd="docs", shell=True)
        else:
            subprocess.check_call(shlex.split(command), cwd="docs")

setuptools.setup(
    name="consav",
    version="0.1.0",
    author="Jeppe Druedahl",
    packages=setuptools.find_packages(),
    cmdclass={"black": BlackCommand, "docs": DocsCommand},
)
