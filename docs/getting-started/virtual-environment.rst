Running PyKOALA in a Python virtual environment
===============================================

What is a virtual environment?
------------------------------

A Python virtual environment is an isolated setup containing a specific Python interpreter and its dependencies, separate from the system’s global Python setup. It enables projects to have their own set of dependencies without interfering with other projects or the system-wide Python installation.
**It enables projects to have their own set of dependencies without interfering with other projects or the system-wide Python installation**. Python environments, like virtual environments or Docker, offer these key benefits:

1. **Isolation**: Keeps dependencies separate, avoiding conflicts.
2. **Dependency Management**: Easily manage dependencies for each project.
3. **Reproducibility**: Ensures consistent environments for development and deployment.
4. **Portability**: Makes projects easily shareable and deployable on different systems.
5. **Conflict Resolution**: Allows experimentation with different package versions.
6. **Security**: Enhances security by limiting system-level access.

In essence, they provide a structured and controlled way to manage dependencies, leading to more reliable projects.


Installation
------------

From the terminal, enter in the PyKOALA directory and type:

::

    python3 -m venv venv_koala

This will create named ``venv_koala`` with the neccesary files to activate and use the environmet. To activate it, use:

::

    source venv_koala/bin/activate

Then install all required packages with:

::

    pip install -r requirements.txt ; pip install . 

The second command will also install PyKOALA in the virtual environment.