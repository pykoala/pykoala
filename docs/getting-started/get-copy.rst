Working with PyKOALA and git
=============================

Getting a copy
--------------

To get PyKOALA, clone the repository, using git:

::
    
    git clone https://github.com/pykoala/koala.git


Setting username and email address
----------------------------------

Before pushing changes, make sure your name and email address are set as follows:

::

    cd koala
    git config --global user.name "Ben Kenobi"
    git config --global user.email "obi.wan@mq.edu.au"



Please use your full name in the specified format, as it will appear in the commit logs and the AUTHORS file.

Getting updates from your fork 
------------------------------

If the main repository has been updated, you may want to get those in your local version of the code. If so, your version will be in on of these cases:

- If you have not made changes in your local version, enter in the ``koala`` directory and use:

::

    git pull

- If you have made modifications that you do not wish to keep, use this instead:

::

    git stash 
    git pull
    
- In the case you want to keep the changes, type:

::

    git stash 
    git pull
    git stash pop


