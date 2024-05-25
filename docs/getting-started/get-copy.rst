Working with PyKOALA and git
=============================

Getting a copy
--------------

To get PyKOALA, clone the repository, using git:

::
    
    git clone https://github.com/pykoala/koala.git


Alternatively, you can create your own :doc:`fork<../developer-guide/fork>`. We also describe the steps for developers in this :doc:`guide<../developer-guide/index>`

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

If you have not made changes in your local version, enter in the ``koala`` directory and use:

::

    git pull

Conversely, if you have made modifications that you do not wish to keep, use this instead:

::

    git stash 
    git pull
    
Finally, if you want to keep the changes, use this after the previous two commands:

::

    git stash pop


