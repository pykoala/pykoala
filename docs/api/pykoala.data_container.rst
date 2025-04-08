.. automodule:: pykoala.data_container
   :members: pykoala.data_container
   :undoc-members:
   :show-inheritance:

DataContainer
-------------
.. autoclass:: pykoala.data_container.DataContainer
   :members:
   :undoc-members:
   :show-inheritance:

SpectraContainer
----------------
.. autoclass:: pykoala.data_container.SpectraContainer
   :members:
   :undoc-members:
   :show-inheritance:

RSS
^^^
.. autoclass:: pykoala.data_container.RSS
   :members:
   :undoc-members:
   :show-inheritance:

Cube
^^^^
.. autoclass:: pykoala.data_container.Cube
   :members:
   :undoc-members:
   :show-inheritance:

Special properties
------------------

All :class:`DataContainer` objects have two important attributes.

- :class:`DataContainerHistory` tracks the data reduction processes storing all the relevant information that should be preserved in the final version.
- :class:`DataMask` stores flagged pixels/fibres that can be used for multiple purposes.

Below you can find documentation and details about their properties and methods.


.. autoclass:: pykoala.data_container.DataContainerHistory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pykoala.data_container.HistoryRecord
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pykoala.data_container.DataMask
   :members:
   :undoc-members:
   :show-inheritance:
