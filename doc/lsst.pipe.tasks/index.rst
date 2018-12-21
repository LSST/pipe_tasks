.. _lsst.pipe.tasks:

###############
lsst.pipe.tasks
###############

``lsst.pipe.tasks`` provides many of the `~lsst.pipe.base.Task` classes that drive the LSST Science Pipelines.
The :ref:`command-line tasks <lsst.pipe.tasks-command-line-tasks>` listed here are useful data processing entry points for most users.
You can also assemble your own pipelines by combining individual tasks through their Python APIs.

``lsst.pipe.tasks`` does not provide all the tasks and command-line tasks in the LSST Science Pipelines.
For a complete overview of the available tasks, see the *Processing Data* documentation section (to be completed).
To learn more about the task framework in general, see the :ref:`lsst.pipe.base <lsst.pipe.base>` module documentation.

.. .. _lsst.pipe.tasks-using:

.. Using lsst.pipe.tasks
.. =====================

.. toctree linking to topics related to using the module's APIs.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.pipe.tasks-contributing:

Contributing
============

``lsst.pipe.tasks`` is developed at https://github.com/lsst/pipe_tasks.
You can find Jira issues for this module under the `pipe_tasks <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20pipe_tasks>`_ component.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.pipe.tasks-command-line-taskref:

Task reference
==============

.. _lsst.pipe.tasks-command-line-tasks:

Command-line tasks
------------------

.. lsst-cmdlinetasks::
   :root: lsst.pipe.tasks

.. _lsst.pipe.tasks-tasks:

Tasks
-----

.. lsst-tasks::
   :root: lsst.pipe.tasks
   :toctree: tasks

.. _lsst.pipe.tasks-configs:

Configurations
--------------

.. lsst-configs::
   :root: lsst.pipe.tasks
   :toctree: configs

.. _lsst.pipe.tasks-pyapi:

Python API reference
====================

.. automodapi:: lsst.pipe.tasks
