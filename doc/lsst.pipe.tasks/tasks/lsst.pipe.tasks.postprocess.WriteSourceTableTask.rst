.. lsst-task-topic:: lsst.pipe.tasks.postprocess.WriteSourceTableTask

####################
WriteSourceTableTask
####################

``WriteSourceTableTask`` converts table of sources measured on a calexp (dataset `src`) to a
parquet file. All data is copied without transformation, and column names are unchanged.

It is the first of three postprocessing tasks to convert a `src` table to a
per-visit Source Table that conforms to the standard data model. The second is
:doc:`lsst.pipe.tasks.postprocess.TransformSourceTableTask`, and the third is :doc:`lsst.pipe.tasks.postprocess.ConsolidateSourceTableTask`.

``WriteSourceTableTask`` is available as a :ref:`command-line task <lsst.pipe.tasks-command-line-tasks>`, :command:`writeSourceTable.py`.

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-summary:

Processing summary
==================


``WriteSourceTableTask`` reads in the `src` table, calls its `asAstropy` method to produce a DataFrame, and writes it out in parquet format.

.. lsst.pipe.tasks.postprocess.WriteSourceTableTask-cli:

writeSourceTable.py command-line interface
==========================================

.. code-block:: text

   writeSourceTable.py REPOPATH [@file [@file2 ...]] [--output OUTPUTREPO | --rerun RERUN] [--id] [other options]

Key arguments:

:option:`REPOPATH`
   The input Butler repository's URI or file path.

Key options:

:option:`--id`:
   The data IDs to process.

.. seealso::

   See :ref:`command-line-task-argument-reference` for details and additional options.

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.postprocess.WriteSourceTableTask

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-butler:

Butler datasets
===============

When run as the ``writeSourceTable.py`` command-line task, or directly through the `~lsst.pipe.tasks.postprocess.WriteSourceTableTask.runDataRef` method, ``WriteSourceTableTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.
Note that configurations for ``WriteSourceTableTask``, and its subtasks, affect what datasets are persisted and what their content is.

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-butler-inputs:

Input datasets
--------------

``src``
    Full depth source catalog (lsst.afw.table) produced by ProcessCcdTask

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-butler-outputs:

Output datasets
---------------

``source``
    Full depth source catalog (parquet)


.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.postprocess.WriteSourceTableTask

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.postprocess.WriteSourceTableTask

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-examples:

Examples
========

.. code-block:: bash

    writeSourceTable.py /datasets/hsc/repo  --calib /datasets/hsc/repo/CALIB --rerun <rerun name> --id visit=30504 ccd=0..8^10..103

.. _lsst.pipe.tasks.postprocess.WriteSourceTableTask-debug:
