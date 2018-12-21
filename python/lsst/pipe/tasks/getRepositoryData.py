#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""Retrieve collections of metadata or data based on a set of data references

Use this as a base task for creating graphs and reports for a set of data.
"""
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ["DataRefListRunner", "GetRepositoryDataTask"]


class DataRefListRunner(pipeBase.TaskRunner):
    """A task runner that calls run with a list of data references

    Differs from the default TaskRunner by providing all data references at once,
    instead of iterating over them one at a time.
    """
    @staticmethod
    def getTargetList(parsedCmd):
        """Return a list of targets (arguments for __call__); one entry per invocation
        """
        return [parsedCmd.id.refList]  # one argument consisting of a list of dataRefs

    def __call__(self, dataRefList):
        """Run GetRepositoryDataTask.run on a single target

        Parameters
        ----------
        dataRefList :
            argument dict for run; contains one key: dataRefList

        Returns
        -------
        None
            if doReturnResults false
        Struct :
            - ``dataRefList`` : the argument dict sent to runDataRef
            - ``metadata`` : task metadata after execution of runDataRef
            - ``result`` : result returned by task runDataRef
        """
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.runDataRef(dataRefList)

        if self.doReturnResults:
            return pipeBase.Struct(
                dataRefList=dataRefList,
                metadata=task.metadata,
                result=result,
            )


class GetRepositoryDataTask(pipeBase.CmdLineTask):
    """Retrieve data from a repository, e.g. for plotting or analysis purposes
    """
    ConfigClass = pexConfig.Config  # nothing to configure
    RunnerClass = DataRefListRunner
    _DefaultName = "getTaskData"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def runDataRef(self, dataRefList):
        """Get data from a repository for a collection of data references

        Parameters
        ----------
        dataRefList : a list of data references
        """
        raise NotImplementedError("subclass must specify a run method")

    def getIdList(self, dataRefList):
        """Get a list of data IDs in a form that can be used as dictionary keys

        Parameters
        ----------
        dataRefList :
            a list of data references

        Returns
        -------
        result : ``lsst.pipe.base.Struct``
            - ``idKeyTuple`` : a tuple of dataRef data ID keys
            - ``idValList`` : a list of data ID value tuples,
                each tuple contains values in the order in idKeyTuple
        """
        if not dataRefList:
            raise RuntimeError("No data refs")
        idKeyTuple = tuple(sorted(dataRefList[0].dataId.keys()))

        idValList = []
        for dataRef in dataRefList:
            idValTuple = tuple(dataRef.dataId[key] for key in idKeyTuple)
            idValList.append(idValTuple)

        return pipeBase.Struct(
            idKeyTuple=idKeyTuple,
            idValList=idValList,
        )

    def getDataList(self, dataRefList, datasetType):
        """Retrieve a list of data

        Parameters
        ----------
        dataRefList :
            a list of data references
        datasetType :
            datasetType of data to be retrieved

        Returns
        -------
        result : `list`
            a list of data, one entry per dataRef in dataRefList (in order)
        """
        return [dataRef.get(datasetType=datasetType) for dataRef in dataRefList]

    def getMetadataItems(self, dataRefList, datasetType, nameList):
        """Retrieve a list of dictionaries of metadata

        Parameters
        ----------
        dataRefList: a list of data references
        datasetType: datasetType of metadata (or any object that supports get(name))

        Returns
        -------
        valList : `list` of `dict`
            a list of dicts of metadata:
            - each entry in the list corresponds to a dataRef in dataRefList
            - each dict contains name: item of metadata, for each name in nameList;
            numeric and string values will be returned as arrays
        """
        valList = []
        for dataRef in dataRefList:
            metadata = dataRef.get(datasetType=datasetType)
            valList.append(dict((name, metadata.getArray(name)) for name in nameList))
        return valList
