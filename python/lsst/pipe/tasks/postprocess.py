# This file is part of pipe_tasks
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools
import pandas as pd
from collections import defaultdict

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import CmdLineTask, ArgumentParser
from lsst.coadd.utils.coaddDataIdContainer import CoaddDataIdContainer

from .parquetTable import ParquetTable
from .multiBandUtils import makeMergeArgumentParser, MergeSourcesRunner
from .functors import CompositeFunctor, RAColumn, DecColumn, Column


def flattenFilters(df, filterDict, noDupCols=['coord_ra', 'coord_dec'], camelCase=False):
    """Flattens a dataframe with multilevel column index
    """
    newDf = pd.DataFrame()
    for filt, filtShort in filterDict.items():
        subdf = df[filt]
        columnFormat = '{0}{1}' if camelCase else '{0}_{1}'
        newColumns = {c: columnFormat.format(filtShort, c)
                      for c in subdf.columns if c not in noDupCols}
        cols = list(newColumns.keys())
        newDf = pd.concat([newDf, subdf[cols].rename(columns=newColumns)], axis=1)

    newDf = pd.concat([subdf[noDupCols], newDf], axis=1)
    return newDf


class WriteObjectTableConfig(pexConfig.Config):
    priorityList = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="Priority-ordered list of bands for the merge."
    )
    engine = pexConfig.Field(
        dtype=str,
        default="pyarrow",
        doc="Parquet engine for writing (pyarrow or fastparquet)"
    )
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class WriteObjectTableTask(CmdLineTask):
    """Write filter-merged source tables to parquet
    """
    _DefaultName = "writeObjectTable"
    ConfigClass = WriteObjectTableConfig
    RunnerClass = MergeSourcesRunner

    # Names of table datasets to be merged
    inputDatasets = ('forced_src', 'meas', 'ref')

    # Tag of output dataset written by `MergeSourcesTask.write`
    outputDataset = 'obj'

    def __init__(self, butler=None, schema=None, **kwargs):
        # It is a shame that this class can't use the default init for CmdLineTask
        # But to do so would require its own special task runner, which is many
        # more lines of specialization, so this is how it is for now
        CmdLineTask.__init__(self, **kwargs)

    def runDataRef(self, patchRefList):
        """!
        @brief Merge coadd sources from multiple bands. Calls @ref `run` which must be defined in
        subclasses that inherit from MergeSourcesTask.
        @param[in] patchRefList list of data references for each filter
        """
        catalogs = dict(self.readCatalog(patchRef) for patchRef in patchRefList)
        dataId = patchRefList[0].dataId
        mergedCatalog = self.run(catalogs, tract=dataId['tract'], patch=dataId['patch'])
        self.write(patchRefList[0], mergedCatalog)

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser.

        We will use the ArgumentParser to get a list of data
        references for patches; the RunnerClass will sort them into lists
        of data references for the same patch.

        References first of self.inputDatasets, rather than
        self.inputDataset
        """
        return makeMergeArgumentParser(cls._DefaultName, cls.inputDatasets[0])

    def readCatalog(self, patchRef):
        """Read input catalogs

        Read all the input datasets given by the 'inputDatasets'
        attribute.

        Parameters
        ----------
        patchRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for patch

        Returns
        -------
        Tuple consisting of filter name and a dict of catalogs, keyed by
        dataset name
        """
        filterName = patchRef.dataId["filter"]
        catalogDict = {}
        for dataset in self.inputDatasets:
            catalog = patchRef.get(self.config.coaddName + "Coadd_" + dataset, immediate=True)
            self.log.info("Read %d sources from %s for filter %s: %s" %
                          (len(catalog), dataset, filterName, patchRef.dataId))
            catalogDict[dataset] = catalog
        return filterName, catalogDict

    def run(self, catalogs, tract, patch):
        """Merge multiple catalogs.

        Parameters
        ----------
        catalogs : `dict`
            Mapping from filter names to dict of catalogs.
        tract : int
            tractId to use for the tractId column
        patch : str
            patchId to use for the patchId column

        Returns
        -------
        catalog : `lsst.pipe.tasks.parquetTable.ParquetTable`
            Merged dataframe, with each column prefixed by
            `filter_tag(filt)`, wrapped in the parquet writer shim class.
        """

        dfs = []
        for filt, tableDict in catalogs.items():
            for dataset, table in tableDict.items():
                # Convert afwTable to pandas DataFrame
                df = table.asAstropy().to_pandas().set_index('id', drop=True)

                # Sort columns by name, to ensure matching schema among patches
                df = df.reindex(sorted(df.columns), axis=1)
                df['tractId'] = tract
                df['patchId'] = patch

                # Make columns a 3-level MultiIndex
                df.columns = pd.MultiIndex.from_tuples([(dataset, filt, c) for c in df.columns],
                                                       names=('dataset', 'filter', 'column'))
                dfs.append(df)

        catalog = functools.reduce(lambda d1, d2: d1.join(d2), dfs)
        return ParquetTable(dataFrame=catalog)

    def write(self, patchRef, catalog):
        """Write the output.

        Parameters
        ----------
        catalog : `ParquetTable`
            Catalog to write
        patchRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for patch
        """
        patchRef.put(catalog, self.config.coaddName + "Coadd_" + self.outputDataset)
        # since the filter isn't actually part of the data ID for the dataset we're saving,
        # it's confusing to see it in the log message, even if the butler simply ignores it.
        mergeDataId = patchRef.dataId.copy()
        del mergeDataId["filter"]
        self.log.info("Wrote merged catalog: %s" % (mergeDataId,))

    def writeMetadata(self, dataRefList):
        """No metadata to write, and not sure how to write it for a list of dataRefs.
        """
        pass


class WriteSourceTableConfig(pexConfig.Config):
    doApplyExternalPhotoCalib = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Add local photoCalib columns from the calexp.photoCalib? Should only set True if "
             "generating Source Tables from older src tables which do not already have local calib columns")
    )
    doApplyExternalSkyWcs = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Add local WCS columns from the calexp.wcs? Should only set True if "
             "generating Source Tables from older src tables which do not already have local calib columns")
    )


class WriteSourceTableTask(CmdLineTask):
    """Write source table to parquet
    """
    _DefaultName = "writeSourceTable"
    ConfigClass = WriteSourceTableConfig

    def runDataRef(self, dataRef):
        src = dataRef.get('src')
        if self.config.doApplyExternalPhotoCalib or self.config.doApplyExternalSkyWcs:
            src = self.addCalibColumns(src, dataRef)

        ccdVisitId = dataRef.get('ccdExposureId')
        result = self.run(src, ccdVisitId=ccdVisitId)
        dataRef.put(result.table, 'source')

    def run(self, catalog, ccdVisitId=None):
        """Convert `src` catalog to parquet

        Parameters
        ----------
        catalog: `afwTable.SourceCatalog`
            catalog to be converted
        ccdVisitId: `int`
            ccdVisitId to be added as a column

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``table``
                `ParquetTable` version of the input catalog
        """
        self.log.info("Generating parquet table from src catalog")
        df = catalog.asAstropy().to_pandas().set_index('id', drop=True)
        df['ccdVisitId'] = ccdVisitId
        return pipeBase.Struct(table=ParquetTable(dataFrame=df))

    def addCalibColumns(self, catalog, dataRef):
        """Add columns with local calibration evaluated at each centroid

        for backwards compatibility with old repos.
        This exists for the purpose of converting old src catalogs
        (which don't have the expected local calib columns) to Source Tables.

        Parameters
        ----------
        catalog: `afwTable.SourceCatalog`
            catalog to which calib columns will be added
        dataRef: `lsst.daf.persistence.ButlerDataRef
            for fetching the calibs from disk.

        Returns
        -------
        newCat:  `afwTable.SourceCatalog`
            Source Catalog with requested local calib columns
        """
        measureConfig = SingleFrameMeasurementTask.ConfigClass()
        measureConfig.doReplaceWithNoise = False

        # Just need the WCS or the PhotoCalib attached to an exposue
        exposure = dataRef.get('calexp_sub',
                               bbox=lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(0, 0)))

        aliasMap = catalog.schema.getAliasMap()
        mapper = afwTable.SchemaMapper(catalog.schema)
        mapper.addMinimalSchema(catalog.schema, True)
        schema = mapper.getOutputSchema()

        exposureIdInfo = dataRef.get("expIdInfo")
        measureConfig.plugins.names = []
        if self.config.doApplyExternalSkyWcs:
            plugin = 'base_LocalWcs'
            if plugin in schema:
                raise RuntimeError(f"{plugin} already in src catalog. Set doApplyExternalSkyWcs=False")
            else:
                measureConfig.plugins.names.add(plugin)

        if self.config.doApplyExternalPhotoCalib:
            plugin = 'base_LocalPhotoCalib'
            if plugin in schema:
                raise RuntimeError(f"{plugin} already in src catalog. Set doApplyExternalPhotoCalib=False")
            else:
                measureConfig.plugins.names.add(plugin)

        measurement = SingleFrameMeasurementTask(config=measureConfig, schema=schema)
        schema.setAliasMap(aliasMap)
        newCat = afwTable.SourceCatalog(schema)
        newCat.extend(catalog, mapper=mapper)
        measurement.run(measCat=newCat, exposure=exposure, exposureId=exposureIdInfo.expId)
        return newCat

    def writeMetadata(self, dataRef):
        """No metadata to write.
        """
        pass

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", 'src',
                               help="data ID, e.g. --id visit=12345 ccd=0")
        return parser


class PostprocessAnalysis(object):
    """Calculate columns from ParquetTable

    This object manages and organizes an arbitrary set of computations
    on a catalog.  The catalog is defined by a
    `lsst.pipe.tasks.parquetTable.ParquetTable` object (or list thereof), such as a
    `deepCoadd_obj` dataset, and the computations are defined by a collection
    of `lsst.pipe.tasks.functor.Functor` objects (or, equivalently,
    a `CompositeFunctor`).

    After the object is initialized, accessing the `.df` attribute (which
    holds the `pandas.DataFrame` containing the results of the calculations) triggers
    computation of said dataframe.

    One of the conveniences of using this object is the ability to define a desired common
    filter for all functors.  This enables the same functor collection to be passed to
    several different `PostprocessAnalysis` objects without having to change the original
    functor collection, since the `filt` keyword argument of this object triggers an
    overwrite of the `filt` property for all functors in the collection.

    This object also allows a list of flags to be passed, and defines a set of default
    flags that are always included even if not requested.

    If a list of `ParquetTable` object is passed, rather than a single one, then the
    calculations will be mapped over all the input catalogs.  In principle, it should
    be straightforward to parallelize this activity, but initial tests have failed
    (see TODO in code comments).

    Parameters
    ----------
    parq : `lsst.pipe.tasks.ParquetTable` (or list of such)
        Source catalog(s) for computation

    functors : `list`, `dict`, or `lsst.pipe.tasks.functors.CompositeFunctor`
        Computations to do (functors that act on `parq`).
        If a dict, the output
        DataFrame will have columns keyed accordingly.
        If a list, the column keys will come from the
        `.shortname` attribute of each functor.

    filt : `str` (optional)
        Filter in which to calculate.  If provided,
        this will overwrite any existing `.filt` attribute
        of the provided functors.

    flags : `list` (optional)
        List of flags to include in output table.
    """
    _defaultFlags = ('calib_psf_used', 'detect_isPrimary')
    _defaultFuncs = (('coord_ra', RAColumn()),
                     ('coord_dec', DecColumn()))

    def __init__(self, parq, functors, filt=None, flags=None):
        self.parq = parq
        self.functors = functors

        self.filt = filt
        self.flags = list(self._defaultFlags)
        if flags is not None:
            self.flags += list(flags)

        self._df = None

    @property
    def defaultFuncs(self):
        funcs = dict(self._defaultFuncs)
        return funcs

    @property
    def func(self):
        additionalFuncs = self.defaultFuncs
        additionalFuncs.update({flag: Column(flag) for flag in self.flags})

        if isinstance(self.functors, CompositeFunctor):
            func = self.functors
        else:
            func = CompositeFunctor(self.functors)

        func.funcDict.update(additionalFuncs)
        func.filt = self.filt

        return func

    @property
    def noDupCols(self):
        return [name for name, func in self.func.funcDict.items() if func.noDup or func.dataset == 'ref']

    @property
    def df(self):
        if self._df is None:
            self.compute()
        return self._df

    def compute(self, dropna=False, pool=None):
        # map over multiple parquet tables
        if type(self.parq) in (list, tuple):
            if pool is None:
                dflist = [self.func(parq, dropna=dropna) for parq in self.parq]
            else:
                # TODO: Figure out why this doesn't work (pyarrow pickling issues?)
                dflist = pool.map(functools.partial(self.func, dropna=dropna), self.parq)
            self._df = pd.concat(dflist)
        else:
            self._df = self.func(self.parq, dropna=dropna)

        return self._df


class TransformCatalogBaseConfig(pexConfig.Config):
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )
    functorFile = pexConfig.Field(
        dtype=str,
        doc='Path to YAML file specifying functors to be computed',
        default=None
    )


class TransformCatalogBaseTask(CmdLineTask):
    """Base class for transforming/standardizing a catalog

    by applying functors that convert units and apply calibrations.
    The purpose of this task is to perform a set of computations on
    an input `ParquetTable` dataset (such as `deepCoadd_obj`) and write the
    results to a new dataset (which needs to be declared in an `outputDataset`
    attribute).

    The calculations to be performed are defined in a YAML file that specifies
    a set of functors to be computed, provided as
    a `--functorFile` config parameter.  An example of such a YAML file
    is the following:

        funcs:
            psfMag:
                functor: Mag
                args:
                    - base_PsfFlux
                filt: HSC-G
                dataset: meas
            cmodel_magDiff:
                functor: MagDiff
                args:
                    - modelfit_CModel
                    - base_PsfFlux
                filt: HSC-G
            gauss_magDiff:
                functor: MagDiff
                args:
                    - base_GaussianFlux
                    - base_PsfFlux
                filt: HSC-G
            count:
                functor: Column
                args:
                    - base_InputCount_value
                filt: HSC-G
            deconvolved_moments:
                functor: DeconvolvedMoments
                filt: HSC-G
                dataset: forced_src
        flags:
            - calib_psfUsed
            - merge_measurement_i
            - merge_measurement_r
            - merge_measurement_z
            - merge_measurement_y
            - merge_measurement_g
            - base_PixelFlags_flag_inexact_psfCenter
            - detect_isPrimary

    The names for each entry under "func" will become the names of columns in the
    output dataset.  All the functors referenced are defined in `lsst.pipe.tasks.functors`.
    Positional arguments to be passed to each functor are in the `args` list,
    and any additional entries for each column other than "functor" or "args" (e.g., `'filt'`,
    `'dataset'`) are treated as keyword arguments to be passed to the functor initialization.

    The "flags" entry is shortcut for a bunch of `Column` functors with the original column and
    taken from the `'ref'` dataset.

    Note, if `'filter'` is provided as part of the `dataId` when running this task (even though
    `deepCoadd_obj` does not use `'filter'`), then this will override the `filt` kwargs
    provided in the YAML file, and the calculations will be done in that filter.

    This task uses the `lsst.pipe.tasks.postprocess.PostprocessAnalysis` object
    to organize and excecute the calculations.

    """
    @property
    def _DefaultName(self):
        raise NotImplementedError('Subclass must define "_DefaultName" attribute')

    @property
    def outputDataset(self):
        raise NotImplementedError('Subclass must define "outputDataset" attribute')

    @property
    def inputDataset(self):
        raise NotImplementedError('Subclass must define "inputDataset" attribute')

    @property
    def ConfigClass(self):
        raise NotImplementedError('Subclass must define "ConfigClass" attribute')

    def runDataRef(self, patchRef):
        parq = patchRef.get()
        dataId = patchRef.dataId
        funcs = self.getFunctors()
        self.log.info("Transforming/standardizing the catalog of %s", dataId)
        df = self.run(parq, funcs=funcs, dataId=dataId)
        self.write(df, patchRef)
        return df

    def run(self, parq, funcs=None, dataId=None):
        """Do postprocessing calculations

        Takes a `ParquetTable` object and dataId,
        returns a dataframe with results of postprocessing calculations.

        Parameters
        ----------
        parq : `lsst.pipe.tasks.parquetTable.ParquetTable`
            ParquetTable from which calculations are done.
        funcs : `lsst.pipe.tasks.functors.Functors`
            Functors to apply to the table's columns
        dataId : dict, optional
            Used to add a `patchId` column to the output dataframe.

        Returns
        ------
            `pandas.DataFrame`

        """
        filt = dataId.get('filter', None)
        return self.transform(filt, parq, funcs, dataId).df

    def getFunctors(self):
        funcs = CompositeFunctor.from_file(self.config.functorFile)
        funcs.update(dict(PostprocessAnalysis._defaultFuncs))
        return funcs

    def getAnalysis(self, parq, funcs=None, filt=None):
        # Avoids disk access if funcs is passed
        if funcs is None:
            funcs = self.getFunctors()
        analysis = PostprocessAnalysis(parq, funcs, filt=filt)
        return analysis

    def transform(self, filt, parq, funcs, dataId):
        analysis = self.getAnalysis(parq, funcs=funcs, filt=filt)
        df = analysis.df
        if dataId is not None:
            for key, value in dataId.items():
                df[key] = value

        return pipeBase.Struct(
            df=df,
            analysis=analysis
        )

    def write(self, df, parqRef):
        parqRef.put(ParquetTable(dataFrame=df), self.outputDataset)

    def writeMetadata(self, dataRef):
        """No metadata to write.
        """
        pass


class TransformObjectCatalogConfig(TransformCatalogBaseConfig):
    filterMap = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        default={},
        doc=("Dictionary mapping full filter name to short one for column name munging."
             "These filters determine the output columns no matter what filters the "
             "input data actually contain.")
    )
    camelCase = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("Write per-filter columns names with camelCase, else underscore "
             "For example: gPsfFlux instead of g_PsfFlux.")
    )
    multilevelOutput = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Whether results dataframe should have a multilevel column index (True) or be flat "
             "and name-munged (False).")
    )


class TransformObjectCatalogTask(TransformCatalogBaseTask):
    """Compute Flatted Object Table as defined in the DPDD

    Do the same set of postprocessing calculations on all bands

    This is identical to `TransformCatalogBaseTask`, except for that it does the
    specified functor calculations for all filters present in the
    input `deepCoadd_obj` table.  Any specific `"filt"` keywords specified
    by the YAML file will be superceded.
    """
    _DefaultName = "transformObjectCatalog"
    ConfigClass = TransformObjectCatalogConfig

    inputDataset = 'deepCoadd_obj'
    outputDataset = 'objectTable'

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", cls.inputDataset,
                               ContainerClass=CoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2")
        return parser

    def run(self, parq, funcs=None, dataId=None):
        dfDict = {}
        analysisDict = {}
        templateDf = pd.DataFrame()
        # Perform transform for data of filters that exist in parq and are
        # specified in config.filterMap
        for filt in parq.columnLevelNames['filter']:
            if filt not in self.config.filterMap:
                self.log.info("Ignoring %s data in the input", filt)
                continue
            self.log.info("Transforming the catalog of filter %s", filt)
            result = self.transform(filt, parq, funcs, dataId)
            dfDict[filt] = result.df
            analysisDict[filt] = result.analysis
            if templateDf.empty:
                templateDf = result.df

        # Fill NaNs in columns of other wanted filters
        for filt in self.config.filterMap:
            if filt not in dfDict:
                self.log.info("Adding empty columns for filter %s", filt)
                dfDict[filt] = pd.DataFrame().reindex_like(templateDf)

        # This makes a multilevel column index, with filter as first level
        df = pd.concat(dfDict, axis=1, names=['filter', 'column'])

        if not self.config.multilevelOutput:
            noDupCols = list(set.union(*[set(v.noDupCols) for v in analysisDict.values()]))
            if dataId is not None:
                noDupCols += list(dataId.keys())
            df = flattenFilters(df, self.config.filterMap, noDupCols=noDupCols,
                                camelCase=self.config.camelCase)

        self.log.info("Made a table of %d columns and %d rows", len(df.columns), len(df))
        return df


class TractObjectDataIdContainer(CoaddDataIdContainer):

    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList

        Generate a list of data references given tract and/or patch.
        This was adapted from `TractQADataIdContainer`, which was
        `TractDataIdContainer` modifie to not require "filter".
        Only existing dataRefs are returned.
        """
        def getPatchRefList(tract):
            return [namespace.butler.dataRef(datasetType=self.datasetType,
                                             tract=tract.getId(),
                                             patch="%d,%d" % patch.getIndex()) for patch in tract]

        tractRefs = defaultdict(list)  # Data references for each tract
        for dataId in self.idList:
            skymap = self.getSkymap(namespace)

            if "tract" in dataId:
                tractId = dataId["tract"]
                if "patch" in dataId:
                    tractRefs[tractId].append(namespace.butler.dataRef(datasetType=self.datasetType,
                                                                       tract=tractId,
                                                                       patch=dataId['patch']))
                else:
                    tractRefs[tractId] += getPatchRefList(skymap[tractId])
            else:
                tractRefs = dict((tract.getId(), tractRefs.get(tract.getId(), []) + getPatchRefList(tract))
                                 for tract in skymap)
        outputRefList = []
        for tractRefList in tractRefs.values():
            existingRefs = [ref for ref in tractRefList if ref.datasetExists()]
            outputRefList.append(existingRefs)

        self.refList = outputRefList


class ConsolidateObjectTableConfig(pexConfig.Config):
    coaddName = pexConfig.Field(
        dtype=str,
        default="deep",
        doc="Name of coadd"
    )


class ConsolidateObjectTableTask(CmdLineTask):
    """Write patch-merged source tables to a tract-level parquet file
    """
    _DefaultName = "consolidateObjectTable"
    ConfigClass = ConsolidateObjectTableConfig

    inputDataset = 'objectTable'
    outputDataset = 'objectTable_tract'

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)

        parser.add_id_argument("--id", cls.inputDataset,
                               help="data ID, e.g. --id tract=12345",
                               ContainerClass=TractObjectDataIdContainer)
        return parser

    def runDataRef(self, patchRefList):
        df = pd.concat([patchRef.get().toDataFrame() for patchRef in patchRefList])
        patchRefList[0].put(ParquetTable(dataFrame=df), self.outputDataset)

    def writeMetadata(self, dataRef):
        """No metadata to write.
        """
        pass
