# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from lsst.daf.butler import Butler
from lsst.skymap import BaseSkyMap
from lsst.pipe.tasks.makeDiscreteSkyMap import MakeDiscreteSkyMapTask, MakeDiscreteSkyMapConfig
from lsst.obs.base.utils import getInstrument


def makeDiscreteSkyMap(repo, config_file, collections, instrument,
                       skymap_id='discrete', old_skymap_id=None):
    """Implements the command line interface `butler make-discrete-skymap` subcommand,
    should only be called by command line tools and unit test code that tests
    this function.

    Constructs a skymap from calibrated exposure in the butler repository

    Parameters
    ----------
    repo : `str`
        URI to the location to read the repo.
    config_file : `str` or `None`
        Path to a config file that contains overrides to the skymap config.
    collections : `list` [`str`]
        An expression specifying the collections to be searched (in order) when
        reading datasets, and optionally dataset type restrictions on them.
        At least one collection must be specified.  This is the collection
        with the calibrated exposures.
    instrument : `str`
        The name or fully-qualified class name of an instrument.
    skymap_id : `str`, optional
        The identifier of the skymap to save.  Default is 'discrete'.
    old_skymap_id : `str`, optional
        The identifer of the skymap to append to.  Must differ from
        ``skymap_id``.  Ignored unless ``config.doAppend=True``.
    """
    butler = Butler(repo, collections=collections, writeable=True)
    instr = getInstrument(instrument, butler.registry)
    config = MakeDiscreteSkyMapConfig()
    instr.applyConfigOverrides(MakeDiscreteSkyMapTask._DefaultName, config)

    if config_file is not None:
        config.load(config_file)
    # The coaddName for a SkyMap is only relevant in Gen2, and we completely
    # ignore it here; once Gen2 is gone it can be removed.
    oldSkyMap = None
    if config.doAppend:
        if old_skymap_id is None:
            raise ValueError("old_skymap_id must be provided if config.doAppend is True.")
        dataId = {'skymap': old_skymap_id}
        try:
            oldSkyMap = butler.get(BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, collections=collections,
                                   dataId=dataId)
        except LookupError as e:
            msg = (f"Could not find seed skymap with dataId {dataId} "
                   f"in collections {collections} but doAppend is {config.doAppend}.  Aborting...")
            raise LookupError(msg, *e.args[1:])

    datasets = butler.registry.queryDatasets('calexp', collections=collections)
    wcs_md_tuple_list = [(butler.getDirect(ref.makeComponentRef("wcs")),
                          butler.getDirect(ref.makeComponentRef("metadata")))
                         for ref in datasets]
    task = MakeDiscreteSkyMapTask(config=config)
    result = task.run(wcs_md_tuple_list, oldSkyMap)
    result.skyMap.register(skymap_id, butler)
    butler.put(result.skyMap, BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, dataId={'skymap': skymap_id},
               run=BaseSkyMap.SKYMAP_RUN_COLLECTION_NAME)
