#
# LSST Data Management System
# Copyright 2008-2021 AURA/LSST.
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
import numpy as np
import pandas as pd
from smatch.matcher import Matcher

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry


__all__ = ['IsolatedStarAssociationConnections',
           'IsolatedStarAssociationConfig',
           'IsolatedStarAssociationTask']


class IsolatedStarAssociationConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=('instrument', 'tract',),
                                         defaultTemplates={}):
    source_table_visit = pipeBase.connectionTypes.Input(
        doc='Source table in parquet format, per visit',
        name='sourceTable_visit',
        storageClass='DataFrame',
        dimensions=('instrument', 'visit'),
        deferLoad=True,
        multiple=True,
    )
    skymap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warped exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    isolated_star_observations = pipeBase.connectionTypes.Output(
        doc='Catalog of isolated star observations',
        name='isolated_star_observations',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract'),
    )
    isolated_star_cat = pipeBase.connectionTypes.Output(
        doc='Catalog of isolated stars',
        name='isolated_star_cat',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract'),
    )


class IsolatedStarAssociationConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=IsolatedStarAssociationConnections):
    """Configuration for IsolatedStarAssociationTask."""

    inst_flux_field = pexConfig.Field(
        doc=('Full name of instFlux field to use for s/n selection and persistence. '
             'The associated flag will be implicity included in bad_flags.'),
        dtype=str,
        default='apFlux_12_0_instFlux',
    )
    match_radius = pexConfig.Field(
        doc='Match radius (arcseconds)',
        dtype=float,
        default=1.0,
    )
    isolation_radius = pexConfig.Field(
        doc='Isolation radius (arcseconds).  Should be at least 2x match_radius.',
        dtype=float,
        default=2.0,
    )
    band_order = pexConfig.ListField(
        doc=('Order of bands to do "primary" matching.  Any bands not specified '
             'will be used alphabetically.'),
        dtype=str,
        default=['i', 'z', 'r', 'g', 'y', 'u'],
    )
    id_column = pexConfig.Field(
        doc='Name of column with source id.',
        dtype=str,
        default='sourceId',
    )
    ra_column = pexConfig.Field(
        doc='Name of column with right ascension.',
        dtype=str,
        default='ra',
    )
    dec_column = pexConfig.Field(
        doc='Name of column with declination.',
        dtype=str,
        default='decl',
    )
    physical_filter_column = pexConfig.Field(
        doc='Name of column with physical filter name',
        dtype=str,
        default='physical_filter',
    )
    band_column = pexConfig.Field(
        doc='Name of column with band name',
        dtype=str,
        default='band',
    )
    extra_columns = pexConfig.ListField(
        doc='Extra names of columns to read and persist (beyond instFlux and error).',
        dtype=str,
        default=['x',
                 'y',
                 'apFlux_17_0_instFlux',
                 'apFlux_17_0_instFluxErr',
                 'apFlux_17_0_flag',
                 'localBackground_instFlux',
                 'localBackground_flag']
    )
    source_selector = sourceSelectorRegistry.makeField(
        doc='How to select sources',
        default='science'
    )

    def setDefaults(self):
        super().setDefaults()

        source_selector = self.source_selector['science']

        flux_flag_name = self.inst_flux_field[0: -len('instFlux')] + 'flag'

        source_selector.flags.bad = ['pixelFlags_edge',
                                     'pixelFlags_interpolatedCenter',
                                     'pixelFlags_saturatedCenter',
                                     'pixelFlags_crCenter',
                                     'pixelFlags_bad',
                                     'pixelFlags_interpolated',
                                     'pixelFlags_saturated',
                                     'centroid_flag',
                                     flux_flag_name]

        source_selector.signalToNoise.fluxField = self.inst_flux_field
        source_selector.signalToNoise.errField = self.inst_flux_field + 'Err'

        source_selector.isolated.parentName = 'parentSourceId'
        source_selector.isolated.nChildName = 'deblend_nChild'

        source_selector.unresolved.name = 'extendedness'


class IsolatedStarAssociationTask:
    """Match isolated stars and suchlike.
    """
    ConfigClass = IsolatedStarAssociationConfig
    _DefaultName = 'isolatedStarAssociation'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_ref_dict = butlerQC.get(inputRefs)

        tract = butlerQC.quantum.dataId['tract']

        source_table_refs = input_ref_dict['sourceTable_visit']

        self.log.info('Running with %d sourceTable_visit dataRefs',
                      len(source_table_refs))

        source_table_ref_dict_temp = {source_table_ref.dataId['visit']: source_table_ref for
                                      source_table_ref in source_table_refs}

        # Need to sort by visit
        source_table_ref_dict = {visit: source_table_ref_dict_temp[visit] for
                                 visit in sorted(source_table_ref_dict_temp.keys())}

        # struct = self.run(tract_info, source_table_ref_dict)
        struct = self.run(input_ref_dict['skymap'], tract, source_table_ref_dict)

        butlerQC.put(pd.DataFrame(struct.star_obs_cat),
                     outputRefs.isolated_star_observations)
        butlerQC.put(pd.DataFrame(struct.star_cat),
                     outputRefs.isolated_star_cat)

    def run(self, skymap, tract, source_table_ref_dict):
        """Run the isolated star association task.

        Parameters
        ----------
        skymap : `lsst.skymap.SkyMap`
            Skymap object.
        tract : `int`
            Tract number.
        source_table_ref_dict : `dict`
            Dictionary of source_table refs.  Key is visit, value is dataref.

        Returns
        -------
        struct : `lsst.pipe.base.struct`
            Struct with outputs for persistence.
        """
        star_obs_cat = self.make_all_star_obs(skymap[tract], source_table_ref_dict)

        star_obs_cat, star_cat = self.match_star_obs(star_obs_cat)

        return pipeBase.Struct(star_obs_cat=star_obs_cat,
                               star_cat=star_cat)

    def make_all_star_obs(self, tract_info, source_table_ref_dict):
        """Make a catalog of all the star observations.

        Parameters
        ----------
        tract_info : `lsst.skymap.TractInfo`
            Information about the tract.
        source_table_ref_dict : `dict`
            Dictionary of source_table refs.  Key is visit, value is dataref.

        Returns
        -------
        star_obs_cat : `np.ndarray`
            Catalog of star observations.
        """
        # Internally, we use a numpy recarray, they are by far the fastest
        # option in testing for relatively narrow tables.
        # (have not tested wide tables)
        all_columns, persist_columns = self._get_source_table_visit_columns()
        poly = tract_info.getOuterSkyPolygon()

        tables = []
        for visit in source_table_ref_dict:
            source_table_ref = source_table_ref_dict[visit]
            df = source_table_ref.get(parameters={'columns': all_columns})

            goodSrc = self.sourceSelector.selectSources(df)

            table = df[persist_columns][goodSrc.selected].to_records()

            # Append columns that include the row in the source table
            # and the matched object index (to be filled later).
            table = np.lib.recfunctions.append_fields(table,
                                                      ['source_row',
                                                       'obj_index'],
                                                      [np.where(goodSrc.selected)[0],
                                                       np.zeros(goodSrc.selected.sum(), dtype=np.int32)],
                                                      dtypes=['i4', 'i4'],
                                                      usemask=False)

            # We cut to the outer tract polygon to ensure consistent matching
            # from tract to tract.
            tract_use = poly.contains(np.deg2rad(table[self.config.ra_column]),
                                      np.deg2rad(table[self.config.dec_column]))

            tables.append(table[tract_use])

        # Combine tables
        star_obs_cat = np.concatenate(tables)

        return star_obs_cat

    def match_star_obs(self, skymap, tract, star_obs_cat):
        """Match the star observations.

        Parameters
        ----------
        skymap : `lsst.skymap.SkyMap`
            Skymap object.
        tract : `int`
            Tract number.
        star_obs_cat : `np.ndarray`
            Star observation catalog.

        Returns
        -------
        star_obs_cat : `np.ndarray`
            Sorted and cropped star observation catalog.
        star_cat : `np.ndarray`
            Catalog of stars and indexes.
        """
        # Figure out band order.
        bands = np.sort(np.unique(star_obs_cat['band']))
        primary_bands = self._primary_band_order(self.config.band_order, bands)

        # Do primary matching
        primary_star_cat = self._match_primary_stars(primary_bands, star_obs_cat)

        primary_star_cat = self._remove_neighbors(primary_star_cat)

        # Crop to the inner tract.
        inner_tract_ids = skymap.findTractIdArray(primary_star_cat[self.config.ra_column],
                                                  primary_star_cat[self.config.dec_column],
                                                  degrees=True)
        use = (inner_tract_ids == tract)
        self.log.info('Total of %d isolated stars in inner tract.', use.sum())

        primary_star_cat = primary_star_cat[use]

        # Set the unique ids here.
        primary_star_cat['isolated_star_id'] = self._compute_unique_ids(skymap,
                                                                        tract,
                                                                        len(primary_star_cat))

        star_obs_cat, primary_star_cat = self._match_observations(bands, star_obs_cat, primary_star_cat)

    def _get_source_table_visit_columns(self):
        """Get the list of sourceTable_visit columns from the config.

        Returns
        -------
        all_columns : `list` [`str`]
            All columns to read
        persist_columns : `list` [`str`]
            Columns to persist (excluding selection columns)
        """
        columns = [self.config.id_column,
                   'visit', 'detector',
                   self.config.ra_column, self.config.dec_column,
                   self.config.physical_filter_column, self.config.band_column,
                   self.config.inst_flux_field, self.config.inst_flux_field + 'Err']
        columns.extend(self.config.extra_columns)

        all_columns = columns.copy()
        if self.source_selector.config.doFlags:
            all_columns.extend(self.source_selector.config.flags.bad)
        if self.source_selector.config.doUnresolved:
            all_columns.append(self.source_selector.config.unresolved.name)
        if self.source_selector.config.doIsolated:
            all_columns.append(self.source_selector.config.isolated.parentName)
            all_columns.append(self.source_selector.config.isolated.nChildName)

        return all_columns, columns

    @staticmethod
    def _primary_band_order(band_order, bands):
        """Compute order of primary bands.

        This will return ordered list of bands according to band_order,
        with extra bands left in input order (typically alphabetically).

        Parameters
        ----------
        band_order : `list` [`str`]
            Preferred order of bands.
        bands : `list` [`str`]
            List of unique bands to put in order.

        Returns
        -------
        primary_band_order : `list` [`str`]
            Ordered list of bands.
        """
        primary_bands = []
        # Put bands in the configured band order
        for band in band_order:
            if band in bands:
                primary_bands.append(band)
        # For any remaining bands not configured, leave in order.
        for band in bands:
            if band not in primary_bands:
                primary_bands.append(band)

        return primary_bands

    def _match_primary_stars(self, primary_bands, star_obs_cat):
        """Match primary stars.

        Parameters
        ----------
        primary_bands : `list` [`str`]
            Ordered list of primary bands.
        star_obs_cat : `np.ndarray`
            Catalog of star observations.

        Returns
        -------
        primary_star_cat : `np.ndarray`
            Catalog of primary star positions
        """
        ra_col = self.config.ra_column
        dec_col = self.config.dec_column

        max_len = max([len(primary_band) for primary_band in primary_bands])

        dtype = [('isolated_star_id', 'i8'),
                 (ra_col, 'f8'),
                 (dec_col, 'f8'),
                 ('primary_band', f'U{max_len}'),
                 ('obs_cat_index', 'i4'),
                 ('nobs', 'i4')]

        for band in primary_bands:
            dtype.append((f'obs_cat_index_{band}', 'i4'))
            dtype.append((f'nobs_{band}', 'i4'))

        primary_star_cat = None
        for primary_band in primary_bands:
            use = (star_obs_cat['band'] == primary_band)

            ra = star_obs_cat[ra_col][use]
            dec = star_obs_cat[dec_col][use]

            with Matcher(ra, dec) as matcher:
                idx = matcher.query_self(self.config.match_radius/3600., min_match=1)

            count = len(idx)

            if count == 0:
                self.log.info('Found 0 primary stars in %s band.', band)
                continue

            band_cat = np.zeros(count, dtype=dtype)
            band_cat['primary_band'] = primary_band

            # rotate if necessary
            rotated = False
            if ra.min() < 60.0 and ra.max() > 300.0:
                ra_temp = (ra + 180.0) % 360. - 180.
                rotated = True
            else:
                ra_temp = ra

            # Compute mean position for each primary star
            for i, row in enumerate(idx):
                row = np.array(row)
                band_cat[ra_col][i] = np.sum(ra_temp[row])/len(row)
                band_cat[dec_col][i] = np.sum(dec[row])/len(row)

            if rotated:
                band_cat[ra_col] %= 360.0

            # Match to previous band catalog(s), and remove duplicates.
            if primary_star_cat is None or len(primary_star_cat) == 0:
                primary_star_cat = band_cat
            else:
                with Matcher(band_cat[ra_col], band_cat[dec_col]) as matcher:
                    idx = matcher.query_radius(primary_star_cat[ra_col],
                                               primary_star_cat[dec_col],
                                               self.config.match_radius/3600.)
                # Any object with a match should be removed.
                match_indices = np.array([i for i in range(len(idx)) if len(idx[i]) > 0])
                band_cat = np.delete(band_cat, match_indices)

                primary_star_cat = np.append(primary_star_cat, band_cat)
            self.log.info('Found %d primary stars in %s band.', len(band_cat), primary_band)

        return primary_star_cat

    def _remove_neighbors(self, primary_star_cat):
        """Remove neighbors from the primary star catalog.

        Parameters
        ----------
        primary_star_cat : `np.ndarray`
            Primary star catalog.

        Returns
        -------
        primary_star_cat_cut : `np.ndarray`
            Primary star cat with neighbors removed.
        """
        ra_col = self.config.ra_column
        dec_col = self.config.dec_column

        with Matcher(primary_star_cat[ra_col], primary_star_cat[dec_col]) as matcher:
            # By setting min_match=2 objects that only match to themselves
            # will not be recorded.
            idx = matcher.query_self(self.config.isolation_radius/3600., min_match=2)

        neighbor_indices = []
        for row in idx:
            neighbor_indices.extend(row)

        if len(neighbor_indices) > 0:
            neighbored = np.unique(neighbor_indices)
            self.log.info('Cutting %d objects with close neighbors.', len(neighbored))
            primary_star_cat = np.delete(primary_star_cat, neighbored)

        return primary_star_cat

    def _match_observations(self, bands, star_obs_cat, primary_star_cat):
        """Match observations to primary stars.

        Parameters
        ----------
        bands : `list` [`str`]
            List of bands.
        star_obs_cat : `np.ndarray`
            Array of star observations.
        primary_star_cat : `np.ndarray`
            Array of primary stars.

        Returns
        -------
        star_obs_cat_cut : `np.ndarray`
            Cropped and sorted array of star observations.
        primary_star_cat : `np.ndarray`
            Catalog of isolated stars, with indexes to star_obs_cat_cut.
        """
        ra_col = self.config.ra_column
        dec_col = self.config.dec_column

        # We match observations per-band because it allows us to have sorted
        # observations for easy retrieval of per-band matches.
        n_obs_per_band_per_obj = np.zeros((len(bands),
                                           len(primary_star_cat)),
                                          dtype=np.int32)
        band_uses = []
        idxs = []
        with Matcher(primary_star_cat[ra_col], primary_star_cat[dec_col]) as matcher:
            for b, band in enumerate(bands):
                band_use, = np.where(star_obs_cat['band'] == band)

                idx = matcher.query_radius(star_obs_cat[ra_col][band_use],
                                           star_obs_cat[dec_col][band_use],
                                           self.config.match_radius/3600.)
                n_obs_per_band_per_obj[b, :] = np.array([len(row) for row in idx])
                idxs.append(idx)
                band_uses.append(band_use)

        n_obs_per_obj = np.sum(n_obs_per_band_per_obj, axis=0)

        primary_star_cat['nobs'] = n_obs_per_obj
        primary_star_cat['obs_cat_index'][1:] = np.cumsum(n_obs_per_obj)[:-1]

        n_tot_obs = primary_star_cat['obs_cat_index'][-1] + primary_star_cat['nobs'][-1]

        # Temporary arrays until we crop/sort the observation catalog
        obs_index = np.zeros(n_tot_obs, dtype=np.int32)
        obj_index = np.zeros(n_tot_obs, dtype=np.int32)

        ctr = 0
        for i in range(len(primary_star_cat)):
            obj_index[ctr: ctr + n_obs_per_obj[i]] = i
            for b in range(len(bands)):
                obs_index[ctr: ctr + n_obs_per_band_per_obj[b, i]] = band_uses[b][idxs[b][i]]
                ctr += n_obs_per_band_per_obj[b, i]

        obs_cat_index_band_offset = np.cumsum(n_obs_per_band_per_obj, axis=0)

        for b, band in enumerate(bands):
            primary_star_cat[f'nobs_{band}'] = n_obs_per_band_per_obj[b, :]
            if b == 0:
                # The first band listed is the same as the overall star
                primary_star_cat[f'obs_cat_index_{band}'] = primary_star_cat['obs_cat_index']
            else:
                # Other band indices are offset from the previous band
                primary_star_cat[f'obs_cat_index_{band}'] = (primary_star_cat['obs_cat_index']
                                                             + obs_cat_index_band_offset[b - 1, :])

        # Reset all indices to illegal value -1
        star_obs_cat['obj_index'] = -1
        star_obs_cat['obj_index'][obs_index] = obj_index

        star_obs_cat = star_obs_cat[obs_index]

        return star_obs_cat, primary_star_cat

    def _compute_unique_ids(self, skymap, tract, nstar):
        """Compute unique star ids.

        This is a simple hash of the tract and star to provide an
        id that is unique for a given processing.

        Parameters
        ----------
        skymap : `lsst.skymap.Skymap`
            Skymap object.
        tract : `int`
            Tract id number.
        nstar : `int`
            Number of stars.

        Returns
        -------
        ids : `np.ndarray`
            Array of unique star ids.
        """
        # The end of the id will be big enough to hold the tract number
        mult = 10**(int(np.log10(len(skymap))) + 1)

        return (np.arange(nstar) + 1)*mult + tract
