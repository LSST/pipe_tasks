# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import math

import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.meas.algorithms.apertureCorrection as maApCorr
import lsst.meas.photocal as photocal
import lsst.pipe.base as pipeBase
from .astrometry import AstrometryTask
from .repair import RepairTask
from .measurePsf import MeasurePsfTask
from .photometry import PhotometryTask, RephotometryTask

def propagateFlag(flag, old, new):
    """Propagate a flag from one source to another"""
    if old.getFlagForDetection() & flag:
        new.setFlagForDetection(new.getFlagForDetection() | flag)


class CalibrateTask(pipeBase.Task):
    """Conversion notes:
    
    Disabled display until we figure out how to turn it off or on
    
    Warning: I'm not sure I'm using metadata correctly (to replace old sdqa code)
    
    Made new subtasks for measuring PSF and astrometry    
    
    Eliminated the background subtask because it was such a thin layer around muDetection.estimateBackground
    
    Modified to NOT estimate a new background model if the user supplies one. The old code first applied
    the user-supplied background (if any) then fit and subtracted a new background.
    """
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair", RepairTask, keepCRs=True)
        self.makeSubtask("photometry", PhotometryTask)
        self.makeSubtask("measurePsf", MeasurePsfTask)
        self.makeSubtask("rephotometry", RephotometryTask)
        self.makeSubtask("astrometry", AstrometryTask)

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, background=None):
        """Calibrate an exposure: measure PSF, subtract background, measure astrometry and photometry

        @param exposure Exposure to calibrate
        @param defects List of defects on exposure
        @param background Background model; if None and background subtraction is enabled, background is fit
        @return a pipeBase.Struct with fields:
        - psf: Point spread function
        - apcorr: Aperture correction
        - sources: Sources used in calibration
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        """
        assert exposure is not None, "No exposure provided"

        fakePsf, wcs = self.makeFakePsf(exposure)

        self.repair.run(exposure, fakePsf, defects=defects)

        if self.policy.doPsf or self.policy.doAstrometry or self.policy.doZeropoint:
            with self.timer("photDuration"):
                photRet = self.photometry.run(exposure, fakePsf)
                sources = photRet.sources
                footprints = photRet.footprints
        else:
            sources, footprints = None, None

        if self.policy.doPsf:
            psfRet = self.measurePsf.run(exposure, sources)
            psf = psfRet.psf
            cellSet = psfRet.cellSet
        else:
            psf, cellSet = None, None

        if self.policy.doPsf and self.policy.doapcorr:
            apcorr = self.apCorr(exposure, cellSet) # calculate the aperture correction; we may use it later
        else:
            apcorr = None

        # Wash, rinse, repeat with proper PSF

        if self.policy.doPsf:
            self.repair(exposure, psf, defects=defects, preserve=False)

        if self.policy.doBackground:
            with self.timer("backgroundDuration"):
                if background is not None:
                    background = value.getImageF()
                    exposure += background
                    self.log.log(self.log.INFO, "Subtracted supplied background")
                else:
                    # Subtract background
                    background, exposure = muDetection.estimateBackground(
                        exposure, self.policy.background, subtract=True)
                    self.log.log(self.log.INFO, "Fit and subtracted background")

        if self.policy.doPsf and (self.policy.doAstrometry or self.policy.doZeropoint):
            rephotRet = self.rephotometry.run(exposure, footprints, psf, apcorr)
            for old, new in zip(sources, rephotRet.sources):
                for flag in (measAlg.Flags.STAR, measAlg.Flags.PSFSTAR):
                    propagateFlag(flag, old, new)
            sources = rephotRet.sources
            del rephotRet
        
        if self.policy.doAstrometry or self.policy.doZeropoint:
            astromRet = self.astrometry.run(exposure, sources)
            matches = astromRet.matches
            matchMeta = astromRet.matchMeta
        else:
            matches, matchMeta = None, None

        if self.policy.doZeropoint:
            self.zeropoint(exposure, matches)

#        self.display('calibrate', exposure=exposure, sources=sources, matches=matches)

        return pipeBase.Struct(
            psf = psf,
            apcorr = apcorr,
            sources = sources,
            matches = matches,
            matchMeta = matchMeta,
        )

    def makeFakePsf(self, exposure):
        """Initialise the calibration procedure by setting the PSF and WCS

        @param exposure Exposure to process
        @return PSF, WCS
        """
        assert exposure, "No exposure provided"
        
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        calibrate = self.config['calibrate']
        model = calibrate['model']
        fwhm = calibrate['fwhm'] / wcs.pixelScale().asArcseconds()
        size = calibrate['size']
        psf = afwDet.createPsf(model, size, size, fwhm/(2*math.sqrt(2*math.log(2))))
        return psf, wcs

    @pipeBase.timeMethod
    def apCorr(self, exposure, cellSet):
        """Measure aperture correction

        @param exposure Exposure to process
        @param cellSet Set of cells of PSF stars
        """
        assert exposure, "No exposure provided"
        assert cellSet, "No cellSet provided"
        policy = self.config['apcorr'].getPolicy()
        control = maApCorr.ApertureCorrectionControl(policy)
        apCorrMetadata = self.metadata.add("apCorr")
        corr = maApCorr.ApertureCorrection(exposure, cellSet, apCorrMetadata, control, self.log)
        x, y = exposure.getWidth() / 2.0, exposure.getHeight() / 2.0
        value, error = corr.computeAt(x, y)
        self.log.log(self.log.INFO, "Aperture correction using %d/%d stars: %f +/- %f" %
                     (apCorrMetadata["phot.apCorr.numAvailStars"],
                      apCorrMetadata["phot.apCorr.numGoodStars"],
                      value, error))
        return corr

    @pipeBase.timeMethod
    def zeropoint(self, exposure, matches):
        """Photometric calibration

        @param exposure Exposure to process
        @param matches Matched sources
        """
        assert exposure, "No exposure provided"
        assert matches, "No matches provided"

        zp = photocal.calcPhotoCal(matches, log=self.log, goodFlagValue=0)
        self.log.log(self.log.INFO, "Photometric zero-point: %f" % zp.getMag(1.0))
        exposure.getCalib().setFluxMag0(zp.getFlux(0))
        return


class CalibratePsfTask(CalibrateTask):
    """Calibrate only the PSF for an image.
    
    Explicitly turns off other functions.
    
    Conversion notes:
    - Is it really necessary to restore the old policy?
    - Surely there is a cleaner way to do this, such as creating a policy that
      has these flags explicitly turned off?
    """
    def run(self, *args, **kwargs):
        oldPolicy = self.policy.copy()
        self.policy.doBackground = False
        self.policy.doDistortion = False
        self.policy.doAstrometry = False
        self.policy.doZeropoint = False

        retVal = CalibrateTask.run(self, *args, **kwargs)

        self.policy = oldPolicy
        
        return retVal
