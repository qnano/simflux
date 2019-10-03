
import sys,os

#if __name__ == "__main__":
#    basedir=os.path.abspath(os.path.dirname(__file__) + '/../..')
 #   print(basedir)
  #  os.chdir(basedir)
   # sys.path.append(basedir)

import numpy as np
import tqdm
from PyQt5.QtWidgets import QApplication, QLineEdit, QFileDialog, QDialog,QVBoxLayout
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pickle

import simflux.ui.main_ui as main_ui
import simflux.ui.linklocs_ui as linklocs_ui

from simflux.ui.progressbar import ProgressBar

import threading
import json

import simflux.tiff_to_locs as tiff_to_locs
import simflux.extract_rois as extract_rois

from smlmlib.util import imshow_hstack

import simflux.locs_to_pattern as simflux_pattern

from simflux.ui.drift_correct_dlg import DriftCorrectionDialog


class LinkLocsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = linklocs_ui.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.btnBrowse.clicked.connect(self._onBrowse)
        self.ui.btnEstimate.clicked.connect(self.estimate)
    
    def setLocsFile(self,fn):
        self.ui.txtLocsFile.setText(fn)

    def _onBrowse(self):
        options = QFileDialog.Options()
#        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"", "","All Files (*);;HDF5 Files (*.hdf5)", options=options)
        if fileName:
            self.ui.txtLocsFile.setText(fileName)
            
    def estimate(self):
        from utils.link_locs import estimate_on_time
        maxdist = self.ui.maxDistance.value()
        frameskip = self.ui.frameskip.value()
        fig,bins,framecounts = estimate_on_time(self.ui.txtLocsFile.text(),maxdist,frameskip)

        import simflux.ui.qtplot as qtplot 
        plotdlg=qtplot.PlotDialog(fig,self)
        plotdlg.setModal(True)
        plotdlg.show()

def getWidgetValues(widgets):
    d={}
    for w in widgets:
        if type(w) == QtWidgets.QDoubleSpinBox or type(w) == QtWidgets.QSpinBox:
            v = w.value()
        elif type(w) == QLineEdit:
            v = w.text()
        else:
            continue
        d[w.objectName()] = v
    return d

def setWidgetValues(widgets,values):
    for w in widgets:
        if w.objectName() in values:
            v = values[w.objectName()]
            if type(w) == QtWidgets.QDoubleSpinBox or type(w) == QtWidgets.QSpinBox:
                w.setValue(v)
            elif type(w) == QLineEdit:
                w.setText(v)

class Window(QDialog):
    localizeDone = QtCore.pyqtSignal()
    roiExtractionDone = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.title = 'SIMFLUX Viewer'

        self.ui = main_ui.Ui_Dialog()
        ui=self.ui
        ui.setupUi(self)
        
        ui.btnBrowseTiff.clicked.connect(self.onBrowseTiff)
        ui.btnEstimAnglePitch.clicked.connect(self.estimAnglePitch)
        ui.btnEstimPhaseDepth.clicked.connect(self.estimPhaseDepth)
        ui.btnLocalize.clicked.connect(self.localize)
        ui.btnLinkLocs.clicked.connect(self.linklocs)
        
        ui.btnBrowseCameraOffset.clicked.connect(self.onBrowseCameraOffsetFile)
        ui.btnBrowseROIs.clicked.connect(self.onBrowseROIFile)
        
        ui.btnFixDepth.clicked.connect(self.onFixDepth)
        
        ui.btnDriftCorrection.clicked.connect(self.onDriftCorrection)
                
        ui.btnExtractROIs.clicked.connect(self.onExtractROIs)
        
        ui.btnLoadMod.clicked.connect(self.onLoadMod)
        ui.btnSaveMod.clicked.connect(self.onSaveMod)
        
        ui.btnRunSimflux.clicked.connect(self.simflux)
        
        self.localizeDone.connect(self.onLocalizeDone)
        self.roiExtractionDone.connect(self.onROIExtractionDone)
        
        self.cfgFile = 'simflux/ui/ui-cfg.json'
        self.cfgWidgets = {
                ui.fixDepth, 
                ui.roisize,
                ui.gain,
                ui.offset,
                ui.detectionThreshold,
                ui.maxpitch,
                ui.minpitch,
                ui.numPhaseSteps,
                ui.pixelsize,
                ui.psfSigmaX, ui.psfSigmaY,
                ui.tiffPath,
                ui.smlmLocsFile,
                ui.txtCameraOffsetFile,
                ui.startFrame,
                ui.maxLinkDistance,
                ui.maxLinkFrameskip,
                ui.txtROIFile,
                ui.roiExtractMinSpotFrames,
                ui.roiExtractSpotFrames,
                ui.roiExtractAppend,
                ui.maxLinkDistanceIntensity,
                ui.angle0, ui.angle1, ui.pitch0, ui.pitch1
                }
        self.load()
        
    def onDriftCorrection(self):
        dlg = DriftCorrectionDialog(self, self.ui.smlmLocsFile.text())
        dlg.show()
        
    def load(self):
        path = os.path.abspath(os.curdir+"/"+self.cfgFile)
        print(f"Loading UI state from {path}")
        if os.path.exists(self.cfgFile):
            with open(self.cfgFile,'r') as f:
                d = json.load(f)
                setWidgetValues(self.cfgWidgets,d)
        
    def save(self):
        d = getWidgetValues(self.cfgWidgets)
        with open(self.cfgFile,'w') as f:
            json.dump(d,f,indent=4)
        
    def closeEvent(self,event):
        self.save()
        
    def linklocs(self):
        dlg = LinkLocsDialog(self)
        dlg.setLocsFile(self.ui.smlmLocsFile.text())
        dlg.show()
        
    def updatePaths(self):
        tiff_path = self.ui.tiffPath.text()
        locs_fn = os.path.splitext(tiff_path)[0]+".hdf5"
        self.ui.smlmLocsFile.setText(locs_fn)
                
        rois_path = os.path.splitext(tiff_path)[0]+".rois"
        self.ui.txtROIFile.setText(rois_path)        
        
        
    def onBrowseCameraOffsetFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse image/movie to use as offset:", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.txtCameraOffsetFile.setText(fileName)

    def onBrowseROIFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse ROI file", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.txtROIFile.setText(fileName)

    def onBrowseTiff(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse TIFF", "","All Files (*);;TIFF File (*.tif)", options=options)
        if fileName:
            self.ui.tiffPath.setText(fileName)
            self.updatePaths()

    def onLoadMod(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Browse mod pickle", "","All Files (*);;Pickle files (*.pickle)", options=options)
        if fileName:
            with open(fileName, "rb") as pf:
                mod = pickle.load(pf)['mod']
            self.setModulation(mod)

    def onSaveMod(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,"Browse mod pickle", "","All Files (*);;Pickle files (*.pickle)", options=options)
        if fileName:
            mod = self.getModulation()
            phase, depth, relint = self.getPhaseDepth()
            angle, pitch = self.getAngleAndPitch()
            
            mod_info = {"mod": mod, "pitch": pitch, "angles": angle, "phase": phase, "depth": depth}
            with open(fileName, "wb") as df:
                pickle.dump(mod_info, df)

    def getModulation(self):

        angles,pitch = self.getAngleAndPitch()
        phase,depth,relint = self.getPhaseAndDepth()

        return simflux_pattern.compute_mod(self.getPatternFrames(), angles, pitch, phase, depth, relint)
            
    def setModulation(self, mod):
        angles,pitch = simflux_pattern.mod_angle_and_pitch(mod,self.getPatternFrames())

        pf = self.getPatternFrames()
        self.setAngleAndPitch(angles,pitch)
        self.setPhaseDepth(mod[:,3][pf], mod[:,2][pf], mod[:,4][pf])
                                          
    def getPatternFrames(self):
        phase_steps=self.ui.numPhaseSteps.value()
        pattern_frames=np.array([np.arange(0,phase_steps*2,2),np.arange(0,phase_steps*2,2)+1])
        return pattern_frames

    def estimAnglePitch(self):
        freq_minmax=[2*np.pi/self.ui.maxpitch.value(),2*np.pi/self.ui.minpitch.value()]
        pattern_frames=self.getPatternFrames()
        angles,pitch=simflux_pattern.estimate_pitch_and_angle(self.ui.smlmLocsFile.text(),pattern_frames,freq_minmax)
        self.setAngleAndPitch(angles,pitch)
        
    def onExtractROIs(self):
        locs_fn = self.ui.smlmLocsFile.text()
        tiff_path = self.ui.tiffPath.text()
        rois_path = self.ui.txtROIFile.text()

        pbar = ProgressBar("Extracting ROIs and estimating spot background and intensity")
        def progress_update(msg,done):
            if msg is not None:
                pbar.setMsg.emit(msg)
            if done is not None:
                pbar.update.emit(done)
            return not pbar.abortPressed
        
        cfg = self.getConfig()

        cfg = {**cfg,
             'maxlinkdistXY': self.ui.maxLinkDistance.value(),
             'maxlinkdistI': self.ui.maxLinkDistanceIntensity.value(),
             'maxlinkframeskip': self.ui.maxLinkFrameskip.value()
             }
        
        maxroiframes = self.ui.roiExtractSpotFrames.value()
        minroiframes = self.ui.roiExtractMinSpotFrames.value()
        appendFrames = self.ui.roiExtractAppend.value()
        
        def process_thread():
            self.rois,self.roiframes = extract_rois.extract_rois(rois_path, tiff_path, cfg, minroiframes, 
                                      maxroiframes, appendFrames, locs_fn, progress_update)
            if not pbar.abortPressed:             
                self.roiExtractionDone.emit()
            
        t = threading.Thread(target=process_thread)
        t.start()
        pbar.show()
        
    def onViewROIs(self):
        rois_path = self.ui.txtROIFile.text()
        roidata = extract_rois.ROIData.load(rois_path)
        
        plt.figure()
        for k in range(20):
            imshow_hstack(roidata.frames[k])

        
    def setAngleAndPitch(self,angles,pitch):
        ad = np.rad2deg(angles)
        pixelsize = self.ui.pixelsize.value()
        p = pitch*pixelsize
        
        self.ui.angle0.setValue(ad[0])
        self.ui.angle1.setValue(ad[1])
        self.ui.pitch0.setValue(p[0])
        self.ui.pitch1.setValue(p[1])
        
    def onFixDepth(self):
        phase,depth,relint=self.getPhaseAndDepth()
        depth[:,:]=self.ui.fixDepth.value()
        self.setPhaseDepth(phase,depth,relint)
        
    def getAngleAndPitch(self):
        angle = np.array( [ self.ui.angle0.value(), self.ui.angle1.value() ] )
        pitch = np.array( [ self.ui.pitch0.value(), self.ui.pitch1.value() ] )
        angle = np.deg2rad(angle)
        pitch_in_pixels = pitch / self.ui.pixelsize.value()

        return angle,pitch_in_pixels
                
    def estimPhaseDepth(self):
        locs_fn = self.ui.smlmLocsFile.text()
        
        angle,pitch = self.getAngleAndPitch()
                        
        pattern_frames = self.getPatternFrames()
        phase, depth, relint = simflux_pattern.estimate_phase_and_depth(locs_fn, angle, pitch, pattern_frames)

        self.setPhaseDepth(phase,depth,relint)
        
    def setPhaseDepth(self,phase,depth,relint):
        tbl = self.ui.tableMod
        pattern_frames = self.getPatternFrames()
        phase_steps = self.ui.numPhaseSteps.value()
        tbl.setRowCount(len(pattern_frames.flatten()))
        tbl.setColumnCount(4)
        
        headers = ['Axis', 'Phase (deg)', 'Depth', 'Rel. intensity']
        tbl.setHorizontalHeaderLabels(headers)

        phase = np.rad2deg(phase)

        for i, pf in enumerate(pattern_frames):
            for step, fn in enumerate(pf):
                lbl = QtWidgets.QLabel()
                lbl.setText(str(i))
                tbl.setCellWidget(i*phase_steps+step, 0, lbl)

                w = QtWidgets.QLabel()
                w.setText(f"{phase[i,step]:.5f}")
                tbl.setCellWidget(i*phase_steps+step, 1, w)

                w = QtWidgets.QLabel()
                w.setText(f"{depth[i,step]:.5f}")
                tbl.setCellWidget(i*phase_steps+step, 2, w)

                w = QtWidgets.QLabel()
                w.setText(f"{relint[i,step]:.5f}")
                tbl.setCellWidget(i*phase_steps+step, 3, w)
        
    def getPhaseAndDepth(self):
        tbl = self.ui.tableMod
        pattern_frames = self.getPatternFrames()
        
        numaxis = len(pattern_frames)
        numsteps = self.ui.numPhaseSteps.value()
        
        phase = np.zeros((numaxis,numsteps))
        depth = np.zeros((numaxis,numsteps))
        relint = np.zeros((numaxis,numsteps))
            
        for i in range(numaxis):
            for j in range(numsteps):
                phase[i,j]=float(tbl.cellWidget(i*numsteps+j,1).text())
                depth[i,j]=float(tbl.cellWidget(i*numsteps+j,2).text())
                relint[i,j]=float(tbl.cellWidget(i*numsteps+j,3).text())
                
            
        return np.deg2rad(phase),depth,relint
        
        
    def getConfig(self):
        offset = self.ui.offset.value()

        offsetFile = self.ui.txtCameraOffsetFile.text()
        if len(offsetFile)>0:
            offset = offsetFile
        
        sigmaX= self.ui.psfSigmaX.value()
        sigmaY = self.ui.psfSigmaY.value()
        
        cfg = {
                'sigma': [sigmaX, sigmaY],
                'roisize': self.ui.roisize.value(),
                'threshold': self.ui.detectionThreshold.value(),
                'gain': self.ui.gain.value(),
                'offset': offset,
                'startframe': self.ui.startFrame.value()
            }
        return cfg
    
    def simflux(self):
        rois_path = self.ui.txtROIFile.text()
        
        sf_fn = os.path.splitext(rois_path)[0]+"_sf.hdf5"
        g2d_fn = os.path.splitext(rois_path)[0]+"_g2d.hdf5"

        sigma = self.getConfig()['sigma']
        mod = self.getModulation()
        
        rd = extract_rois.ROIData.load(rois_path)
        
        print(f"Estimating with 2D Gauss..")
        results = extract_rois.localize(rd.rois,rd.frames,sigma,mod=None)
        results.SaveHDF5(g2d_fn, rd.imgshape)

        print(f"Estimating with SIMFLUX..")
        results = extract_rois.localize(rd.rois,rd.frames,sigma,mod=mod)
        results.SaveHDF5(sf_fn, rd.imgshape)
        
    def localize(self):
        tiff_path = self.ui.tiffPath.text()
        if not os.path.exists(tiff_path):
            return
        
        cfg = self.getConfig()
        locs_fn = self.ui.smlmLocsFile.text()
        est_sigma = self.ui.checkEstimSigma.isChecked()
        
        self.ui.labelLocsInfo.setText('')
        
        pbar = ProgressBar("Running spot detection and 2D Gaussian localization...")

        def progress_update(msg,done):
            if msg is not None:
                pbar.setMsg.emit(msg)
            if done is not None:
                pbar.update.emit(done)
            return not pbar.abortPressed
        
        def localize_thread():
            print (f"Localize thread: {threading.get_ident()}")
            self.results, self.imgshape = tiff_to_locs.localize(tiff_path, cfg, locs_fn, progress_update, est_sigma)
            if not pbar.abortPressed:
                self.localizeDone.emit()
            
        t = threading.Thread(target=localize_thread)
        t.start()
        
        pbar.show()

    @QtCore.pyqtSlot()
    def onLocalizeDone(self):
        print("localize done")
                
        if 'sx' in self.results.colnames:
            sx = self.results.estim[:, self.results.ColIdx('sx')]
            sy = self.results.estim[:, self.results.ColIdx('sy')]
            self.ui.psfSigmaX.setValue(np.median(sx))
            self.ui.psfSigmaY.setValue(np.median(sy))
            
            plt.figure()
            plt.hist([sx,sy],label=['Sigma X','Sigma Y'],range=(1,3),bins=30)
            plt.legend()
            plt.xlabel('PSF Sigma [pixels]')
            plt.show()
        self.showLocsInfo()

    @QtCore.pyqtSlot()
    def onROIExtractionDone(self):
        print("roi extraction done")
        
    def showLocsInfo(self):
        m_crlb_x = np.median(self.results.CRLB()[:,0])
        m_bg= np.median(self.results.estim[:,3])
        self.ui.labelLocsInfo.setText(f"#Spots: {len(self.results.estim)}. Imgsize:{self.imgshape[0]}x{self.imgshape[1]} pixels. Median CRLB X: {m_crlb_x:.2f} [pixels], bg:{m_bg:.1f}")

def run_ui():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
       
    wnd = Window()
    wnd.show()
    wnd.activateWindow()
    app.exec_()
#    del tqdm    # prevent exception at exit about not being able to join thread
    del app     # prevent IPython+Qt issue https://github.com/spyder-ide/spyder/issues/2970

if __name__ == '__main__':
    run_ui()