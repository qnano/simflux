"""

Main SIMFLUX data processing pipeline.

- Takes TIFF path and known angle and frequencies
- Does spot detection and localization on summed-frames
- Estimates phases on the dataset split in 10 pieces (to know the precision of the phase estimates)
- Filters out all spots that have frames with intensity below certain threshold (= min filter)
- Run non-blinking SIMFLUX localization on it.
- For both Gauss2D and SIMFLUX:
    - Apply drift correction on the results

"""
import numpy as np
from smlmlib.base import SMLM
from smlmlib import util as su
import gaussian.fitters as gaussfit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio
import utils.peaks as peaks
import utils.localizations as loc
from utils.localizations import DataColumns
from spotlist import SpotList
import matplotlib.patches
import math
import os,pickle
from smlmlib.context import Context

import tqdm
from smlmlib.psf import PSF
from smlmlib.psf_queue import PSF_Queue
import smlmlib.spotdetect as spotdetect
from smlmlib.calib import GainOffset_Calib,GainOffsetImage_Calib
from smlmlib.simflux import SIMFLUX
import time

import utils.caching as caching

import read_tiff
import tiff_to_locs

figsize=(9,7)

import matplotlib as mpl
#mpl.use('svg')
new_rc_params = {
#    "font.family": 'Times',
    "font.size": 15,
#    "font.serif": [],
    "svg.fonttype": 'none'} #to store text as text, not as path
mpl.rcParams.update(new_rc_params)


# Make sure the angles dont wrap around, so you can plot them and take mean
def unwrap_angle(ang):
    r = ang * 1
    ang0 = ang.flatten()[0]
    r[ang > ang0 + math.pi] -= 2 * math.pi
    r[ang < ang0 - math.pi] += 2 * math.pi
    return r


# Pattern angles wrap at 180 degrees
def unwrap_pattern_angle(ang):
    r = ang * 1
    ang0 = ang.flatten()[0]
    r[ang > ang0 + math.pi / 2] -= math.pi
    r[ang < ang0 - math.pi / 2] += math.pi
    return r


def print_phase_info(mod):
    for axis in [0, 1]:
        steps = np.diff(mod[axis::2, 3])
        steps[steps > np.pi] = -2 * np.pi + steps[steps > np.pi]
        print(f"axis {axis} steps: {-steps*180/np.pi}")

# Get the rows in mod for which the given axis is the dominant axis, get_axis_mod(mod,0) gives all X modulations
def get_axis_mod(mod, axis):
    dominant_axis = np.argmax(np.abs(mod[:,[0,1]]),1)
    return np.nonzero(dominant_axis==axis)[0]

# Flip kx,ky,phase
def flip_mod(mod_row):
    mod_row[[0,1,3]] = -1*mod_row[[0,1,3]]
    mod_row[3] += np.pi
    return mod_row

# Flip modulation directions so they all align with the first modulation pattern of a particular axis
def fix_mod_angles(mod):
    mod = 1*mod
    k = mod[:,[0,1]]
    for axis in range(2):
        am = get_axis_mod(mod,axis)
#        print(f"am0: {mod[am[0]]}")
        for i in np.arange(1,len(am)):
            if np.dot(k[am[0]],k[am[i]])<0: 
#                print(f"should flip axis {am[i]}: {mod[am[i]]}")
                mod[am[i]] = flip_mod(mod[am[i]])
    return mod

def draw_mod(allmod, showPlot=True, filename=None):
    fig,axes = plt.subplots(1,2)
    fig.set_size_inches(*figsize)
    for axis in range(2):
        axisname = ['X', 'Y']
        ax = axes[axis]
        indices = get_axis_mod(allmod,axis)
        freq = np.sqrt(allmod[indices[0],0]**2+allmod[indices[0],1]**2)
        period = 2*np.pi/freq
        x = np.linspace(0, period, 200)
        sum = x*0
        for i in indices:
            mod = allmod[i]
            q = (1+mod[2]*np.sin(x*freq-mod[3]) )*mod[4]
            ax.plot(x, q, label=f"Pattern {i}")
            sum += q
        ax.plot(x, sum, label=f'Summed {axisname[axis]} patterns')
        ax.legend()
        ax.set_title(f'{axisname[axis]} modulation')
        ax.set_xlabel('Pixels');ax.set_ylabel('Modulation intensity')
    fig.suptitle('Modulation patterns')
    if filename is not None: fig.savefig(filename)
    if not showPlot: plt.close(fig)
    return fig

def result_dir(path):
    dir, fn = os.path.split(path)
    return dir + "/results/" + os.path.splitext(fn)[0] + "/"


        
def load_mod(tiffpath):
    with open(os.path.splitext(tiffpath)[0]+"_mod.pickle", "rb") as pf:
        mod = pickle.load(pf)['mod']

        return mod
    
    
def density_filter(searchdist, mincount=5):
    def filterfn(lr : loc.LocResultList):
        counts = lr.count_neighbors(searchdist)
        def locfilter(i, locs):
            return counts[i] >= mincount
        lr.filter(locfilter)
        
    return filterfn


def save_spots_for_plots(path, spotlist: SpotList, mod, nspots=1000):
    data = (
            spotlist.IBg[:nspots],
            spotlist.rois[:nspots],
            spotlist.result.get_xyIBg()[:nspots],
            spotlist.result.get_roipos()[:nspots],
            mod)

    with open(os.path.splitext(path)[0]+"_selspots.pickle", "wb") as pf:
        pickle.dump(data, pf)
    


def print_mod(reportfn, mod, pattern_frames, pixelsize):
    for k in range(len(mod)):
        reportfn(f"Pattern {k}: kx={mod[k,0]:.4f} ky={mod[k,1]:.4f} Phase {mod[k,3]*180/np.pi:8.2f} Depth={mod[k,2]:5.2f} "+
               f"Power={mod[k,4]:5.3f} ")

    for ang in range(len(pattern_frames)):
        pat=pattern_frames[ang]
        depth = np.mean(mod[pat,2])
        phases = mod[pat,3]
        shifts = (np.diff(phases[-1::-1]) % (2*np.pi)) * 180/np.pi
        shifts[shifts > 180] = 360 - shifts[shifts>180]
        
        with np.printoptions(precision=3, suppress=True):
            reportfn(f"Angle {ang} shifts: {shifts} (deg) (patterns: {pat}). Depth={depth:.3f}")
    

    
def tiff_to_rois(path, num_patterns, cfg, debugMode):
    with SMLM(debugMode=debugMode) as smlm, Context(smlm) as ctx:
        imgshape = read_tiff.tiff_get_image_size(path)
        sigma = cfg['sigma']
        roisize = cfg['roisize']
        minIntensity = cfg['detectionMinIntensity']
        maxframes = cfg['maxframes'] if 'maxframes' in cfg else -1
        
        spotDetectorConfig = spotdetect.SpotDetectorConfig(sigma, roisize, minIntensity)

        calib = tiff_to_locs.create_calib_obj(cfg['gain'],cfg['offset'],imgshape,ctx)

        sf = SIMFLUX(ctx)        
        psf = sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma, num_patterns, roisize, 
                                             num_patterns, False)
        roipsf = PSF.CopyROI_Create(psf, ctx)
        roiqueue = PSF_Queue(roipsf, batchSize=1024)

        tiff_to_locs.process_movie(imgshape, spotDetectorConfig, calib, roiqueue, 
                           read_tiff.tiff_read_file(path, cfg['startframe'], maxframes), num_patterns)
        
        roiqueue.WaitUntilDone()
        
        r = roiqueue.GetResults()
        r.SortByID() # sort by frame numbers
        
        print(f"Filtering {len(r.estim)} spots...")
        minX = 2.1
        minY = 2.1
        r.FilterXY(minX,minY,roisize-minX-1, roisize-minY-1)
#        numbad = loc_sum.filter(lambda i,d: np.sum(d[:, [5, 6]], 1) < 4)  # CRLB X,Y < 4
#        crlb = r.CRLB()
#        r.Filter(np.where( (crlb[:,0]+crlb[:,1])<3 )[0])

        r.Filter(np.where(r.iterations<80)[0])
        
        plt.figure()
        plt.hist(r.iterations)
        plt.title('2D Gaussian iterations')

        r.rois = r.diagnostics.reshape((len(r.ids),*psf.samplesize))
        nframes =  np.max(r.ids)+1 if len(r.ids)>0 else 1
        print(f"Num spots: {len(r.estim)}. {len(r.estim) // nframes} spots/frame")
        return r,imgshape

def estimate_IBg(smlm, rois, roipos, sigma):
    print(f"Computing intensity/background per pattern for {len(rois)} spots." )
    with Context(smlm) as ctx:
        num_patterns=rois.shape[1]
        roisize = rois.shape[2]
                
        sf = SIMFLUX(ctx)
        mod = np.zeros((num_patterns, 5))
        psf = sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, num_patterns, simfluxEstim=False)

        q = PSF_Queue(psf, batchSize=1024)
        q.Schedule(rois,roipos=roipos,ids=np.arange(len(rois)))
        q.WaitUntilDone()
        
        r = q.GetResults()
        # Keep the same order as the original rois
        r.SortByID(isUnique=True)
        
    d = r.diagnostics.reshape((len(r.ids),num_patterns,4))
    IBg = d[:,:,[0,1]]
    IBg_crlb = d[:,:,[2,3]]
    return IBg,IBg_crlb

def process(path, cfg, pattern_frames, pixelsize, 
            freq_minmax,
            num_phase_bins = 10,
            plot_ffts = False,
            estimate_angle_bins = 1,
            drift_correct=False, 
            draw_silm_compare=False, 
            show_plots=True, 
            debugMode=False,
            spotfilter=['minfilter-firstlast', 30],
            filter_bg=None,
            fix_depths=False,
            resultsdir=None,
            unfiltered_smlm=False,
            fix_phase_shifts=None, mod=None): 
    
    pattern_frames = np.array(pattern_frames)
    num_patterns = pattern_frames.size
    
    if mod is None:
        mod = np.zeros((num_patterns,5))

    dir, fn = os.path.split(path)
    if resultsdir is None:
        resultsdir = result_dir(path)
    os.makedirs(resultsdir, exist_ok=True)
    resultprefix = resultsdir
        
    reportfile =  resultprefix + "report.txt"
    with open(reportfile,"w") as f:
        f.write("")
    
    def report(msg):
        with open(reportfile,"a") as f:
            f.write(msg+"\n")
        print(msg)
            
    report(f"Using spot detection threshold {cfg['detectionMinIntensity']}")
    report(f"Using PSF sigma {cfg['sigma']}")

    qr,imgshape = caching.read(path, 'rois_cache', (num_patterns,cfg), 
                 lambda path,_: tiff_to_rois(path, num_patterns, cfg, debugMode))

    if filter_bg is not None:
        qr.Filter(np.where(qr.estim[:,3] < filter_bg)[0])

    with SMLM(debugMode) as smlm:
        #loc_sum = loc.load(path, {**cfg, "sumframes": len(mod)}, smlm, tag="sum")
        
        loc_sum = loc.from_psf_queue_results(qr,cfg,[0,0,imgshape[1],imgshape[0]],path,1)
        
#        loc_sum.save_picasso_hdf5(resultprefix+"g2d.hdf5")
        
        IBg,IBg_crlb = estimate_IBg(smlm, qr.rois, qr.roipos, cfg['sigma'])
        
        spotlist = SpotList(loc_sum, pixelsize=pixelsize, outdir=resultsdir, rois=qr.rois, IBg=IBg, IBg_crlb=IBg_crlb)
        
        median_crlb_x = np.median(loc_sum.get_crlb()[:,0])
        median_I = np.median(loc_sum.get_xyI()[:,2])

        nanibg = np.nonzero(np.sum(np.isnan(spotlist.IBg),(1,2)))[0]
        report(f"#nan ibg: {len(nanibg)}. g2d mean I={median_I:.1f}. mean crlb x {median_crlb_x:.4f}")

        fr = np.arange(len(loc_sum.frames))

        if estimate_angle_bins is not None:
            angles, pitch = spotlist.estimate_angle_and_pitch(
                pattern_frames, 
                frame_bins=np.array_split(fr, estimate_angle_bins), 
                smlm=smlm,
                freq_minmax=freq_minmax
            )

            print("Pitch and angle estimation: ")
            for k in range(len(pattern_frames)):
                angles[angles[:, k] > 0.6 * np.pi] -= np.pi  # 180 deg to around 0
                angles[:, k] = unwrap_pattern_angle(angles[:, k])
                angles_k = angles[:, k]
                pitch_k = pitch[:, k]
                report(f"Angle {k}: { np.rad2deg(np.mean(angles_k)) :7.5f} [deg]. Pitch: {np.mean(pitch_k)*pixelsize:10.5f} ({2*np.pi/np.mean(pitch_k):3.3f} [rad/pixel])")

                freq = 2 * np.pi / np.mean(pitch_k)
                kx = np.cos(np.mean(angles_k)) * freq
                ky = np.sin(np.mean(angles_k)) * freq
                mod[pattern_frames[k], 0] = kx
                mod[pattern_frames[k], 1] = ky
                            
        else:
            q = mod[:, 0] + 1j * mod[:, 1]
            pitch = 2 * np.pi / np.abs(q)
            angles = np.angle(q)
            print(angles)
            
        #mod = fix_mod_angles(mod) 

        phase,depth,power = mod[:,3],mod[:,2],mod[:,4]
        phase_all=phase
        depth_all=depth
        mod_info = {"mod": mod, "pitch": pitch, "angles": angles, "phase": phase, "depth": depth, 'power': power}
            
        if num_phase_bins is not None:
            def phase_depth_estim(method):
                phase, depth, power = spotlist.estimate_phase_and_depth(mod, np.array_split(fr, num_phase_bins), method=method)
                phase_all, depth_all, power_all = spotlist.estimate_phase_and_depth(mod, [fr], method=method)
        
                axis = lambda k: ['x','y'][int(np.abs(mod[k,0])>np.abs(mod[k,1]))]
                
                fig = plt.figure(figsize=figsize)
                styles = [":", "-"]
                for k in range(len(mod)):
                    plt.plot(unwrap_angle(phase[:, k]) * 180 / np.pi, styles[k % 2], label=f"Phase {k} ({axis(k)})")
                plt.legend()
                plt.title(f"Phases for {fn}")
                plt.xlabel("Timebins"); plt.ylabel("Phase [deg]")
                plt.grid()
                plt.tight_layout()
                fig.savefig(resultprefix + "phases.png")
                if not show_plots: plt.close(fig)
        
                fig = plt.figure(figsize=figsize)
                for k in range(len(mod)):
                    plt.plot(depth[:, k], styles[k%2], label=f"Depth {k} ({axis(k)})")
                plt.legend()
                plt.title(f"Depths for {fn}")
                plt.xlabel("Timebins"); plt.ylabel("Modulation Depth")
                plt.grid()
                plt.tight_layout()
                fig.savefig(resultprefix + "depths.png")
                if not show_plots: plt.close(fig)
        
                fig = plt.figure(figsize=figsize)
                for k in range(len(mod)):
                    plt.plot(power[:, k], styles[k%2], label=f"Power {k} ({axis(k)})")
                plt.legend()
                plt.title(f"Power for {fn}")
                plt.xlabel("Timebins"); plt.ylabel("Modulation Power")
                plt.grid()
                plt.tight_layout()
                fig.savefig(resultprefix + "power.png")
                if not show_plots: plt.close(fig)
        
                # Update mod
                phase_std = np.zeros(len(mod))
                for k in range(len(mod)):
                    ph_k = unwrap_angle(phase[:, k])
                    mod[k, 3] = phase_all[0, k]
                    mod[k, 2] = depth_all[0, k]
                    mod[k, 4] = power_all[0, k]
                    phase_std[k] = np.std(ph_k)

                s=np.sqrt(num_phase_bins)
                for k in range(len(mod)):
                    report(f"Pattern {k}: Phase {mod[k,3]*180/np.pi:8.2f} (std={phase_std[k]/s*180/np.pi:6.2f}) "+
                           f"Depth={mod[k,2]:5.2f} (std={np.std(depth[:,k])/s:5.3f}) "+
                           f"Power={mod[k,4]:5.3f} (std={np.std(power[:,k])/s:5.5f}) ")

#            report("Estimating phase, depth, power with these pitch values:")
 #           for k in range(len(mod)):
  #              report(f"Pattern {k}: kx={mod[k,0]:.8f} ky={mod[k,1]:.8f} ")

            phase_depth_estim(method=1)
            mod=spotlist.refine_pitch(mod, smlm, spotfilter, plot=True)[2]

            if fix_phase_shifts:
                report(f'Fixing phase shifts to {fix_phase_shifts}' )
                phase_shift_rad = fix_phase_shifts / 180 * np.pi
                mod[::2,3] = mod[0,3] + np.arange(len(mod)//2) * phase_shift_rad
#                mod[::2,3] += 0.1
                mod[1::2,3] = mod[1,3] + np.arange(len(mod)//2) * phase_shift_rad
            
                mod=spotlist.refine_pitch(mod, smlm, spotfilter, plot=True)[2]

            info_mat = {**mod_info, 'phase': phase_all, 'depth': depth_all}
            sio.savemat(resultprefix + "info.mat", info_mat)

            mod_info = {"mod": mod, "pitch": pitch, "angles": angles, "phase": phase, "depth": depth, 'power': power}
            with open(os.path.splitext(path)[0]+"_mod.pickle", "wb") as df:
                pickle.dump(mod_info, df)

        for angIndex in range(len(pattern_frames)):
            mod[pattern_frames[angIndex], 4] = np.mean(mod[pattern_frames[angIndex], 4])
            # Average modulation depth
            mod[pattern_frames[angIndex], 2] = np.mean(mod[pattern_frames[angIndex], 2])

        mod[:,4] /= np.sum(mod[:,4])

        if fix_depths:
            report(f'Fixing modulation depth to {fix_depths}' )
            mod[:,2]=fix_depths

        report("Final modulation pattern parameters:")
        print_mod(report, mod, pattern_frames, pixelsize)

        moderrs = spotlist.compute_modulation_error(mod, spotfilter)
        spotlist.plot_moderr_vs_intensity(mod, spotfilter)
        errs = spotlist.compute_modulation_chisq(mod, pattern_frames, spotfilter, plot=False)#, frames=np.arange(fr))
        report(f"Modulation qualities per axis:")
        for k in range(len(pattern_frames)):
            ind = pattern_frames[k]
            for step in range(len(ind)):
                report(f"\tAxis {k}, step {step}: {errs[ind[step]]:.5f}")
        
        report(f"RMS moderror: {np.sqrt(np.mean(moderrs**2)):.3f}")

        spotlist.plot_modulation_chisq_timebins(mod, pattern_frames, spotfilter, 50)        

        if len(pattern_frames)==2: # assume XY modulation
            draw_mod(mod, False, resultprefix + "patterns.png")

        spotlist.silm_bias_plot2D(mod, smlm, spotfilter, tag='')
#        spotlist.plot_intensity_variations(mod, minfilter, pattern_frames)

        if plot_ffts:
            spotlist.generate_projections(mod, 4, smlm)
            spotlist.plot_proj_fft()

        med_sum_I = np.median(spotlist.IBg[:,:,0].sum(1))
        lowest_power = np.min(mod[:,4])
        depth = mod[np.argmin(mod[:,4]),2]
        median_intensity_at_zero = med_sum_I * lowest_power * (1-depth)
        report(f"Median summed intensity: {med_sum_I:.1f}. Median intensity at pattern zero: {median_intensity_at_zero:.1f}")
        report(f"Using spot filter: {spotfilter}. ")

        for k in range(len(mod)):
            png_file= f"{spotlist.outdir}patternspots{k}.png"
            print(f"Generating {png_file}...")
            spotlist.draw_spots_in_pattern(png_file, mod, 
                                       k, tiffname=os.path.split(fn)[1], numpts= 2000, spotfilter=spotfilter)
            spotlist.draw_spots_in_pattern(f"{spotlist.outdir}patternspots{k}.svg", mod, 
                                       k, tiffname=os.path.split(fn)[1], numpts= 2000, spotfilter=spotfilter)

        spotlist.draw_axis_intensity_spread(pattern_frames, mod, spotfilter)
        
        print(f"Running simflux fits...")
        # g2d_results are the same set of spots used for silm, for fair comparison
        silm_results,g2d_results = spotlist.silm(mod,smlm, spotfilter=spotfilter,
                                                 draw_silm_compare=draw_silm_compare)
        
        border = 2.1
        roisize = cfg['roisize']
        num_removed, indices = silm_results.filter_inroi(border, border, roisize-border-1, roisize-border-1)
        report(f"Removing {num_removed} ({100*num_removed/len(spotlist.IBg)}%) unconverged SIMFLUX fits")
        #g2d_results.filter_indices(indices)
        
        if unfiltered_smlm:
            g2d_results=spotlist.result

        drift = None

        if drift_correct is not None:
            if type(drift_correct) == bool:
                drift_correct = 250

            framelist_half = np.nonzero(np.arange(len(loc_sum.frames)) > len(loc_sum.frames)//2)[0]
            spots_half = loc_sum.get_frame_spot_indices(framelist_half)
            
#            loc_sum.data[spots_half,DataColumns.X] += 2
#            loc_sum.get_xy()[spots_half,0] += 2
                
            drift = loc_sum.drift_estimate_rcc(smlm, drift_correct, scale=2, useIntensities=False, maxdrift=3)
            loc_sum.plot_drift(f"Drift estimate for {fn}").savefig(resultprefix + "drift.png")
            
            np.savetxt(resultprefix+"drift.txt",loc_sum.drift)

            silm_results.save_picasso_hdf5(resultprefix+"simflux-nodriftc.hdf5")
            g2d_results.save_picasso_hdf5(resultprefix+"g2d-nodriftc.hdf5")

            sio.savemat(resultprefix+"drift.mat", {'drift': loc_sum.drift})
            silm_results.apply_drift_correction(drift)
            g2d_results.apply_drift_correction(drift)
        elif drift_correct:
            drift_fn = os.path.split(path)[0]+ "/" + drift_correct
            g2d_results.drift_correct_from_file(drift_fn, 1)#len(mod))
            silm_results.drift_correct_from_file(drift_fn, 1)#len(mod))
            silm_results.plot_drift('drift' )

        elif drift_correct == "nn":
            raise NotImplementedError()
            
        silm_results.save_picasso_hdf5(resultprefix+"simflux.hdf5")
        g2d_results.save_picasso_hdf5(resultprefix+"g2d.hdf5")
        
        spotlist.save_IBg(resultprefix+'spots_IBg.hdf5',mod)
        
        crlb = g2d_results.get_crlb()
        maxdist = 1.5*np.sqrt(np.sum(np.mean(crlb[:,[0,1]],0)**2))
        report(f"Linking localizations (max dist: {maxdist:.2f} pixels)...")
        silm_linked = silm_results.link_locs(smlm,maxdist)
        g2d_linked = g2d_results.link_locs(smlm,maxdist)
        
        silm_linked.save_picasso_hdf5(resultprefix+"simflux-linked.hdf5")
        g2d_linked.save_picasso_hdf5(resultprefix+"g2d-linked.hdf5")

#        silm_data = silm_results.save_csv_with_drift(resultprefix + "simflux.csv")
 #       g2d_data = g2d_results.save_csv_with_drift(resultprefix + "g2d.csv")
  #      sio.savemat(resultprefix + "localizations.mat", {"simflux": silm_data, "g2d": g2d_data})
                                    
        image = loc_sum.draw(2,smlm)
        
#        save_spots_for_plots(path, spotlist,mod)
        
        return spotlist, image, mod, silm_results, g2d_results

