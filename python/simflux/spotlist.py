
import numpy as np
import utils.localizations as loc
from smlmlib.gaussian import Gaussian
from smlmlib.psf_queue import PSF_Queue
from smlmlib.context import Context
from smlmlib.simflux import SIMFLUX
from smlmlib.base import SMLM
import tqdm
import tifffile
import matplotlib.pyplot as plt
import scipy.stats
import utils.caching as cache
import sys
import utils.findpeak1D as findpeak1D
from smlmlib import util as su
import utils.dftpeak as dftpeak
import utils.fitpoints2D as fitpoints2D
import pickle

figsize = (8,6)

def polyfit_ransac(x,y,deg,filterfrac=0.5,iterations=3):
    indices = np.arange(len(x))    

    for k in range(iterations):    
        c = np.polyfit(x[indices],y[indices],deg)
#        print(f"{k}: {c}")
        if k == iterations-1:
            break
        
        errs = (np.polyval(c, x) - y)**2
        sorted_errs = np.argsort(errs)
        indices = sorted_errs[:int(len(sorted_errs)*filterfrac)]

    return c


class SpotList:
    IBg = None
    num_patterns = 0
    result = None  # Type: loc.LocResultList
    outdir = ""  # output directory for plots etc
    rois = None

    def __init__(self, r: loc.LocResultList, pixelsize, outdir, IBg=None, IBg_crlb=None,rois=None):
        self.result = r
        self.pixelsize = pixelsize
        self.outdir = outdir
        self.rois = rois
        self.IBg = IBg
        self.IBg_crlb = IBg_crlb
    
    def get_xyI(self, ep, normalize=False):
        xyI = self.result.get_xyI() * 1
        xyI[:, 2] = self.IBg[:, ep, 0]
        if normalize:
            xyI[:,2] /= np.sum(self.IBg[:,:,0], 1)
        return xyI

    def generate_projections(self, mod, scale, smlm):
        w = self.result.imgshape[0]
        self.proj = np.zeros((len(mod), w * scale))
        self.proj_shift = np.zeros(len(mod))
        
        with Context(smlm) as ctx:
            for k in range(len(mod)):
                m = mod[k]
                ang = np.arctan2(m[1], m[0])
                xyI = self.get_xyI(k)
                xyI = xyI[~np.isnan(xyI[:, 2]), :]
                projected, shift = SIMFLUX(ctx).ProjectPoints(xyI, w * scale, scale, [ang])
                self.proj[k] = projected[0]
                self.proj_shift[k] = shift

    def plot_proj_fft(self):
        from matplotlib.backends.backend_pdf import PdfPages

        pp = PdfPages(f"{self.outdir}fft-spectra.pdf")
        for k in range(self.num_patterns):
            fig = plt.figure()
            scale = len(self.proj[k]) / self.result.imgshape[0]
            freq = np.fft.fftshift(np.fft.fftfreq(len(self.proj[k])))
            plt.plot(freq * scale, np.fft.fftshift(np.abs(np.fft.fft(self.proj[k] ** 2))))
            plt.title(f"FFT for pattern {k}")
            # plt.xlabel("pixel^-1");
            plt.tight_layout()
            pp.savefig(fig)
            plt.close(fig)
        pp.close()
        
    def _estimate_angle_and_pitch_dft(self, pattern_indices, frames, smlm:SMLM, freq_minmax):
        w,h=self.result.get_image_size()

        z=6
        ft_sum = np.zeros((h*z,w*z))
        for ep in pattern_indices:
            img = self.render_sr_image_per_pattern(ep, frames, sr_factor=z,smlm=smlm)
            ft_img = smlm.FFT2(img)
            ft_sum += np.abs(ft_img)
        ft_sum = np.fft.fftshift(ft_sum)
        
        freq = np.fft.fftshift( np.fft.fftfreq(h*z) )*z*2*np.pi
        XFreq, YFreq = np.meshgrid(freq,freq)
        Freq = np.sqrt(XFreq**2+YFreq**2)
        
#        plt.imsave(self.outdir + f"pattern-{pattern_indices}-FFT-mask.png", mask)
        
        saved_img = ft_sum*1
        saved_img[Freq<freq_minmax[0]] = 0
        plt.imsave(self.outdir + f"pattern-{pattern_indices}-FFT-ep{pattern_indices}.png", saved_img)

        mask = (Freq>freq_minmax[0]) & (Freq<freq_minmax[1])
        ft_sum[~mask] = 0
        
        max_index = np.argmax(ft_sum)
        max_indices = np.unravel_index(max_index, ft_sum.shape)
                
        yx = freq[list(max_indices)]
        xy = self._find_dft2_peak(yx[[1,0]], pattern_indices, frames, smlm)
        
        ang = np.arctan2(xy[1],xy[0])
        freq = np.sqrt(xy[0]**2+xy[1]**2)
        return ang,2*np.pi/freq
        
    def _find_dft2_peak(self, xy, pattern_indices, frames, smlm:SMLM):
        indices = np.concatenate([self.result.frames[i] for i in frames])
        S = 0.02
        kxrange = np.linspace(xy[0]-S, xy[0]+S, 50)
        kyrange = np.linspace(xy[1]-S, xy[1]+S, 50)

        img = np.zeros((len(kyrange),len(kxrange)))
        for ep in pattern_indices:
            xyI = self.get_xyI(ep)[indices]
            sig = dftpeak.dft_points(xyI, kxrange, kyrange, smlm, useCuda=True)
            img += np.abs(sig**2)
            
#        plt.figure()
 #       plt.imshow(img)
        plt.imsave(self.outdir + f"pattern-{pattern_indices}-FFT-peak.png", img)
        
        peak = np.argmax(img)
        peak = np.unravel_index(peak, img.shape)

        kx_peak = findpeak1D.quadraticpeak(img[peak[0], :], kxrange, npts=11, plotTitle=None)#='X peak')
        ky_peak = findpeak1D.quadraticpeak(img[:, peak[1]], kyrange, npts=11, plotTitle=None)#='Y peak')

        return kx_peak, ky_peak

    def estimate_angle_and_pitch(self, pattern_frames, frame_bins, smlm,freq_minmax):
        angle = np.zeros((len(frame_bins), len(pattern_frames)))
        pitch = np.zeros((len(frame_bins), len(pattern_frames)))
        for b in range(len(frame_bins)):
            sys.stdout.write(f"Estimating angle/pitch {b}/{len(frame_bins)}")
            for m in tqdm.trange(len(pattern_frames)):
                angle[b, m], pitch[b, m] = self._estimate_angle_and_pitch_dft(
                    pattern_frames[m], frame_bins[b], smlm,
                    freq_minmax=freq_minmax
                )
            sys.stdout.write("\n")
        return angle, pitch
    
    def estimate_phase_and_depth(self, mod_list, frame_bins, method=0):
        phase = np.zeros((len(frame_bins), len(mod_list)))
        depth = np.zeros((len(frame_bins), len(mod_list)))
        relint = np.zeros((len(frame_bins), len(mod_list)))

        for b, fr_bin in enumerate(frame_bins):
            for m, mod in enumerate(mod_list):
                pattern_frames = np.arange(0,len(mod),2) + (m % 2)
                if method == 0:
                    phase[b, m], depth[b, m], relint[b, m] = self._estimate_phase_and_depth(mod, m, pattern_frames, fr_bin)
                else:
                    phase[b, m], depth[b, m], relint[b, m] = self._estimate_phase_and_depth2(mod, m, pattern_frames, fr_bin)

            #print(power[b,:])                    
            relint[b] /= np.sum(relint[b,:])
        return phase, depth, relint

    # Re-estimate the phase and depth of a single modulation pattern
    def _estimate_phase_and_depth(self, mod, ep, frames):
        indices = np.concatenate([self.result.frames[i] for i in frames])
        xyI = self.get_xyI(ep)[indices]

        spotPhaseField = mod[0] * xyI[:, 0] + mod[1] * xyI[:, 1]

        I0 = np.sum(xyI[:, 2])
        Iw = np.sum(xyI[:, 2] * np.exp(-1j * spotPhaseField))

        phase = -np.angle(Iw / I0 * 1j)
        depth = 2 * np.abs(Iw / I0)
        relint = np.mean(xyI[:,2] / self.result.get_xyI()[indices,2])
        return phase, depth, relint
    
    
    def plot_intensity_variations(self,mod,minfilter,pattern_frames):
        
        xyI = self.result.get_xyI()
        sel = np.nonzero((self.IBg[:,0,0] > minfilter ) & 
                         (self.IBg[:,-1,0] > minfilter) & 
                         (xyI[:,2] < 3000) & (xyI[:,2] > 500)) [0]

        # compute expected modulation matrix [spotidx, pattern]
        xyI = self.result.get_xyI()[sel,np.newaxis]
        mod_ = mod[np.newaxis]
        exc = mod_[:,:,4]*(1+mod_[:,:,2]*np.sin(xyI[:,:,0]*mod_[:,:,0]+xyI[:,:,1]*mod_[:,:,1]-mod_[:,:,3]))
        expectedIntensity = exc * xyI[:,:,2]
        
        expectedRelErr = self.IBg_crlb[sel,:,0]/expectedIntensity
        measuredRelErr = np.abs((self.IBg[sel,:,0]-expectedIntensity) / expectedIntensity)

        if len(pattern_frames)==2:
            axname = ['Y', 'X']
        else:
            axname = [f"{i}" for i in range(len(pattern_frames))]
        for axis in range(len(pattern_frames)):
            fr = pattern_frames[axis]
            plt.figure()
            plt.hist([expectedRelErr[:,fr].flatten(), measuredRelErr[:,fr].flatten()],label=['Expected rel err: CRLB(N_k) / (P_k*N)', 'Measured err: |N_k - P_k*N|/(P_k*N)'], range=(0,2), bins=50)
            plt.title(f'Measured vs expected modulated intensity variations - {axname[axis]}')
            plt.xlabel("Relative error: (err-expected)/expected")
            plt.legend()
            plt.savefig(f"{self.outdir}/rel-intensity-variations-ax{axis}.png")

            """            
            plt.figure()
            plt.scatter(expectedIntensity[:,fr].flatten(), self.IBg[sel][:,fr,0].flatten(), s=1,label='Measured intensity')
            plt.xlabel(f"Expected intensity [photons] - {axname[axis]}")
            plt.legend()
            
            plt.figure()
            plt.scatter(expectedIntensity[:,fr].flatten(), np.abs(self.IBg[sel][:,fr,0]-expectedIntensity[:,fr]).flatten(), s=1,label="|N_k - P_k*N|")
            plt.scatter(expectedIntensity[:,fr].flatten(), self.IBg_crlb[sel][:,fr,0].flatten(), s=1,label="CRLB(N_k)")
            plt.legend()
            plt.xlabel("Expected intensity [photons]")
            plt.title(f'CRLB vs intensity error - {axname[axis]}')
            """
        
        with open(f"{self.outdir}/intensities.pickle", "wb") as df:
            pickle.dump({'exc': exc, 'xyI': xyI, 
                         'expectedIntensity': expectedIntensity, 
                         'expectedRelErr': expectedRelErr,
                         'measuredRelErr': measuredRelErr}, df)
        
#        mod[np.newaxis,:,0]*xyI[:,np.newaxis,0]+
        

    def _estimate_phase_and_depth2(self, mod, ep, pattern_frames, frames):
        indices = np.concatenate([self.result.frames[i] for i in frames])
        min_intensity = 30
#        sel = np.nonzero(np.min(self.IBg[indices,:,0],1) > min_intensity) [0]
        sel = np.nonzero((self.IBg[indices,0,0] > min_intensity ) & 
                         (self.IBg[indices,-1,0] > min_intensity )) [0]
 
        #indices = indices[sel]

#        sumI = np.sum(self.IBg[indices][: , pattern_frames, 0],1)
        sumI = np.sum(self.IBg[indices][: , :, 0],1)
        xyI = self.result.get_xyI()[indices] * 1
        
        xyI_modulated = xyI*1
        xyI_modulated[:,2] = self.IBg[indices,ep,0]

        I = xyI[:,2]*1
        spotPhaseField = mod[0] * xyI[:, 0] + mod[1] * xyI[:, 1]
        
        basefreq = np.linspace(-1, 1, 3)
        
        weights = np.ones(len(I))

        for it in range(2):

            # DFT on modulated and unmodulated data
            f = [ np.sum(weights * xyI_modulated[:,2] * np.exp(-1j * spotPhaseField * k)) for k in basefreq ]
            B = [ np.sum(weights * I * np.exp(-1j * spotPhaseField * k)) for k in basefreq - 1]
            A = [ np.sum(weights * I * np.exp(-1j * spotPhaseField * k)) for k in basefreq ]
            C = [ np.sum(weights * I * np.exp(-1j * spotPhaseField * k)) for k in basefreq + 1]
            
            # Solve Ax = b
            nrows = len(basefreq)
            M = np.zeros((nrows,3), dtype=np.complex)
            b = np.zeros(nrows, dtype=np.complex)
            for i in range(nrows):
                M[i] = [ B[i], C[i], A[i] ] 
                b[i] = f[i]
                            
            # Actually x[1] is just the complex conjugate of x[0], 
            # so it seems a solution with just 2 degrees of freedom is also possible
#            x, residual, rank, s = np.linalg.lstsq(M,b,rcond=None)
            x = np.linalg.solve(M,b)
            b,c,a = x
           
            depth = np.real(2*np.abs(b)/a)
            phase = - np.angle(b*1j)
            relint = np.real(2*a)/2  

            q = relint * (1+depth*np.sin(mod[0]*xyI[:,0]+mod[1]*xyI[:,1]-phase))            
            xyQ = xyI_modulated*1
            xyQ[:,2] /= sumI

            errs = np.abs(xyQ[:,2]-q)
            
           # print(f"{it}: err sum: {np.sum(errs)}. phase={phase}")
 #           plt.figure()
#            plt.hist(errs)
            median_err = np.percentile(errs,50)# median(errs)
            weights = errs < median_err
            

#        print(f"ep{ep}. depth={depth}, phase={phase}")
        return phase, depth, relint

    def silm(self,mod,smlm:SMLM, spotfilter, draw_silm_compare=False):
        sel = self.get_filtered_spots(spotfilter,mod)
        rois = self.rois[sel]

        with Context(smlm) as ctx:
            sf = SIMFLUX(ctx)
            roisize = self.result.cfg['roisize']
            psf = sf.CreateSIMFLUX2D_Gauss2D_PSF(self.result.cfg['sigma'], mod, 
                                                 roisize, len(mod),
                                                 simfluxEstim=True,defineStartEnd=False)

            roipos = self.result.get_roipos()[sel]
            sf_roipos=np.zeros((len(rois),3),dtype=np.int32)
            sf_roipos[:,[2,1]] = roipos
            
            queue = PSF_Queue(psf,numStreams=1)
            queue.Schedule(rois,roipos=sf_roipos,ids=np.arange(len(rois)))
            queue.WaitUntilDone()
            sf_results = queue.GetResults()
            sf_results.SortByID(isUnique=True)
            queue.Destroy()
            
            crlb = sf_results.CRLB()
            est = sf_results.estim

        data = self.result.data[sel, :] * 1
        data[:, [loc.DataColumns.X, loc.DataColumns.Y]] = est[:, [0, 1]] + roipos
        data[:, [loc.DataColumns.I, loc.DataColumns.BG]] = est[:, [2, 3]]
        data[:, [loc.DataColumns.CRLB_X, loc.DataColumns.CRLB_Y, loc.DataColumns.CRLB_I, loc.DataColumns.CRLB_BG]] = crlb
        data[:, loc.DataColumns.ITERATIONS] = sf_results.iterations
        data[:, loc.DataColumns.LL] = sf_results.ll
                
        if draw_silm_compare:
            import matplotlib.ticker as ticker
            # matplotlib gets really slow with large point counts
            draw_sel = sel[:50000 if len(sel)>50000 else len(sel)]*1
            gxy = self.result.get_xyI()[draw_sel]
            fig,axes=plt.subplots(2,len(mod)//2,sharey=True,sharex=True)
            for k in range(self.num_patterns):
                ax = axes[k%2,k//2]
                # The gaussian localizations projected onto the modulation axis
                k_len = np.sqrt(mod[k,0]**2+mod[k,1]**2)
                g2d_c = mod[k,0] * gxy[:,0] + mod[k,1] * gxy[:,1]
                # The SILM localizations projected onto the modulation axis
                silm_c = mod[k,0] * data[:len(draw_sel),loc.DataColumns.X] + mod[k,1] * data[:len(draw_sel),loc.DataColumns.Y]
                phases = ( (g2d_c - mod[k,3]) % (2*np.pi) ) / np.pi
                ax.scatter( (g2d_c - silm_c) / k_len, phases,s=0.5,marker='.')
                ax.set_xlim(-2,2)
                ax.set_title(f'[{k}]: kx={mod[k,0]:.2f},ky={mod[k,1]:.2f}')
                if k%2!=0: ax.set_xlabel('[pixels]'); 
                ax.set_ylabel(f'Phase in pattern {k}')
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
            fig.suptitle("Difference between Gaussian and SILM localization mapped to position in pattern")
            fig.savefig(self.outdir + "phase-bias.png")
            plt.close(fig)
        
        return (loc.LocResultList(data,self.result.cfg, self.result.get_area(),src_indices=sel),
                loc.LocResultList(self.result.data[sel,:]*1,self.result.cfg, self.result.get_area(),src_indices=sel))
    
    
    

    def refine_pitch(self, mod, smlm:SMLM, spotfilter, plot=False):
        mod=mod*1
        
        # Get the rows in mod for which the given axis is the dominant axis, get_axis_mod(mod,0) gives all X modulations
        def get_axis_mod(mod, axis):
            dominant_axis = np.argmax(np.abs(mod[:,[0,1]]),1)
            return np.nonzero(dominant_axis==axis)[0]

        xmod = get_axis_mod(mod,0)
        ymod = get_axis_mod(mod,1)
        
        kx = np.mean(mod[xmod,0])
        ky = np.mean(mod[ymod,1])
        
        def estim_shift(mod,plot=False):
            silm_d,g2d_d = self.silm(mod,smlm,spotfilter)
            shift = silm_d.get_xyI()[:,[0,1]] - g2d_d.get_xyI()[:,[0,1]]
            pos=g2d_d.get_xyI()
            cx = np.polyfit(pos[:,0],shift[:,0],1)
            cy = np.polyfit(pos[:,1],shift[:,1],1)
            #cx = polyfit_ransac(pos[:,0],shift[:,0],1,0.4)
           # cy = polyfit_ransac(pos[:,1],shift[:,1],1,0.4)
 
            if plot:
                # plot a random subset of the points
                np.random.seed(0)
                indices = np.arange(len(pos))# np.nonzero(np.min( self.IBg[:,:,0],1) > min_intensity)[0]
                np.random.shuffle(indices)
                indices = indices[:2000]
                                
                w,h=self.result.get_image_size()
                figsize=(8,6)
                fig,axes=plt.subplots(2,1,figsize=figsize)
                px = np.polyfit(pos[indices,0],shift[indices,0],1)
                py = np.polyfit(pos[indices,1],shift[indices,1],1)

                sx=self.pixelsize/1000
                sy=self.pixelsize
                
                axes[0].scatter(sx*pos[indices,0],sy*shift[indices,0],s=3)
                axes[0].plot(np.linspace(0,w)*sx,np.polyval(px,np.linspace(0,w))*sy, 'g-')
                axes[0].set_title(f'X slope ({cx[0]*kx:.5f} rad/pixel), offset ({cx[1]:.3f})')

                axes[0].set_title('Difference between SIMFLUX and SMLM in X')
                axes[0].set_xlabel('X position in FOV [um]')
                axes[0].set_ylabel('Difference [nm]')

                axes[0].set_ylim(-50,50)
                axes[1].scatter(sx*pos[indices,1],sy*shift[indices,1],s=3)
                axes[1].plot(np.linspace(0,h)*sx,np.polyval(py,np.linspace(0,h))*sy,'g-')
                axes[1].set_title(f'Y shift ({cy[0]*ky:.5f} rad/pixel), offset ({cy[1]:.3f}))')
                axes[1].set_title('Difference between SIMFLUX and SMLM in Y')
                axes[1].set_xlabel('Y position in FOV [um]')
                axes[1].set_ylabel('Difference [nm]')
                axes[1].set_ylim(-50,50)
                fig.tight_layout()
                fig.savefig(self.outdir+"fov-shift.png")
                fig.savefig(self.outdir+"fov-shift.svg")
                fig.show()
#                plt.close(fig)
            
            return silm_d,g2d_d,cx,cy
        
        for k in range(2):            
            silm_d,g2d_d,cx,cy=estim_shift(mod,plot and k==0)
            # cx[0] is shifted pixel / pixel
            # cx[0]*kx shifted radians / pixel, iaw frequency change

            # Adjust pitch
            print(f"Adjust kx: {cx[0]*kx:.4f}, ky: {cy[0]*ky:.4f}. phase shift x: {cx[1]*kx:.4f}, phase shift y: {cy[1]*ky:.4f}")
            mod[xmod,0] += cx[0]*kx
            mod[ymod,1] += cy[0]*ky

            mod[xmod,3] -= cx[1]*kx
            mod[ymod,3] -= cy[1]*ky
            
        return silm_d,g2d_d,mod

    def silm_bias_plot2D(self, mod, smlm:SMLM, spotfilter, tag):
        silm_d,g2d_d = self.silm(mod,smlm,spotfilter=spotfilter)
        shift = silm_d.get_xyI()[:,[0,1]] - g2d_d.get_xyI()[:,[0,1]]
        pos = g2d_d.get_xyI()

        rms=np.sqrt(np.mean(shift**2,0))
        print(f"RMS x shift={rms[0]*self.pixelsize}, y shift={rms[1]*self.pixelsize}")

        shape = self.result.get_image_size()
        w = 100

        print(f"running 2D bias fit over {len(pos)} points")

        gridshape = [w,w]        

#        statfn = 'median'
        
        def median_mincount(x, minsize):
            if len(x) < minsize:
                return np.nan
            return np.median(x)
        
        def rms_mincount(x,minsize):
            if len(x)<minsize:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        statfn = lambda x: median_mincount(x,20)
        metric='Median bias'
 #       statfn = lambda x: rms_mincount(x,20)
#        metric='RMS'
        scatter=True
         
        clustergrid=[w,w]
        xysx = fitpoints2D.hist2d_statistic_points(pos[:,0],pos[:,1],shift[:,0], shape, clustergrid, statistic=statfn)
        xysy = fitpoints2D.hist2d_statistic_points(pos[:,0],pos[:,1],shift[:,1], shape, clustergrid, statistic=statfn)
        
        rms_x=np.sqrt(np.mean(xysx[:,2]**2))
        rms_y=np.sqrt(np.mean(xysy[:,2]**2))
        print(f"RMS clusters x shift={rms_x*self.pixelsize}, y shift={rms_y*self.pixelsize}")

        print(f"2D bias plot: plotting {len(xysx)} clusters")
        
        plt.figure(figsize=figsize)
        plt.hist([ xysx[:,2]*self.pixelsize, xysy[:2]*self.pixelsize], label=[ 'X bias','Y bias'], range=[-50,50], bins=50)
        plt.title('SMLM-SIMFLUX difference histogram')
        plt.xlabel('Mean difference between SMLM and SIMFLUX within one cluster [nm]')
        plt.legend()
        plt.savefig(self.outdir + tag+"bias-hist.svg")
        plt.savefig(self.outdir + tag+"bias-hist.png")
        
        xysx[:,2] = np.clip(xysx[:,2],-1,1)
        xysy[:,2] = np.clip(xysy[:,2],-1,1)
                
        img_x,coeff_x = fitpoints2D.fitlsq(xysx[:,0], xysy[:,1], xysx[:,2], shape, gridshape)
        img_y,coeff_y = fitpoints2D.fitlsq(xysy[:,0], xysy[:,1], xysy[:,2], shape, gridshape)

        plt.figure(figsize=figsize)
        plt.set_cmap('viridis')
#        plt.imshow(img_x, origin='lower',extent=[0,shape[1],0,shape[0]])
        s=self.pixelsize/1000
        sc=self.pixelsize
        if scatter: plt.scatter(s*xysx [:,0],s*xysx [:,1],c=sc*xysx [:,2])
        cb=plt.colorbar()
        cb.set_label('Bias in X [nm]')
        plt.clim([-20,20])
        plt.xlabel(u'X position in FOV [um]')
        plt.ylabel(u'Y position in FOV [um]')
        plt.title(f'{metric} in X over FOV')
        plt.savefig(self.outdir + tag+"x-bias-fov.png" )
        plt.savefig(self.outdir + tag+"x-bias-fov.svg" )
        
        plt.figure(figsize=figsize)
 #       plt.imshow(img_y, origin='lower',  extent=[0,shape[1],0,shape[0]])
        if scatter: plt.scatter(s*xysy[:,0],s*xysy[:,1],c=sc*xysy[:,2])
        cb=plt.colorbar()
        cb.set_label('Bias in Y [nm]')
        plt.clim([-20,20])
        plt.xlabel(u'X position in FOV [um]')
        plt.ylabel(u'Y position in FOV [um]')
        plt.title(f'{metric} in Y over FOV')
        plt.savefig(self.outdir + tag+"y-bias-fov.png" )
        plt.savefig(self.outdir + tag+"y-bias-fov.svg" )
        

    
    def render_sr_image_per_pattern(self, ep, frames, sr_factor, smlm:SMLM):
        indices = np.concatenate([self.result.frames[i] for i in frames])
        xyI = self.get_xyI(ep, normalize=True)[indices]
        w,h=self.result.get_image_size()
        img = np.zeros((h*sr_factor,w*sr_factor))
        img = loc.draw_spots(img,xyI[:,0],xyI[:,1],xyI[:,2],1/sr_factor,1/sr_factor,sr_factor,smlm)
        return img
            
    def draw_spots_in_pattern(self, filename, allmod, ep, tiffname, spotfilter,numpts=2000):
        sel = self.get_filtered_spots(spotfilter,allmod)
        
        np.random.seed(0)
        indices = np.arange(len(sel))# np.nonzero(np.min( self.IBg[:,:,0],1) > min_intensity)[0]
        np.random.shuffle(indices)
        
        sel = sel[:np.minimum(numpts,len(indices))]
        
        mod = allmod[ep]
        xyI = self.get_xyI(ep)[sel]
        
        moderr = self.calc_moderr(allmod,sel)[0]
        spot_jitter = np.std(moderr,1)
        std_jitter = np.std(moderr)

        sumI = np.sum(self.IBg[sel, :, 0],1)
#        pattern_frames = np.arange(0,len(allmod),2) + (ep % 2)
#        sumI = np.sum(self.IBg[sel][:, pattern_frames, 0],1)
        
        # normalize
        normI = xyI[:,2] / sumI
        
        k = np.sqrt(mod[0]**2+mod[1]**2)
        proj = xyI[:,0]*mod[0]+xyI[:,1]*mod[1]
        x = proj % (np.pi*2) 

        fig=plt.figure(figsize=figsize)
#        center = self.result.get_image_size()/2
#        r = ( np.abs(proj - (center[0]*mod[0]+center[1]*mod[1])) ) / np.linalg.norm(mod[0:2])
#        plt.scatter(x, normI,c=r,cmap='viridis', marker='.')
#        plt.colorbar()
        plt.scatter(x, normI, c=spot_jitter, marker='.')

        sigx = np.linspace(0,2*np.pi,400)
        exc = mod[4]*(1+mod[2]*np.sin(sigx-mod[3]))
        plt.plot(sigx, exc, 'r', linewidth=4, label='Estimated P')

        plt.plot(sigx, exc - std_jitter, ':k',label='P - std(moderr)')
        plt.plot(sigx, exc + std_jitter, ':k', label='P + std(moderr)')

        mean_crlb_I_f = np.mean(self.IBg_crlb[sel,:,0])

        for filtername, filterval in [spotfilter]:
            if filtername == "moderror":
                plt.plot(sigx, exc - filterval, label='Threshold min')
                plt.plot(sigx, exc + filterval, label='Threshold max')
                
            if filtername == "moderror2":
                plt.plot(sigx, exc * (1 + np.sqrt(filterval)), label='Minimum moderror')
                plt.plot(sigx, exc * (1 - np.sqrt(filterval)), label='Minimum moderror')
        
       # plt.plot(sigx, np.ones(len(sigx))*min_intensity_line_y, label='Min intensity filter cutoff')

        plt.ylim([-0.01,0.51])        
        plt.xlabel('Phase [radians]')
        plt.ylabel(r'Normalized intensity ($I_k$)')
        plt.title(f': Pattern {ep}. K={k:.4f} ' + f" Phase={allmod[ep,3]*180/np.pi:.3f})")
        plt.colorbar()
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def draw_spots_in_patterns(self, title, allmod, pattern_indices, numpts=2000):
        np.random.seed(0)
        indices = np.arange(len(self.IBg))# np.nonzero(np.min( self.IBg[:,:,0],1) > min_intensity)[0]
        np.random.shuffle(indices)   # this gets rid of blobs
                
        fig=plt.figure(figsize=(5,4))
        
        for i,ep in enumerate(pattern_indices):
            mod = allmod[ep]
            xyI = self.get_xyI(ep)[indices]
            sumI = np.sum(self.IBg[indices, :, 0],1)
            
            # normalize
            normI = xyI[:,2] / sumI
            proj = xyI[:,0]*mod[0]+xyI[:,1]*mod[1]
            x = proj % (np.pi*2)
            sigx = np.linspace(0,2*np.pi,400)

            q_expected=mod[4]*(1+mod[2]*np.sin(proj - mod[3]))
            err = ( q_expected-normI ) ** 2
            sel = np.arange(len(err))[:numpts]#np.nonzero(err<np.percentile(err, 80))[0]#[:numpts]
            
            s=1/(2*np.pi)
            plt.scatter(x[sel]*s, normI[sel], marker='.', s=10)#, c=['b','r','c','y','g'][i])
            plt.plot(sigx*s, mod[4]*(1+mod[2]*np.sin(sigx-mod[3])), '-', linewidth=6, label=f'Pattern {ep}')
        
        plt.ylim([0,0.4])

        plt.xlabel('Position in pattern [0-1]')
        plt.ylabel(f'Excitation power')
        plt.title(title)
#        plt.legend()
#        plt.tight_layout()
        fn=f"combined-phases-{pattern_indices}"
        plt.savefig(self.outdir + fn+".svg")
        plt.savefig(self.outdir + fn+".png")

        plt.close(fig)

    def draw_axis_intensity_spread(self,pattern_indices,mod,spotfilter):
        sel = self.get_filtered_spots(spotfilter,mod)
        
        totals =np.sum(self.IBg[sel, :,0],1)
        v = []
        for k in range(len(pattern_indices)):
            axis_sums = np.sum(self.IBg[sel][:,pattern_indices[k],0],1)/totals
            v.append(axis_sums)

        if len(pattern_indices)==2:
            axname = ['Y', 'X']
        else:
            axname = [f"{i}" for i in range(len(pattern_indices))]
            
        print(f"Plotting relative axis intensity for {len(sel)} spots")
        plt.figure()
        plt.hist(v,label=[f'Axis {axname[k]}' for k in range(len(pattern_indices))],bins=50)
        plt.legend()
        plt.title('Relative power of each axis')
        
    def get_filtered_spots(self, spotfilters, mod, frames=None, reportfn=None):
        if frames is None:
            frames = np.arange(len(self.result.frames))

        indices = np.concatenate([self.result.frames[i] for i in frames])
        startlen = len(indices)

        if type(spotfilters) != list:
            spotfilters = [spotfilters]

        for i,f in enumerate(spotfilters):
            min_intensity = f[1]
            if f[0] == 'minfilter-all':
                indices = indices [np.nonzero(np.min(self.IBg[indices,:,0],axis=1) > min_intensity)[0]]
            elif f[0] == 'minfilter-firstlast':
                indices = indices [np.nonzero(
                     (self.IBg[indices, 0,0] > min_intensity) & 
                     (self.IBg[indices,-1,0] > min_intensity))[0]]
            elif f[0] == 'moderror':
                exc = self.excitation_matrix(mod, indices)
                sumI = np.sum(self.IBg[indices, :, 0],1)
                normI = self.IBg[indices, :, 0] / sumI[:,None]    
                err = np.max( (normI-exc)**2, 1 )
                indices = indices[np.nonzero(err<f[1])[0]]
            elif f[0] == 'moderror2':
                mean_crlb_I_f = np.mean(self.IBg_crlb[indices,:,0])

                exc = self.excitation_matrix(mod, indices)
                sumI = np.sum(self.IBg[indices, :, 0],1)
                normI = self.IBg[indices, :, 0] / sumI[:,None]    
                err = np.max( (normI/exc-1)**2, 1 )
                indices = indices[np.nonzero(err < f[1])[0]]
            elif f[0] == 'intensitypercentile':
                intensities = self.result.get_xyI()[indices,2]
                maxi = np.percentile(intensities, f[1])
                indices = indices[np.nonzero(intensities<maxi)[0]]
            else:
                raise ValueError('Invalid spot filter')
               
            if reportfn is not None:
                reportfn(f"Filter {f} removed {startlen-len(indices)}/{startlen} spots")
            startlen = len(indices)

        return indices

    
    def excitation_matrix(self, mod, sel):
        xyI = self.result.get_xyI()[sel,np.newaxis]
        mod_ = mod[np.newaxis]
        exc = mod_[:,:,4]*(1+mod_[:,:,2]*np.sin(xyI[:,:,0]*mod_[:,:,0]+xyI[:,:,1]*mod_[:,:,1]-mod_[:,:,3]))
        return exc

    def calc_moderr(self,mod,sel):
        exc = self.excitation_matrix(mod,sel)
        sumI = np.sum(self.IBg[sel, :, 0],1)

        normI = self.IBg[sel, :, 0] / sumI[:,None]
        moderr = normI-exc
        moderr_crlb = self.IBg_crlb[sel,:,0] / sumI[:,None]
        return moderr, moderr_crlb
    

    def plot_moderr_vs_intensity(self,mod, spotfilter):
        sel = np.arange(len(self.IBg))# self.get_filtered_spots(spotfilter, mod, frames)
        intensities = self.result.get_xyI()[:,2]
        I_bins = np.linspace(0,3000, 50)
        indices = np.digitize(intensities, I_bins)
        moderr, moderr_crlb=self.calc_moderr(mod, sel)
        moderr_crlb = np.mean(moderr_crlb,1)
        spot_jitter = np.std(moderr,1)
        
        y = [ np.mean(spot_jitter[indices==k]) for k in range(len(I_bins)) ]
        crlb = [ np.mean(moderr_crlb[indices==k]) for k in range(len(I_bins)) ]
        plt.figure()
        plt.plot(I_bins, y, label='Spot jitter')
        plt.plot(I_bins, crlb, label='CRLB')
        plt.legend()

        if False:  # Too many single spots
            plt.scatter(intensities, spot_jitter)
            plt.xlim([0, np.mean(intensities)*4])
            plt.ylim([0, np.mean(spot_jitter)*4])

        plt.title('Modulation errors vs spot intensities')
        plt.xlabel('Spot Intensity [ph]')
        plt.ylabel('Spot Jitter [normalized excitation]')
        plt.savefig(self.outdir+"moderr_vs_intensity.png")
        plt.savefig(self.outdir+"moderr_vs_intensity.svg")
            
        return moderr

    def compute_modulation_error(self, mod, spotfilter, frames=None):
        sel = self.get_filtered_spots(spotfilter, mod, frames)

        moderr = self.calc_moderr(mod,sel)[0]
        mse = moderr**2

        plt.figure()
        std = np.std(moderr)
        v,bins,_= plt.hist(moderr.flatten(), range=[-std*3,std*3],bins=50)
        fit = scipy.stats.norm.pdf(bins, np.mean(moderr), std)
        plt.plot(bins, fit * np.mean(v) / np.mean(fit))
        plt.title("Modulation error distribution")
        plt.xlabel('Modulation error ( P - I_f / I )')
        plt.savefig(self.outdir + 'mod-err.png')
        
        # compute std(N')
        """
        alpha = np.std(self.IBg[sel, :, 0], 1) / np.std(exc * sumI[:,None], 1)
        fig=plt.figure()
        m_alpha = np.median(alpha)
        plt.hist(alpha, bins=50, range=[0,m_alpha*5])
        plt.title(f'Molecule brightness variation (median(alpha)={m_alpha:.1f}).')
        plt.xlabel('Alpha')
        plt.savefig(self.outdir + "alpha-hist.png")
#        plt.close(fig)
        """
        
        fig,ax=plt.subplots(2,1)
        
        maxmse = np.max(mse,1)
        ax[0].hist(mse.flatten(), bins=50, range=[0,np.median(mse)*5])
        ax[1].hist(maxmse, bins=50, range=[0,np.median(maxmse)*5])
        ax[0].set_title(f'Modulation squared errors (median={np.median(mse):.3f}')
        ax[1].set_title(f'Per spot max sq. error (median={np.median(maxmse):.3f}')
        plt.savefig(self.outdir + "modulation-squared-error-hist.png")
        plt.close(fig)

        return moderr
    
    def compute_modulation_chisq(self, mod, pattern_indices, spotfilter, plot=False, frames=None):
        sel = self.get_filtered_spots(spotfilter, mod, frames)
        
        exc = self.excitation_matrix(mod, sel)
        intensities = self.IBg[sel, :, 0]
        var_I = self.IBg_crlb[sel, :, 0]**2

        chisq = np.zeros(len(mod))
        for axis in range(len(pattern_indices)):
            pf = pattern_indices[axis]
            # excitation matrix normalized for axis
            exc_axis = exc[:,pf] / np.sum(exc[:,pf],1)[:,None]

            Nj = np.sum(intensities[:, pf], 1)
            x = intensities[:, pf] - exc_axis * Nj[:, None]
            var_sum = np.sum(var_I[:,pf],1)
            var_x = (1-exc_axis)**2 * var_I[:,pf] + exc_axis**2 * (
                    var_sum[:,None]-var_I[:,pf])
    
            if plot:
                fig,ax=plt.subplots(3,2,squeeze=False)
                for k in range(len(pf)):
                    ax[k,0].hist(x[:,k],bins=50,label=f'x - phase {k}')
    #                ,var_x[:,k]],bins=50,range=(-200,500),label=[f'x - phase {k}', f'var x - phase {k}'])
                    ax[k,0].legend()
                    ax[k,0].set_xlabel('photons')
                    ax[k,1].hist(var_x[:,k],bins=50,label=f'var(x) - phase {k}')
                    ax[k,1].legend()
                    ax[k,1].set_xlabel('photons^2')
#                ax[0].set_title(f"Axis {axis} modulation quality (Chi-Squared)")
                fig.tight_layout()
            chisq[pf] = np.sum(x**2 / var_x,0)
        goodness_of_fit = chisq / len(sel)

        return goodness_of_fit                   
        
        
    def plot_modulation_chisq_timebins(self, mod, pattern_indices, spotfilter, numbins, chisq=True):
        fr = np.arange(len(self.result.frames))
        errs = np.zeros((numbins, len(mod)))
        for k, frbin in enumerate( np.array_split(fr, len(errs)) ):
            errs[k] = self.compute_modulation_chisq(mod, pattern_indices, spotfilter, frames=frbin)
    
        fig=plt.figure()        
        for ax,axi in enumerate(pattern_indices):
            for st in range(len(axi)):
                plt.plot(errs[:,axi[st]], label=f'Axis {ax}, step {st}')
        plt.title('Modulation fit quality over time')
        plt.xlabel('Frame bin')
        plt.legend()
        plt.savefig(self.outdir+'modulation-chisq-timebins.png')
        plt.close(fig)

    def save_IBg(self, fn, mod):
        import h5py 
        with h5py.File(fn, 'w') as f:
            dtype = [('frame', '<u4'),
                 ('smlm_x', '<f4'), 
                 ('smlm_y', '<f4'), 
                 ('photons', '<f4'), 
                 ('bg', '<f4'),
                 ('roix', '<u4'),
                 ('roiy', '<u4')]

            for k in range(len(mod)):
                dtype.append((f'I{k}', '<f4'))
                dtype.append((f"bg{k}", '<f4'))
                dtype.append((f'I_crlb{k}', '<f4'))
                dtype.append((f"bg_crlb{k}", '<f4'))

            locs = f.create_dataset('locs', shape=(len(self.IBg),), dtype=dtype)
            locs['frame'] = self.result.get_frame_numbers()
            xyIBg = self.result.get_xyIBg()
            locs['smlm_x'] = xyIBg[:,0]
            locs['smlm_y'] = xyIBg[:,1]
            locs['photons'] = xyIBg[:,2]
            locs['bg'] = xyIBg[:,3]
            roipos = self.result.get_roipos()
            locs['roix'] = roipos[:,0]
            locs['roiy'] = roipos[:,1]
            for k in range(len(mod)):
                locs[f'I{k}'] = self.IBg[:, k, 0]
                locs[f'bg{k}'] = self.IBg[:, k, 1]
                locs[f'I_crlb{k}'] = self.IBg_crlb[:, k, 0]
                locs[f'bg_crlb{k}'] = self.IBg_crlb[:, k, 1]

    
