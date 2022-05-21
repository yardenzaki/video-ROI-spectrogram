import cv2

# determine what OpenCV version we are using
try:
    import cv2.cv as cv

    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False

import sys
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from matplotlib import mlab
import random
import scipy.fftpack
import math
from scipy import signal
from scipy.signal import find_peaks ,butter, lfilter
import operator
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter
from sklearn import preprocessing

def get_fft(y, frequency):
    y_fft = np.fft.fft(y)
    y_fft = np.abs(y_fft)
    x_fft = np.fft.fftfreq(len(y_fft), frequency)
    x_fft = x_fft[:len(x_fft) // 2]
    y_fft = y_fft[:len(y_fft) // 2]

    return x_fft, y_fft

def create_oscilating():
    # video Writer
    # func_fourcc = cv2.VideoWriter_fourcc
    # fourcc = func_fourcc('M', 'J', 'P', 'G')
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    fps_vid_writer = 10 #240
    # Total time:
    duration=5 #sec
    # Total N
    n=fps_vid_writer*duration
    ts = np.linspace(0, duration, int(n))

    num_of_sines = 5  # how many different frequencies data the circle has
    gains = [round(random.uniform(3, 8), 0) for i in range(num_of_sines)]  # assuming that every avg of window of pixels can vary between 10 to 40 gray levels (I)
    gains = sorted(gains,reverse=True)  # Sorting so that low freq fluctations have higher gain (amplitude) than high freq flucs.
    #freq_bands = random.sample(range(1, int(fps_vid_writer / 2)), num_of_sines)
    freq_bands = random.sample(range(1, 10), num_of_sines)
    freq_bands = sorted(freq_bands)
    print("gains",gains)
    print("freq_bands", freq_bands)

    func_list = [gains[i] * sin(ts * 2 * pi * freq_bands[i]) for i in range(num_of_sines)]

    # defines and creates WHITE NOISE signal
    noise_amp = 2  # The amplitude of signal noise
    mean = 0
    std = 1
    num_samples = int(n)
    noise = np.random.normal(mean, std, size=num_samples)
    noise = [round(noise_amp * noise[i], 0) for i in range(len(noise))]  # amplifying noise amps

    # print("noise",noise)
    data = noise
    for fun in func_list:
        data = data + fun

    vidFname ="osc_circle_"
    vidFname = vidFname+'%sHZ_%sAmpl_%dsec_%dFPS.avi' % (str(freq_bands),str(gains),duration,fps_vid_writer)
    vidWriter = cv2.VideoWriter(vidFname, fourcc, fps_vid_writer, (200, 200), True)

    for fr in range(1,n):
        new_frame = np.zeros((200, 200, 3), np.uint8)
        circle_center_x=100
        circle_center_y=100
        circle_center_y=int(circle_center_y+data[fr])
        cv2.circle(new_frame, (circle_center_x, circle_center_y), 10, (0, 0, 255), -1)
        cv2.imshow("new_frame", new_frame)
        cv2.waitKey(20) & 0xFF
        vidWriter.write(new_frame)
        #print("fr",fr)

    cv2.waitKey(200) & 0xFF
    cv2.destroyAllWindows()
    vidWriter.release()
    return

def create_flashing():
    # video Writer
    # func_fourcc = cv2.VideoWriter_fourcc
    # fourcc = func_fourcc('M', 'J', 'P', 'G')
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    fps_vid_writer = 240
    # Total time:
    duration=2 #sec
    #Blink freq:
    BlnkFrq=90 #Hz or FPS
    duty = 50

    n=fps_vid_writer*duration
    xarr=np.ones(n)
    oc_samp=fps_vid_writer/BlnkFrq
    on_samp=oc_samp*duty/100
    vidFname ="bd_"
    vidFname = vidFname+'%dHZ_%dsec_%dFPS.avi' % (BlnkFrq,duration,fps_vid_writer)
    vidWriter = cv2.VideoWriter(vidFname, fourcc, fps_vid_writer, (200, 200), True)

    temp=0
    for fr in range(1,n+1):
        fro=int(temp+on_samp)
        to=int(fr*oc_samp)
        xarr[fro:to]=0
        #print(xarr, fro, to)
        temp=temp+oc_samp
    print(xarr,len(xarr))

    fr = np.zeros((200, 200, 3), np.uint8)
    for i in xarr:
        if i==1:
            cv2.circle(fr,(100,100),25,(0,0,255),-1)
            print("DRAW")
        else:
            fr=np.zeros((200, 200, 3), dtype="uint8")
            print("BLANK")

        cv2.imshow("fr", fr)
        cv2.waitKey(20) & 0xFF
        vidWriter.write(fr)

    cv2.waitKey(200) & 0xFF
    cv2.destroyAllWindows()
    vidWriter.release()
    return

def normalize(X):
    data = np.array(X)
    dmean = data.mean()
    dmax=data.max()
    dmin=data.min()
    print("original data",dmin, dmax, dmean)
    data=data-dmean


    dmean = data.mean()
    dmax=data.max()
    dmin=data.min()

    print("substracting mean data",dmin, dmax, dmean)

    X_scaled = preprocessing.minmax_scale(data, feature_range=(-1, 1))


    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    return X_scaled

def num_to_bgr(val, max_val):
    minimum=0
    maximum=max_val
    ratio = 2 * (val-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return (b,g,r)

def sliding_window(image, stepSize, windowSize):
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def GetWindowedAvgI(AllFrames,KernelSize,FrameShape):
    # pre-process to sharpen image
    outline = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    framelst = []
    MeanLst=[] #[[frameCounter,windowCounter,(xc,yc),mean_I]]
    y_len, x_len = FrameShape
    frameCounter=0
    for frame in AllFrames:
        if len(framelst)>0:
            MeanLst.append([framelst])
        framelst=[]
        print("FRAME")
        # apply convolution of kernel with frame
        FrameReshaped=np.reshape(frame,(x_len,y_len))
        FrameReshaped = cv2.filter2D(FrameReshaped, -1, outline)
        cv2.imshow("After Processing",FrameReshaped)

        windows = sliding_window(FrameReshaped, KernelSize, (KernelSize, KernelSize))
        windowCounter =0
        frameCounter = frameCounter + 1
        for window in windows:
            xc=window[0]
            yc=window[1]
            win=window[2]
            #print("WINDOW",xc,yc,win.shape)
            # mean and median after image processing
            #cv2.imshow("window", window[2])
            #cv2.waitKey(1000) & 0xFF
            #mean_I = round(np.median(win), 2)
            mean_I= round(np.mean(win),2)

            MeanLst.append(mean_I)
            windowCounter=windowCounter+1
            #print("mean_I",mean_I)
    MeanArr = MeanLst
    print("MeanArr_shape",len(MeanArr))
    return MeanArr

def MakePallete(low_f,high_f):
    incs=20
    incs_set=np.linspace(low_f, high_f,incs)
    pallete=np.full((len(incs_set)*30,100,3), 255, dtype=np.uint8)

    count=0
    for i in incs_set:
        text = str(int(i))
        f_to_plot = num_to_bgr(i, high_f)
        x=30*count
        y=0
        pallete[x:x+100, y:y+30] = f_to_plot
        cv2.putText(pallete, text, (y + 10, x + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255, 255, 255), 1)
        count=count+1
    return pallete

def ROI_Spectrogram(vidFname):
    #Initialization params:
    FirstFrame = True
    ROI = None
    all_frames=[]


    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)
    vidReader.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)


    if USE_CV2:
        # OpenCV 2.x interface
        vidFrames = int(vidReader.get(cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(vidReader.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv.CV_CAP_PROP_FPS))
        frame_count = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))
        func_fourcc = cv.CV_FOURCC
        duration = frame_count / fps
    else:
        # OpenCV 3.x interface

        vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv2.CAP_PROP_FPS))
        frame_count = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))
        func_fourcc = cv2.VideoWriter_fourcc
        duration = frame_count / fps

    if np.isnan(fps):
        fps = 29

    print(' %d frames' % vidFrames)
    print (' (%d x %d)' % (width, height))
    print(' FPS:%d' % fps)


    # how many frames
    nrFrames = vidFrames
    print("nrFrames",nrFrames)

    # read video
    for frameNr in range(nrFrames):
        # print(frameNr)
        sys.stdout.flush()
        #print(frameNr,nrFrames)
        if frameNr <= nrFrames:
            # read frame
            _, im = vidReader.read()



            #Check if its the FirstFrame:
            if FirstFrame:
                FF = im
                ROI=cv2.selectROI("PLease Select region of interest",im,showCrosshair=True,fromCenter=True)
                FirstFrame=False

            if im is None:
                # if unexpected, quit
                break

            AreaToSpectogram = im[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]):int(ROI[0] + ROI[2])]
            rect_size=AreaToSpectogram.shape
            rect_size=rect_size[0]*rect_size[1]

            #AreaToSpectogram = im[ROI[1] - rect_size:ROI[1] + rect_size, ROI[0] - rect_size:ROI[0] + rect_size]
            # FF[ROI[1]-rect_size:ROI[1]+rect_size,ROI[0]-rect_size:ROI[0]+rect_size]=(255,0,0)
            cv2.imshow("selected ROI", FF)
            cv2.imshow("AreaToSpectogram", AreaToSpectogram)
            outline = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])
            filtered_im = cv2.filter2D(AreaToSpectogram, -1, outline)
            cv2.imshow("Outlined", filtered_im)

            # convert to gray image
            if len(im.shape) > 2:
                grayIm = cv2.cvtColor(AreaToSpectogram, cv2.COLOR_RGB2GRAY)
            else:
                # already a grayscale image?
                grayIm = AreaToSpectogram


            ROIpixels=np.ravel(grayIm)

            all_frames.append(ROIpixels)
            print("len_all_frames", len(all_frames))

            print("frameNr", frameNr)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                vidReader.release()
                cv2.destroyAllWindows()
                break

    # free the video reader/writer
    vidReader.release()


    #vertical stacking of all frames:
    result_arr = np.vstack(all_frames)
    print("result_arr",result_arr.shape)
    #Getting Array of mean intensity windows for each frame:
    MeanArr=GetWindowedAvgI(result_arr, int(2*rect_size), grayIm.shape) #Returns: Array - with shape: (no_of_frames,no_of_total_windows,4) the last item is: [[frameCounter,windowCounter,(xc,yc),mean_I]]


    TrueFPS=fpsForBandPass
    TrueDuration = frame_count/TrueFPS

    #### NORMALIZING I VALUES FIRST:
    MeanArr = normalize(MeanArr)

    print("TrueFPS",TrueFPS)
    print("TrueDuration", TrueDuration)
    print("True_n", frame_count)

    # Plot setup
    ts = np.linspace(0, TrueDuration, frame_count)

    xfs,yfs=get_fft(MeanArr,1/TrueFPS)
    #print("xfs", xfs)
    #print("yfs", yfs)

    win = ButterBandpassFilter(9, lowFreq, highFreq, TrueFPS)
    win.update(MeanArr)
    filtered = win.collect()
    filtered = butter_bandpass_filter(MeanArr, lowFreq, highFreq, TrueFPS, order=6)
    amplified_filtered = Factor * filtered
    xfs_data, yfs_data = get_fft(MeanArr, 1 / TrueFPS)
    xf_filter, yf_filter = get_fft(filtered, 1 / TrueFPS)
    xf_amp, yf_amp = get_fft(amplified_filtered, 1 / TrueFPS)
    nperseg = len(MeanArr)-round(0.1*len(MeanArr))  # even
    #nperseg = nperseg-1  # odd
    psd_mlab, f_mlab=matplotlib.mlab.psd(MeanArr, NFFT=nperseg, Fs=TrueFPS, noverlap=(nperseg // 2),
    detrend=None, scale_by_freq=True, window=mlab.window_hanning)


    #yf_filtered = scipy.fftpack.fft(filtered)
    #yf_amp_filtered = scipy.fftpack.fft(amplified_filtered)

    peaks_data, properties_data = find_peaks(yfs_data, height=60, threshold=5)
    peaks_amp, properties_amp = find_peaks(yf_amp, height=60, threshold=5)
    peaks_psd, properties_psd = find_peaks(psd_mlab, height=0.2, threshold=0.02)

    print("DATA peaks info:")
    for p in peaks_data:
        print(xfs_data[p], yfs_data[p], properties_data)
    print("AMPLIFIED peaks info:")
    for p in peaks_amp:
        print(xf_amp[p], yf_amp[p], properties_amp)

    if debug:
        # create plot:
        fig, axs = plt.subplots(5)
        fig.suptitle('Window intensity')

        axs[0].plot(ts, MeanArr)
        axs[0].minorticks_on()
        axs[0].set_xlabel("[s]")
        axs[0].set_ylabel("Intensity")
        axs[0].grid(False, 'both', 'both')

        axs[1].plot(xfs, np.abs(yfs),'k') #use semilogy instead of plot
        axs[1].plot(xfs, yfs, 'k', label ="Original Data")  # use semilogy instead of plot
        axs[1].plot(xf_filter, yf_filter, 'r', label ="Filtered")  # use semilogy instead of plot
        axs[1].plot(xf_amp, yf_amp, 'g', label ="Filtered+Amplified")  # use semilogy instead of plot


        #axs[1].plot(xfs, np.abs(yf_filtered))  # use semilogy instead of plot
        #axs[1].plot(xfs, np.abs(yf_amp_filtered))  # use semilogy instead of plot
        if len(peaks_amp) > 0:
            axs[1].plot(xf_amp[peaks_amp], yf_amp[peaks_amp], "x")
        axs[1].set(xlim=(0))
        axs[1].minorticks_on()
        axs[1].set_frame_on(1)
        axs[1].legend()
        axs[1].grid(True, 'both', 'both')
        axs[1].set_xlabel("[Hz]")
        axs[1].set_ylabel("Power")


        ##PLOT PSD INSTEAD OF FFT
        axs[2].plot(f_mlab, psd_mlab, '-^',label ="PSD")
        if len(peaks_psd) > 0:
            axs[2].plot(f_mlab[peaks_psd], psd_mlab[peaks_psd], "x")
        axs[2].set(xlim=(0))
        axs[2].minorticks_on()
        axs[2].set_frame_on(1)
        axs[2].legend()
        axs[2].grid(True, 'both', 'both')
        axs[2].set_xlabel("[Hz]")
        axs[2].set_ylabel("PSD")

        #PLOT SPECTROGRAMS
        fsg, tsg, Sxxsg = signal.spectrogram(MeanArr, TrueFPS,window=('tukey', 0.25),nperseg=30, nfft=None)
        fsg_filtered, tsg_filtered, Sxxsg_filtered = signal.spectrogram(amplified_filtered, TrueFPS, window=('tukey', 0.25), nperseg=30, nfft=None)
        print(len(fsg),len(tsg),len(Sxxsg))

        fig.suptitle('spectrogram')

        axs[3].pcolormesh(tsg, fsg, Sxxsg, shading='gouraud')
        axs[3].set_xlabel("Time [sec]")
        axs[3].set_ylabel("Frequency [Hz]")

        fig.suptitle('Spectrogram Filtered Data')

        axs[4].pcolormesh(tsg_filtered, fsg_filtered, Sxxsg_filtered, shading='gouraud')
        axs[4].set_xlabel("Time [sec]")
        axs[4].set_ylabel("Frequency [Hz]")

        show()

        print("----------------------------------------------")
        cv2.waitKey(2000)

################# main script

#vidFname = 'media/osc_circle_[5, 59, 117, 160, 208]HZ_[8.0, 7.0, 6.0, 6.0, 5.0]Ampl_20sec_30FPS.avi'
vidFname = 'media/crane1.mp4'
#vidFname = 'media/vent.MOV'
debug=True
Factor=5

# the fps used for the bandpass
fpsForBandPass =30 #-1 #600  # use -1 for input video fps
#תחום התדרים בהם נרצה להגביר אמפליטודות תנודה של כל פיקסל שלו תנועה בתחום תדרים אלו
# low ideal filter
lowFreq =0.1#60 #72
# high ideal filter
highFreq =10#100 #92
# output video filename

starting_frame=0 # DO NOT TOUCH

#APPLY IF WANT TO CREATE FLASHING DOT
#create_flashing()
#APPLY IF WANT TO CREATE Oscilating DOT
#create_oscilating()
ROI_Spectrogram(vidFname)

print("Done!!!")
