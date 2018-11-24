import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from scipy.io import wavfile
from scipy.fftpack import dct
from python_speech_features import mfcc, logfbank

'''
@brief class to extract the characteristics of the audio file to analize
'''
class AudioSignal:
    def __init__(self,pFileName):
        self.fileName = pFileName
        self.frequencySampling = None 
        self.audioSignal = None 
        self.featuresMFCC = None
        self.filterbankFeatures = None        
        self.plotFlag = False
    
    '''
    @brief set the audio file name and directory (must be a .wav format)
    @param pFileName the directory of the file
    '''
    def setFileName(self, pFileName):
        self.fileName = pFileName

    '''
    @brief set the audio frequency sampling
    @param pFrequencySampling the value
    '''
    def setFrequencySampling(self, pFrequencySampling):
        self.frequencySampling = pFrequencySampling

    '''
    @brief set the audio audio signal characterization
    @param self the directory of the file
    '''
    def setAudioSignal(self, pAudioSignal):
        self.audioSignal = pAudioSignal

    '''
    @brief set the audio mfcc features
    @param pMFCC
    '''
    def setMFCCFeature(self, pMFCC):
        self.featuresMFCC = pMFCC

    '''
    @brief set the audio filter bank Features
    @param pFilterbankFeatures
    '''
    def setFilterBankFeature(self, pFilterbankFeatures):
        self.filterbankFeatures = pFilterbankFeatures
    
    '''
    @brief methods that get the frequency and the audio signal of an audio file
    @returns frequency_sampling, audioSignal
    '''
    def getAudioSignalFrequency(self):
        try:
            self.frequencySampling, self.audioSignal = wavfile.read(self.fileName)
            pre_emphasis = 0.97
            self.audioSignal = np.append(self.audioSignal[0], self.audioSignal[1:] - pre_emphasis * self.audioSignal[:-1])
        except:
            print("Unable to get frequency_sampling, audioSignal from " + self.fileName)
            return 

    '''
    @brief normalize the energy
    '''
    def normalizeEnergy(self):
        from pydub import AudioSegment

        def match_target_amplitude(sound, target_dBFS):
            change_in_dBFS = target_dBFS - sound.dBFS
            return sound.apply_gain(change_in_dBFS)

        sound = AudioSegment.from_file(self.fileName, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export("./datos_normalizados/" + self.fileName.split("/")[-1], format="wav")
        self.fileName = "./datos_normalizados/" + self.fileName.split("/")[-1]
        self.getAudioSignalFrequency()
        

        #if(self.plotFlag):
        #   self.plotSignal('Frequency (kHz)', 'Signal power (dB)', 'Freq vs Power', 1000.0, lengthSignal, len_fts, 1, self.signalPower)

    '''
    '''
    def signalFeatures(self, frame_size, frame_stride):        
        # params
        frame_length, frame_step = frame_size * self.frequencySampling, frame_stride * self.frequencySampling  # Convert from seconds to samples
        signal_length = len(self.audioSignal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(self.audioSignal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # hamming window
        frames *= np.hamming(frame_length)

        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.frequencySampling / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / self.frequencySampling)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        self.filterbankFeatures = np.dot(pow_frames, fbank.T)
        self.filterbankFeatures = np.where(self.filterbankFeatures == 0, np.finfo(float).eps, self.filterbankFeatures)  # Numerical Stability
        self.filterbankFeatures = 20 * np.log10(self.filterbankFeatures)  # dB
        
        num_ceps = 20
        low_freq_mel = 0
        self.featuresMFCC = dct(self.filterbankFeatures, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
        
        cep_lifter = 22
        low_freq_mel = 0
        (nframes, ncoeff) = self.featuresMFCC.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        low_freq_mel = 0
        self.featuresMFCC *= lift  #*


    '''
    @brief method to plot the audio signal
    '''
    def plotSignal(self,pXLabel, pYLabel, pTitle, pAmount, pSize, pSize2, pType, pSignal):
        if(pType ==0): 
            time_axis = pAmount * np.arange(0, len(pSignal), 1) / float(self.frequencySampling)
        else: 
            time_axis = np.arange(0, pSize2, 1)*(self.frequencySampling / pSize) / pAmount
        plt.plot(time_axis, pSignal, color='blue')
        plt.xlabel(pXLabel)
        plt.ylabel(pYLabel)
        plt.title(pTitle)

    def normalized_fb(self):
        self.filterbankFeatures -= (np.mean(self.filterbankFeatures, axis=0) + 1e-8)


    def normalized_mfcc(self):
        self.featuresMFCC -= (np.mean(self.featuresMFCC, axis=0) + 1e-8)

    '''
    @brief plot the mfcc features
    @param pFeaturesMFCC the mfcc features
    '''
    def pltoMFCCFeatures(self):
        pFeaturesMFCC = self.featuresMFCC.T
        plt.matshow(pFeaturesMFCC)
        #plt.imshow(np.flipud(pFeaturesMFCC.T), cmap=cm.jet, aspect=0.2, extent=[0,T,0,4])
        plt.title('MFCC Features')
        plt.savefig("./images/MFCC/"+self.fileName.split("/")[-1].split(".")[0].split("_")[0]+"_"+self.fileName.split("/")[-1].split(".")[0])

    '''
    @brief plot the mfcc features
    @param pFeaturesMFCC the mfcc features
    '''
    def pltoFilterBankFeatures(self):
        filterbankFeatures = self.filterbankFeatures.T
        plt.matshow(filterbankFeatures)
        #plt.imshow(np.flipud(filterbankFeatures.T), cmap=cm.jet, aspect=0.2, extent=[0,T,0,4])
        plt.title('Filter bank')
        plt.savefig("./images/FB/"+self.fileName.split("/")[-1].split(".")[0].split("_")[0]+"_"+self.fileName.split("/")[-1].split(".")[0])