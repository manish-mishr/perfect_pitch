import os.path
from numpy import array, ma
import numpy as np
import numpy,scipy, sklearn
import librosa
from aubio import source,pitch
import matplotlib.pyplot as plt

def array_from_text_file(filename, dtype = 'float'):
    return array([line.split() for line in open(filename).readlines()],
        dtype = dtype)


def pitch_detection(filename,sr,ground_truth,win_s,hop_s,downsample,tolerance,pitch_method,draw=True):
    # global pitch,source
    samplerate = sr
    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    pitch_o = pitch(pitch_method, win_s, hop_s, samplerate)
    # pitch_o.set_unit("midi") # default is HZ
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        pitch_sample = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()

        pitches += [pitch_sample]
        confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    # detect pitches
    skip = 1

    pitches = array(pitches[skip:])
    confidences = array(confidences[skip:])
    times = [t * hop_s for t in range(len(pitches))]

    if os.path.isfile(ground_truth):
        ground_truth = array_from_text_file(ground_truth)
        true_freqs = ground_truth[:,1]
        true_freqs = ma.masked_where(true_freqs < 2, true_freqs)
        true_times = ground_truth[:,0]

    cleaned_pitches = pitches
    cleaned_pitches = ma.masked_where(confidences < tolerance, cleaned_pitches)
    actual_times = [t/float(samplerate) for t in times]
    cleaned_pitches = ma.masked_where(cleaned_pitches > 800, cleaned_pitches)
    cleaned_pitches = ma.masked_where(cleaned_pitches < 50, cleaned_pitches)

    if draw:
        plt.plot(actual_times, cleaned_pitches, 'g')
        plt.plot(true_times, true_freqs, 'r')
        plt.show()
    return (actual_times,cleaned_pitches),(true_times,true_freqs)

def filter_freq(predict_time,true_time_freq):
    result = [9999] if predict_time[0] == 0 else []
    filtered = filter(lambda x:x[0] in predict_time,true_time_freq)
    result.extend(map(lambda x:x[1],filtered))
    return np.array(result)

def gross_error(pitch_detection_result):
    predict_time,predict_freq = pitch_detection_result[0]
    predict_time = np.round(np.array(predict_time),2)
    predict_freq = np.array(predict_freq)
    true_time,true_freq = pitch_detection_result[1]
    true_freq.mask = ma.nomask
    true_time_freq = zip(true_time,true_freq)
    true_freq_filtered = filter_freq(predict_time,true_time_freq)

    return sum((abs(predict_freq-true_freq_filtered)>true_freq_filtered*0.2))/(len(predict_freq)+0.0)

def write_output(output_path,filenames,gross_errors,types):
    output_file = os.path.join(output_path,'output.txt')
    if not os.path.isfile(output_file):
        print 'writing output file',output_file
        with open(output_file, 'a') as f:
            for i in range(len(filenames)):
                name,error,mask_type = filenames[i],gross_errors[i],types[i]
                f.write(str(name)+'\t'+str(error)+'\t'+mask_type+'\n')
        f.close()
    else:
        print 'output already exist. no need to write'

wav_path = "MIR-1K_for_MIREX/results/"
truth_path = 'MIR-1K_for_MIREX/PitchLabel/'

sr = 22050
downsample = 1
win_s = 2048 // downsample # fft size
hop_s = 1024  // downsample # hop size
tolerance = 0.85
pitch_method='yin'

filenames,gross_errors = [],[]
count = 0
for file in os.listdir(wav_path):
    if file.endswith(".wav"): # find all files ending in wav
        count += 1
        if count % 500 == 0:
            print count
        file_path = os.path.join(wav_path,file)
        file_head = '_'.join(file.split('_')[:3])

        truth_file = os.path.join(truth_path,file_head+'.pv') # organize ground truth file names
        print file
        result = pitch_detection(file_path,sr,truth_file,win_s,hop_s,downsample,tolerance,pitch_method,draw=False)
        error = gross_error(result)
        filenames.append(file)
        gross_errors.append(error)

types = map(lambda x:x.split('_')[-2],filenames)
write_output("MIR-1K_for_MIREX",filenames,gross_errors,types)
