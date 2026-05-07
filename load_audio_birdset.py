import soundfile as sf
import librosa

def load_audio(sample, min_len, max_len, sampling_rate):
    path = str(sample["filepath"])

    file_info = sf.info(path)
    sr = file_info.samplerate
    total_duration = file_info.duration
    
    if sample["detected_events"] is not None:
        start = sample["detected_events"][0]
        end = sample["detected_events"][1]
        event_duration = end - start
        
        if event_duration < min_len:
            extension = (min_len - event_duration) / 2
            
            # try to extend equally 
            new_start = max(0, start - extension)
            new_end = min(total_duration, end + extension)
            
            if new_start == 0:
                new_end = min(total_duration, new_end + (start - new_start))
            elif new_end == total_duration:
                new_start = max(0, new_start - (new_end - end))
            
            start, end = new_start, new_end

        if end - start > max_len:
            # if longer than max_len
            end = min(start + max_len, total_duration)
            if end - start > max_len:
                end = start + max_len
    else:
        start = sample["start_time"]
        end = sample["end_time"]

    start, end = int(start * sr), int(end * sr)
    audio, sr = sf.read(path, start=start, stop=end)

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return audio, sr
