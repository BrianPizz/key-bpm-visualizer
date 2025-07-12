from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import mode
from collections import Counter

app = Flask(__name__)
CORS(app)

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Note names
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_key(chroma):
    """
    Detect key using Krumhansl-Schmuckler algorithm
    
    Args:
        chroma: 12-dimensional chroma vector
    
    Returns:
        tuple: (key_name, mode, confidence)
    """
    # Normalize chroma
    chroma = chroma / np.sum(chroma)
    
    # Calculate correlation with all major and minor keys
    major_correlations = []
    minor_correlations = []
    
    for i in range(12):
        # Rotate profiles to match different roots
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)
        
        # Calculate correlation
        major_corr = np.corrcoef(chroma, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma, minor_rotated)[0, 1]
        
        major_correlations.append(major_corr)
        minor_correlations.append(minor_corr)
    
    # Find best matches
    best_major_idx = np.argmax(major_correlations)
    best_minor_idx = np.argmax(minor_correlations)
    
    best_major_corr = major_correlations[best_major_idx]
    best_minor_corr = minor_correlations[best_minor_idx]
    
    # Determine if major or minor
    if best_major_corr > best_minor_corr:
        return PITCHES[best_major_idx], 'Major', best_major_corr
    else:
        return PITCHES[best_minor_idx], 'Minor', best_minor_corr

def estimate_key_simple(chroma):
    """
    Simple key estimation based on chroma energy
    Fallback method if correlation fails
    """
    # Find the most prominent pitch class
    root_idx = np.argmax(chroma)
    root_note = PITCHES[root_idx]
    
    # Simple heuristic: check if minor third is prominent
    minor_third_idx = (root_idx + 3) % 12
    major_third_idx = (root_idx + 4) % 12
    
    minor_third_strength = chroma[minor_third_idx]
    major_third_strength = chroma[major_third_idx]
    
    if minor_third_strength > major_third_strength:
        return root_note, 'Minor', 0.5
    else:
        return root_note, 'Major', 0.5

def preprocess_audio(y, sr):
    """
    Enhanced audio preprocessing for better BPM detection
    """
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Apply high-pass filter to remove very low frequencies that can interfere
    nyquist = sr / 2
    low_cutoff = 80 / nyquist
    b, a = butter(4, low_cutoff, btype='high')
    y = filtfilt(b, a, y)
    
    # Apply gentle compression to even out dynamics
    y = np.sign(y) * np.power(np.abs(y), 0.7)
    
    return y

def enhanced_onset_detection(y, sr, hop_length=512):
    """
    Enhanced onset detection with multiple onset strength functions
    """
    # Multiple onset strength functions
    onset_strengths = []
    
    # 1. Spectral flux (default)
    onset_flux = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, feature=librosa.feature.spectral_centroid
    )
    onset_strengths.append(('spectral_flux', onset_flux))
    
    # 2. Complex domain onset detection
    onset_complex = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, feature=librosa.stft
    )
    onset_strengths.append(('complex', onset_complex))
    
    # 3. Percussive component
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    onset_percussive = librosa.onset.onset_strength(
        y=y_percussive, sr=sr, hop_length=hop_length
    )
    onset_strengths.append(('percussive', onset_percussive))
    
    return onset_strengths

def detect_bpm_enhanced(y, sr):
    """
    Enhanced BPM detection with improved accuracy
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict: BPM analysis results
    """
    # Preprocess audio
    y = preprocess_audio(y, sr)
    
    # Ensure we have enough audio
    min_duration = 10.0
    if len(y) < min_duration * sr:
        print(f"Warning: Audio is only {len(y)/sr:.1f}s long, BPM detection may be unreliable")
    
    # Use up to 90 seconds for better analysis
    max_samples = int(90 * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
        print(f"Using first 90 seconds of audio for BPM analysis")
    
    all_bpms = []
    all_confidences = []
    method_names = []
    
    # Method 1: Enhanced beat tracking with multiple configurations
    hop_lengths = [256, 512, 1024]  # Try different hop lengths
    for hop_length in hop_lengths:
        try:
            # Try different start BPMs
            start_bpms = [90, 120, 140]
            for start_bpm in start_bpms:
                tempo, beats = librosa.beat.beat_track(
                    y=y, 
                    sr=sr, 
                    hop_length=hop_length,
                    start_bpm=start_bpm,
                    trim=False,
                    units='frames'
                )
                
                # Handle array vs scalar tempo
                if hasattr(tempo, '__len__'):
                    tempo = tempo[0] if len(tempo) > 0 else 120.0
                
                if 60 <= tempo <= 220 and len(beats) > 5:
                    # Calculate confidence based on beat consistency
                    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                    intervals = np.diff(beat_times)
                    
                    if len(intervals) > 3:
                        # Check interval consistency
                        median_interval = np.median(intervals)
                        valid_intervals = intervals[
                            (intervals >= median_interval * 0.8) & 
                            (intervals <= median_interval * 1.2)
                        ]
                        
                        if len(valid_intervals) > len(intervals) * 0.6:  # At least 60% consistent
                            consistency = len(valid_intervals) / len(intervals)
                            all_bpms.append(float(tempo))
                            all_confidences.append(consistency * 0.8)
                            method_names.append(f'beat_track_h{hop_length}_s{start_bpm}')
                            
                            # Also calculate from intervals
                            refined_interval = np.median(valid_intervals)
                            interval_bpm = 60.0 / refined_interval
                            if 60 <= interval_bpm <= 220:
                                all_bpms.append(float(interval_bpm))
                                all_confidences.append(consistency * 0.9)
                                method_names.append(f'intervals_h{hop_length}_s{start_bpm}')
        except Exception as e:
            print(f"Beat tracking failed for hop_length={hop_length}: {e}")
    
    # Method 2: Enhanced onset-based detection
    try:
        onset_strengths = enhanced_onset_detection(y, sr, hop_length=512)
        
        for onset_name, onset_strength in onset_strengths:
            # Peak detection on onset strength
            peaks, properties = find_peaks(
                onset_strength, 
                height=np.mean(onset_strength) + 0.5 * np.std(onset_strength),
                distance=int(0.1 * sr / 512)  # Minimum 100ms between peaks
            )
            
            if len(peaks) > 10:
                peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
                intervals = np.diff(peak_times)
                
                # Filter reasonable intervals
                valid_intervals = intervals[
                    (intervals >= 0.25) & (intervals <= 1.2)
                ]
                
                if len(valid_intervals) > 8:
                    # Use clustering to find dominant interval
                    # Round to nearest 0.01 seconds for clustering
                    rounded_intervals = np.round(valid_intervals, 2)
                    interval_counts = Counter(rounded_intervals)
                    
                    # Get the most common interval
                    dominant_interval = interval_counts.most_common(1)[0][0]
                    dominant_count = interval_counts.most_common(1)[0][1]
                    
                    if dominant_count >= 3:  # At least 3 occurrences
                        onset_bpm = 60.0 / dominant_interval
                        if 60 <= onset_bpm <= 220:
                            confidence = dominant_count / len(valid_intervals)
                            all_bpms.append(float(onset_bpm))
                            all_confidences.append(confidence * 0.7)
                            method_names.append(f'onset_{onset_name}')
    except Exception as e:
        print(f"Onset method failed: {e}")
    
    # Method 3: Autocorrelation with multiple window sizes
    try:
        # Use spectral flux onset strength
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        
        # Try different autocorrelation window sizes
        for window_factor in [1.0, 1.5, 2.0]:
            window_size = int(len(onset_strength) * window_factor)
            if window_size > len(onset_strength):
                window_size = len(onset_strength)
            
            # Autocorrelation on a window
            autocorr = librosa.autocorrelate(onset_strength[:window_size])
            
            # Convert to BPM range
            min_lag = int(60 * sr / (512 * 200))  # 200 BPM max
            max_lag = int(60 * sr / (512 * 60))   # 60 BPM min
            
            if max_lag < len(autocorr) and min_lag < max_lag:
                autocorr_section = autocorr[min_lag:max_lag]
                
                # Find multiple peaks
                peaks, properties = find_peaks(
                    autocorr_section,
                    height=np.mean(autocorr_section) + 0.3 * np.std(autocorr_section),
                    distance=int(0.05 * sr / 512)
                )
                
                if len(peaks) > 0:
                    # Get top peaks
                    peak_heights = autocorr_section[peaks]
                    top_peaks = peaks[np.argsort(peak_heights)[-3:]]  # Top 3 peaks
                    
                    for peak_idx in top_peaks:
                        peak_lag = peak_idx + min_lag
                        peak_bpm = 60 * sr / (512 * peak_lag)
                        
                        if 60 <= peak_bpm <= 200:
                            # Confidence based on peak prominence
                            peak_strength = autocorr_section[peak_idx] / np.max(autocorr_section)
                            all_bpms.append(float(peak_bpm))
                            all_confidences.append(peak_strength * 0.6)
                            method_names.append(f'autocorr_w{window_factor}')
    except Exception as e:
        print(f"Autocorrelation method failed: {e}")
    
    # Method 4: Tempogram analysis
    try:
        # Use librosa's tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
            sr=sr,
            hop_length=512
        )
        
        # Find dominant tempo from tempogram
        tempo_axis = librosa.tempo_frequencies(n_bins=tempogram.shape[0], sr=sr, hop_length=512)
        tempo_strength = np.mean(tempogram, axis=1)
        
        # Find peaks in tempo strength
        peaks, properties = find_peaks(
            tempo_strength,
            height=np.mean(tempo_strength) + 0.5 * np.std(tempo_strength),
            distance=5
        )
        
        if len(peaks) > 0:
            for peak_idx in peaks:
                peak_tempo = tempo_axis[peak_idx]
                if 60 <= peak_tempo <= 200:
                    peak_strength = tempo_strength[peak_idx] / np.max(tempo_strength)
                    all_bpms.append(float(peak_tempo))
                    all_confidences.append(peak_strength * 0.8)
                    method_names.append('tempogram')
    except Exception as e:
        print(f"Tempogram method failed: {e}")
    
    # If no methods worked, use simple fallback
    if not all_bpms:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            if hasattr(tempo, '__len__'):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            return {
                'bpm': float(tempo),
                'confidence': 0.3,
                'method': 'simple_fallback',
                'all_estimates': [float(tempo)],
                'method_details': ['simple_fallback']
            }
        except:
            return {
                'bpm': 120.0,
                'confidence': 0.0,
                'method': 'default',
                'all_estimates': [120.0],
                'method_details': ['default']
            }
    
    # Convert to numpy arrays
    all_bpms = np.array([float(x) for x in all_bpms])
    all_confidences = np.array([float(x) for x in all_confidences])
    
    print(f"Found {len(all_bpms)} BPM estimates: {all_bpms}")
    print(f"Methods: {method_names}")
    print(f"Confidences: {all_confidences}")
    
    # Improved clustering and selection
    if len(all_bpms) > 1:
        # Cluster similar BPMs
        clusters = []
        cluster_confidences = []
        
        for bpm, confidence in zip(all_bpms, all_confidences):
            # Find cluster for this BPM
            assigned = False
            for i, cluster in enumerate(clusters):
                cluster_center = np.mean(cluster)
                if abs(bpm - cluster_center) <= 4:  # Within 4 BPM
                    cluster.append(bpm)
                    cluster_confidences[i].append(confidence)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([bpm])
                cluster_confidences.append([confidence])
        
        # Evaluate clusters
        cluster_scores = []
        for cluster, confidences in zip(clusters, cluster_confidences):
            cluster_mean = np.mean(cluster)
            cluster_size = len(cluster)
            avg_confidence = np.mean(confidences)
            
            # Score based on size, confidence, and being in sweet spot
            size_score = min(cluster_size / 3, 1.0)  # Normalize by expected max
            confidence_score = avg_confidence
            
            # Prefer tempos in the sweet spot (90-150 BPM)
            if 90 <= cluster_mean <= 150:
                sweet_spot_score = 1.0
            elif 70 <= cluster_mean <= 180:
                sweet_spot_score = 0.8
            else:
                sweet_spot_score = 0.6
            
            total_score = size_score * confidence_score * sweet_spot_score
            cluster_scores.append(total_score)
        
        # Select best cluster
        best_cluster_idx = np.argmax(cluster_scores)
        best_cluster = clusters[best_cluster_idx]
        best_confidences = cluster_confidences[best_cluster_idx]
        
        # Use weighted average within the best cluster
        final_bpm = np.average(best_cluster, weights=best_confidences)
        final_confidence = np.mean(best_confidences)
        
        print(f"Best cluster: {best_cluster} -> {final_bpm:.1f}")
        
        # Final validation and correction
        # Check for obvious half/double tempo issues
        if final_bpm > 160:
            half_tempo = final_bpm / 2
            if 80 <= half_tempo <= 140:
                print(f"Correcting from {final_bpm:.1f} to {half_tempo:.1f} (half-time)")
                final_bpm = half_tempo
        elif final_bpm < 80:
            double_tempo = final_bpm * 2
            if 90 <= double_tempo <= 160:
                print(f"Correcting from {final_bpm:.1f} to {double_tempo:.1f} (double-time)")
                final_bpm = double_tempo
    else:
        final_bpm = all_bpms[0]
        final_confidence = all_confidences[0]
    
    print(f"Final BPM: {final_bpm:.1f} (confidence: {final_confidence:.2f})")
    
    return {
        'bpm': float(final_bpm),
        'confidence': float(final_confidence),
        'method': 'enhanced_multi_method',
        'all_estimates': [float(x) for x in sorted(set(np.round(all_bpms, 1)))],
        'method_details': method_names
    }

def analyze_harmony(y, sr):
    """
    Analyze harmonic content for key detection
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict: analysis results
    """
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Average chroma across time
    chroma_mean = np.mean(chroma, axis=1)
    
    # Try sophisticated key detection
    try:
        root, mode, confidence = detect_key(chroma_mean)
        if np.isnan(confidence):
            raise ValueError("NaN confidence")
    except:
        # Fallback to simple method
        root, mode, confidence = estimate_key_simple(chroma_mean)
    
    return {
        'root': root,
        'mode': mode,
        'confidence': float(confidence),
        'chroma_vector': chroma_mean.tolist()
    }

# Test route to verify server is working
@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'Flask server is running!'})

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Received request to /analyze")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = 'temp_audio.wav'
    
    try:
        # Save uploaded file temporarily
        file.save(temp_path)
        print(f"Saved file: {temp_path}")

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)
        print(f"Loaded audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.1f} seconds)")
        
        # Analyze BPM with enhanced method
        bpm_analysis = detect_bpm_enhanced(y, sr)
        
        # Analyze key and mode
        harmony_analysis = analyze_harmony(y, sr)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        result = {
            'bpm': int(np.round(bpm_analysis['bpm'])),
            'bpm_confidence': bpm_analysis['confidence'],
            'key': f"{harmony_analysis['root']} {harmony_analysis['mode']}",
            'root_note': harmony_analysis['root'],
            'mode': harmony_analysis['mode'],
            'key_confidence': harmony_analysis['confidence']
        }
        
        print(f"Analysis result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-detailed', methods=['POST'])
def analyze_detailed():
    """Extended analysis with more harmonic information"""
    print("Received request to /analyze-detailed")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = 'temp_audio.wav'
    
    try:
        file.save(temp_path)
        print(f"Saved file: {temp_path}")

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)
        print(f"Loaded audio: {len(y)} samples at {sr} Hz ({len(y)/sr:.1f} seconds)")
        
        # Basic analysis
        bpm_analysis = detect_bpm_enhanced(y, sr)
        harmony_analysis = analyze_harmony(y, sr)
        
        # Additional harmonic analysis
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Get top 3 most prominent pitch classes
        top_pitches_idx = np.argsort(chroma_mean)[-3:][::-1]
        top_pitches = [PITCHES[i] for i in top_pitches_idx]
        top_strengths = [float(chroma_mean[i]) for i in top_pitches_idx]
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        result = {
            'bpm': int(np.round(bpm_analysis['bpm'])),
            'bpm_confidence': bpm_analysis['confidence'],
            'bpm_all_estimates': [int(np.round(t)) for t in bpm_analysis['all_estimates']],
            'bpm_methods': bpm_analysis.get('method_details', []),
            'key': f"{harmony_analysis['root']} {harmony_analysis['mode']}",
            'root_note': harmony_analysis['root'],
            'mode': harmony_analysis['mode'],
            'key_confidence': harmony_analysis['confidence'],
            'top_pitches': top_pitches,
            'pitch_strengths': top_strengths,
            'chroma_profile': [float(x) for x in chroma_mean]
        }
        
        print(f"Detailed analysis result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    # Run without debug mode to prevent restart loops
    app.run(debug=False, port=5001, host='0.0.0.0')