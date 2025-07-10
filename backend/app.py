from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os
from scipy.signal import find_peaks
from scipy.stats import mode

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

def detect_bpm_robust(y, sr):
    """
    Robust BPM detection using multiple methods and validation
    
    Args:
        y: audio time series
        sr: sample rate
    
    Returns:
        dict: BPM analysis results
    """
    # Ensure we have enough audio (at least 10 seconds for reliable BPM)
    min_duration = 10.0
    if len(y) < min_duration * sr:
        print(f"Warning: Audio is only {len(y)/sr:.1f}s long, BPM detection may be unreliable")
    
    # Use first 60 seconds max for efficiency
    max_samples = int(60 * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
        print(f"Using first 60 seconds of audio for BPM analysis")
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    all_bpms = []
    all_confidences = []
    method_names = []
    
    # Method 1: Standard librosa beat tracking
    try:
        # Use multiple hop lengths for better accuracy
        for hop_length in [256, 512, 1024]:
            tempo, beats = librosa.beat.beat_track(
                y=y, 
                sr=sr, 
                hop_length=hop_length,
                start_bpm=120.0,
                trim=False
            )
            
            if 60 <= tempo <= 200 and len(beats) > 3:
                all_bpms.append(float(tempo))
                all_confidences.append(0.7)
                method_names.append(f'beat_track_hop{hop_length}')
                
                # Also calculate from beat intervals
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                if len(beat_times) > 3:
                    intervals = np.diff(beat_times)
                    # Remove outliers
                    q75, q25 = np.percentile(intervals, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        valid_intervals = intervals[
                            (intervals >= q25 - 1.5*iqr) & 
                            (intervals <= q75 + 1.5*iqr)
                        ]
                        if len(valid_intervals) > 2:
                            median_interval = np.median(valid_intervals)
                            if median_interval > 0:
                                interval_bpm = 60.0 / median_interval
                                if 60 <= interval_bpm <= 200:
                                    all_bpms.append(interval_bpm)
                                    # Higher confidence for consistent intervals
                                    consistency = 1.0 - (np.std(valid_intervals) / median_interval)
                                    all_confidences.append(max(0.3, consistency))
                                    method_names.append(f'intervals_hop{hop_length}')
    except Exception as e:
        print(f"Beat tracking method failed: {e}")
    
    # Method 2: Onset-based detection with improved parameters
    try:
        # Multiple onset detection methods
        for onset_method in ['default', 'energy', 'hfc']:
            try:
                onsets = librosa.onset.onset_detect(
                    y=y, 
                    sr=sr,
                    hop_length=512,
                    delta=0.05,
                    wait=int(0.1 * sr / 512),
                    units='frames'
                )
                
                if len(onsets) > 8:  # Need sufficient onsets
                    onset_times = librosa.frames_to_time(onsets, sr=sr)
                    intervals = np.diff(onset_times)
                    
                    # Filter intervals to reasonable beat ranges
                    valid_intervals = intervals[
                        (intervals >= 0.3) & (intervals <= 1.0)  # 60-200 BPM
                    ]
                    
                    if len(valid_intervals) > 5:
                        # Use histogram to find most common interval
                        hist, bins = np.histogram(valid_intervals, bins=40)
                        if len(hist) > 0:
                            peak_idx = np.argmax(hist)
                            dominant_interval = (bins[peak_idx] + bins[peak_idx + 1]) / 2
                            
                            if dominant_interval > 0:
                                onset_bpm = 60.0 / dominant_interval
                                if 60 <= onset_bpm <= 200:
                                    all_bpms.append(onset_bpm)
                                    # Confidence based on peak dominance
                                    confidence = hist[peak_idx] / len(valid_intervals)
                                    all_confidences.append(min(0.8, confidence))
                                    method_names.append(f'onset_{onset_method}')
            except:
                continue
    except Exception as e:
        print(f"Onset method failed: {e}")
    
    # Method 3: Autocorrelation of onset strength
    try:
        # Calculate onset strength with different parameters
        for aggregate in [np.mean, np.median]:
            onset_strength = librosa.onset.onset_strength(
                y=y, 
                sr=sr, 
                hop_length=512,
                aggregate=aggregate
            )
            
            if len(onset_strength) > 100:
                # Autocorrelation
                autocorr = librosa.autocorrelate(onset_strength)
                
                # Convert to BPM range
                min_lag = int(60 * sr / (512 * 200))  # 200 BPM
                max_lag = int(60 * sr / (512 * 60))   # 60 BPM
                
                if max_lag < len(autocorr) and min_lag < max_lag:
                    autocorr_section = autocorr[min_lag:max_lag]
                    
                    # Find peaks
                    peaks, properties = find_peaks(
                        autocorr_section,
                        height=0.2 * np.max(autocorr_section),
                        distance=max(1, int(0.05 * sr / 512))
                    )
                    
                    if len(peaks) > 0:
                        # Convert strongest peaks to BPM
                        peak_lags = peaks + min_lag
                        peak_bpms = 60 * sr / (512 * peak_lags)
                        peak_strengths = properties['peak_heights']
                        
                        # Sort by strength
                        sorted_indices = np.argsort(peak_strengths)[::-1]
                        
                        for i, idx in enumerate(sorted_indices[:3]):
                            if idx < len(peak_bpms):
                                bpm = peak_bpms[idx]
                                if 60 <= bpm <= 200:
                                    all_bpms.append(float(bpm))
                                    confidence = peak_strengths[idx] / np.max(peak_strengths)
                                    all_confidences.append(confidence * 0.6)
                                    method_names.append(f'autocorr_{aggregate.__name__}')
    except Exception as e:
        print(f"Autocorrelation method failed: {e}")
    
    # Method 4: Spectral-based tempo estimation
    try:
        # Use spectral features to estimate tempo
        stft = librosa.stft(y, hop_length=512)
        magnitude = np.abs(stft)
        
        # Calculate spectral flux
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        
        # Find peaks in spectral flux
        flux_peaks, _ = find_peaks(spectral_flux, height=np.percentile(spectral_flux, 70))
        
        if len(flux_peaks) > 10:
            peak_times = librosa.frames_to_time(flux_peaks, sr=sr, hop_length=512)
            intervals = np.diff(peak_times)
            
            # Filter to reasonable beat intervals
            valid_intervals = intervals[
                (intervals >= 0.3) & (intervals <= 1.0)
            ]
            
            if len(valid_intervals) > 5:
                # Cluster intervals to find dominant tempo
                hist, bins = np.histogram(valid_intervals, bins=30)
                peak_idx = np.argmax(hist)
                dominant_interval = (bins[peak_idx] + bins[peak_idx + 1]) / 2
                
                if dominant_interval > 0:
                    spectral_bpm = 60.0 / dominant_interval
                    if 60 <= spectral_bpm <= 200:
                        all_bpms.append(spectral_bpm)
                        all_confidences.append(0.5)
                        method_names.append('spectral_flux')
    except Exception as e:
        print(f"Spectral method failed: {e}")
    
    # If no methods worked, try simple fallback
    if not all_bpms:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return {
                'bpm': float(tempo),
                'confidence': 0.2,
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
    
    # Process all BPM estimates
    all_bpms = np.array(all_bpms)
    all_confidences = np.array(all_confidences)
    
    print(f"Found {len(all_bpms)} BPM estimates: {all_bpms}")
    print(f"Methods: {method_names}")
    
    # Handle tempo multiples (common issue)
    corrected_bpms = []
    corrected_confidences = []
    corrected_methods = []
    
    for i, bpm in enumerate(all_bpms):
        corrected_bpms.append(bpm)
        corrected_confidences.append(all_confidences[i])
        corrected_methods.append(method_names[i])
        
        # Check for double-time (if BPM is very high, try half)
        if bpm > 160:
            half_bpm = bpm / 2
            if 70 <= half_bpm <= 140:
                corrected_bpms.append(half_bpm)
                corrected_confidences.append(all_confidences[i] * 0.9)
                corrected_methods.append(f"{method_names[i]}_half")
        
        # Check for half-time (if BPM is low, try double)
        if bpm < 90:
            double_bpm = bpm * 2
            if 120 <= double_bpm <= 180:
                corrected_bpms.append(double_bpm)
                corrected_confidences.append(all_confidences[i] * 0.9)
                corrected_methods.append(f"{method_names[i]}_double")
    
    corrected_bpms = np.array(corrected_bpms)
    corrected_confidences = np.array(corrected_confidences)
    
    # Cluster similar BPMs and find consensus
    if len(corrected_bpms) > 1:
        # Round to nearest integer for clustering
        rounded_bpms = np.round(corrected_bpms)
        
        # Find clusters within ±3 BPM
        unique_bpms = np.unique(rounded_bpms)
        cluster_scores = []
        
        for target_bpm in unique_bpms:
            # Find all BPMs within ±3 of target
            close_indices = np.where(np.abs(corrected_bpms - target_bpm) <= 3)[0]
            
            if len(close_indices) > 0:
                cluster_bpms = corrected_bpms[close_indices]
                cluster_confidences = corrected_confidences[close_indices]
                
                # Score based on number of methods + total confidence
                cluster_score = len(close_indices) + np.sum(cluster_confidences)
                cluster_scores.append({
                    'bpm': np.average(cluster_bpms, weights=cluster_confidences),
                    'confidence': np.mean(cluster_confidences),
                    'support': len(close_indices),
                    'score': cluster_score
                })
        
        if cluster_scores:
            # Choose best cluster
            best_cluster = max(cluster_scores, key=lambda x: x['score'])
            final_bpm = best_cluster['bpm']
            final_confidence = min(0.95, best_cluster['confidence'] * (1 + 0.1 * best_cluster['support']))
        else:
            # Fallback to weighted average
            final_bpm = np.average(corrected_bpms, weights=corrected_confidences)
            final_confidence = np.mean(corrected_confidences)
    else:
        final_bpm = corrected_bpms[0]
        final_confidence = corrected_confidences[0]
    
    # Final validation and adjustment
    if final_bpm < 70:
        final_bpm = final_bpm * 2
        final_confidence *= 0.8
    elif final_bpm > 180:
        final_bpm = final_bpm / 2
        final_confidence *= 0.8
    
    return {
        'bpm': float(final_bpm),
        'confidence': float(final_confidence),
        'method': 'robust_multi_method',
        'all_estimates': [float(x) for x in sorted(set(np.round(corrected_bpms, 1)))],
        'method_details': corrected_methods
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
        
        # Analyze BPM with robust method
        bpm_analysis = detect_bpm_robust(y, sr)
        
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
        bpm_analysis = detect_bpm_robust(y, sr)
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