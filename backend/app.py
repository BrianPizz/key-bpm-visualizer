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
    
    # Method 1: Standard librosa beat tracking with single best hop length
    try:
        # Use 512 hop length as it's most reliable for most music
        tempo, beats = librosa.beat.beat_track(
            y=y, 
            sr=sr, 
            hop_length=512,
            start_bpm=120.0,
            trim=False
        )
        
        # Ensure tempo is a scalar
        if hasattr(tempo, '__len__'):
            tempo = tempo[0] if len(tempo) > 0 else 120.0
        
        if 60 <= tempo <= 200 and len(beats) > 3:
            all_bpms.append(float(tempo))
            all_confidences.append(0.8)
            method_names.append('beat_track_primary')
            
            # Calculate from beat intervals for verification
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
            if len(beat_times) > 4:
                intervals = np.diff(beat_times)
                # Remove outliers more aggressively
                median_interval = np.median(intervals)
                valid_intervals = intervals[
                    (intervals >= median_interval * 0.7) & 
                    (intervals <= median_interval * 1.3)
                ]
                
                if len(valid_intervals) > 3:
                    refined_interval = np.median(valid_intervals)
                    interval_bpm = 60.0 / refined_interval
                    if 60 <= interval_bpm <= 200:
                        all_bpms.append(float(interval_bpm))
                        # Higher confidence for consistent intervals
                        consistency = 1.0 - (np.std(valid_intervals) / refined_interval)
                        all_confidences.append(float(max(0.4, consistency * 0.9)))
                        method_names.append('beat_intervals_refined')
    except Exception as e:
        print(f"Primary beat tracking failed: {e}")
    
    # Method 2: Onset-based detection (simplified)
    try:
        onsets = librosa.onset.onset_detect(
            y=y, 
            sr=sr,
            hop_length=512,
            delta=0.07,
            wait=int(0.15 * sr / 512),
            units='frames'
        )
        
        if len(onsets) > 10:
            onset_times = librosa.frames_to_time(onsets, sr=sr)
            intervals = np.diff(onset_times)
            
            # Focus on the most common interval
            valid_intervals = intervals[
                (intervals >= 0.3) & (intervals <= 1.0)
            ]
            
            if len(valid_intervals) > 8:
                # Use histogram with fewer bins for cleaner peaks
                hist, bins = np.histogram(valid_intervals, bins=20)
                peak_idx = np.argmax(hist)
                dominant_interval = (bins[peak_idx] + bins[peak_idx + 1]) / 2
                
                if dominant_interval > 0:
                    onset_bpm = 60.0 / dominant_interval
                    if 60 <= onset_bpm <= 200:
                        all_bpms.append(float(onset_bpm))
                        # Confidence based on peak dominance
                        peak_dominance = hist[peak_idx] / len(valid_intervals)
                        all_confidences.append(float(min(0.7, peak_dominance)))
                        method_names.append('onset_histogram')
    except Exception as e:
        print(f"Onset method failed: {e}")
    
    # Method 3: Simplified autocorrelation
    try:
        onset_strength = librosa.onset.onset_strength(
            y=y, 
            sr=sr, 
            hop_length=512
        )
        
        if len(onset_strength) > 100:
            # Autocorrelation
            autocorr = librosa.autocorrelate(onset_strength)
            
            # Convert to BPM range
            min_lag = int(60 * sr / (512 * 180))  # 180 BPM max
            max_lag = int(60 * sr / (512 * 80))   # 80 BPM min
            
            if max_lag < len(autocorr) and min_lag < max_lag:
                autocorr_section = autocorr[min_lag:max_lag]
                
                # Find the strongest peak
                peak_idx = np.argmax(autocorr_section)
                peak_lag = peak_idx + min_lag
                peak_bpm = 60 * sr / (512 * peak_lag)
                
                if 80 <= peak_bpm <= 180:
                    all_bpms.append(float(peak_bpm))
                    # Confidence based on peak strength
                    peak_strength = autocorr_section[peak_idx] / np.max(autocorr_section)
                    all_confidences.append(float(peak_strength * 0.6))
                    method_names.append('autocorr_peak')
    except Exception as e:
        print(f"Autocorrelation method failed: {e}")
    
    # If no methods worked, try simple fallback
    if not all_bpms:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
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
    
    # Convert to numpy arrays - ensure all elements are scalars
    all_bpms = np.array([float(x) for x in all_bpms])
    all_confidences = np.array([float(x) for x in all_confidences])
    
    print(f"Found {len(all_bpms)} BPM estimates: {all_bpms}")
    print(f"Methods: {method_names}")
    print(f"Confidences: {all_confidences}")
    
    # Simple clustering approach
    if len(all_bpms) > 1:
        # Find the BPM with highest confidence
        primary_idx = np.argmax(all_confidences)
        primary_bpm = all_bpms[primary_idx]
        primary_confidence = all_confidences[primary_idx]
        
        print(f"Primary BPM candidate: {primary_bpm:.1f} (confidence: {primary_confidence:.2f})")
        
        # Find all BPMs within ±5 of the primary
        close_mask = np.abs(all_bpms - primary_bpm) <= 5
        close_bpms = all_bpms[close_mask]
        close_confidences = all_confidences[close_mask]
        
        if len(close_bpms) > 1:
            # Use weighted average of close BPMs
            final_bpm = np.average(close_bpms, weights=close_confidences)
            final_confidence = np.mean(close_confidences)
            print(f"Averaged {len(close_bpms)} close estimates: {final_bpm:.1f}")
        else:
            final_bpm = primary_bpm
            final_confidence = primary_confidence
            print(f"Using primary estimate: {final_bpm:.1f}")
        
        # Check for obvious tempo multiples
        other_bpms = all_bpms[~close_mask]
        if len(other_bpms) > 0:
            for other_bpm in other_bpms:
                ratio = final_bpm / other_bpm
                if 1.8 <= ratio <= 2.2:  # final_bpm is roughly double other_bpm
                    if 90 <= other_bpm <= 140:  # other_bpm is in sweet spot
                        print(f"Considering half-time: {other_bpm:.1f} instead of {final_bpm:.1f}")
                        final_bpm = other_bpm
                        final_confidence *= 0.9
                        break
                elif 0.45 <= ratio <= 0.55:  # final_bpm is roughly half other_bpm
                    if 90 <= other_bpm <= 140:  # other_bpm is in sweet spot
                        print(f"Considering double-time: {other_bpm:.1f} instead of {final_bpm:.1f}")
                        final_bpm = other_bpm
                        final_confidence *= 0.9
                        break
    else:
        final_bpm = all_bpms[0]
        final_confidence = all_confidences[0]
    
    print(f"Final BPM: {final_bpm:.1f} (confidence: {final_confidence:.2f})")
    
    return {
        'bpm': float(final_bpm),
        'confidence': float(final_confidence),
        'method': 'robust_simplified',
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