import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/plugins/regions';

type Props = {
  audioUrl: string;
  bpm: number;
};

export default function WaveformViewer({ audioUrl, bpm }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<RegionsPlugin | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    let isMounted = true;

    const initializeWaveSurfer = async () => {
      try {
        // Cleanup previous instance
        if (waveSurferRef.current) {
          waveSurferRef.current.destroy();
          waveSurferRef.current = null;
        }

        if (!isMounted || !containerRef.current) return;

        // Create regions plugin
        const regionsPlugin = RegionsPlugin.create();
        regionsPluginRef.current = regionsPlugin;

        // Create WaveSurfer instance
        const waveSurfer = WaveSurfer.create({
          container: containerRef.current,
          waveColor: '#ccc',
          progressColor: '#10b981',
          barWidth: 2,
          height: 100,
          plugins: [regionsPlugin]
        });

        if (!isMounted) {
          waveSurfer.destroy();
          return;
        }

        waveSurferRef.current = waveSurfer;

        // Set up event listeners
        waveSurfer.on('ready', () => {
          if (isMounted) {
            setIsReady(true);
          }
        });

        waveSurfer.on('error', (error) => {
          if (isMounted) {
            console.error('WaveSurfer error:', error);
            setIsReady(false);
          }
        });

        // Load audio
        await waveSurfer.load(audioUrl);

      } catch (error) {
        if (isMounted) {
          console.error('Error initializing WaveSurfer:', error);
          setIsReady(false);
        }
      }
    };

    setIsReady(false);
    initializeWaveSurfer();

    return () => {
      isMounted = false;
      if (waveSurferRef.current) {
        try {
          waveSurferRef.current.destroy();
        } catch (error) {
          console.error('Error destroying WaveSurfer:', error);
        }
        waveSurferRef.current = null;
      }
      regionsPluginRef.current = null;
    };
  }, [audioUrl]);

  // Add beat markers when WaveSurfer is ready and BPM is available
  useEffect(() => {
    if (!isReady || !waveSurferRef.current || !regionsPluginRef.current || !bpm) {
      return;
    }

    try {
      // Clear existing regions
      regionsPluginRef.current.clearRegions();

      const beatInterval = 60 / bpm; // Time between beats in seconds
      const duration = waveSurferRef.current.getDuration();

      if (duration > 0) {
        // Add beat markers
        for (let t = 0; t < duration; t += beatInterval) {
          regionsPluginRef.current.addRegion({
            start: t,
            end: t + 0.01, // Very short duration for marker
            color: 'rgba(255, 0, 0, 0.3)',
            drag: false,
            resize: false,
          });
        }
      }
    } catch (error) {
      console.error('Error adding beat regions:', error);
    }
  }, [isReady, bpm]);

  return (
    <div className="w-full my-4">
      {!isReady && (
        <div className="flex items-center justify-center h-24 bg-gray-100 rounded">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-600"></div>
          <span className="ml-2 text-gray-600">Loading waveform...</span>
        </div>
      )}
      <div 
        ref={containerRef} 
        className={`w-full ${!isReady ? 'hidden' : ''}`}
      />
      {isReady && bpm && (
        <div className="mt-2 text-sm text-gray-600 text-center">
          BPM: {bpm}
        </div>
      )}
    </div>
  );
}