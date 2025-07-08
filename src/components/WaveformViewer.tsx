import { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/plugins/regions';

type Props = {
  audioUrl: string;
  bpm: number;
};

export default function WaveformViewer({ audioUrl, bpm }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const waveSurfer = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#ccc',
      progressColor: '#10b981',
      barWidth: 2,
      height: 100,
      plugins: [
        RegionsPlugin.create()
      ]
    });

    waveSurfer.load(audioUrl);
    waveSurferRef.current = waveSurfer;

    return () => waveSurfer.destroy();
  }, [audioUrl]);


  useEffect(() => {
    if (!waveSurferRef.current || !bpm) return;
    const beatInterval = 60 / bpm;
    const duration = waveSurferRef.current.getDuration();

    for (let t = 0; t < duration; t += beatInterval) {
      (waveSurferRef.current as any).addRegion({
        start: t,
        end: t + 0.01,
        color: 'rgba(255, 0, 0, 0.3)',
        drag: false,
        resize: false,
      });
    }
  }, [bpm]);

  return <div ref={containerRef} className="w-full my-4" />;
}
