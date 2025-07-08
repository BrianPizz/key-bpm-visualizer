'use client';
import { useState } from 'react';
import WaveformViewer from '@/components/WaveformViewer';
import PianoRoll from '@/components/PianoRoll';

export default function Home() {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setAudioUrl(URL.createObjectURL(file));
  };

  // placeholder values for scale and BPM
  const dummyScale = ['A', 'C', 'D', 'E', 'G']; 
  const dummyBPM = 90;

  return (
    <main className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">🎼 Key & BPM Visualizer</h1>
      <input type="file" accept="audio/*" onChange={handleUpload} />
      {audioUrl && (
        <>
          <WaveformViewer audioUrl={audioUrl} bpm={dummyBPM} />
          <PianoRoll scaleNotes={dummyScale} />
        </>
      )}
    </main>
  );
}
