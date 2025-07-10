"use client";
import { useState } from "react";
import WaveformViewer from "@/components/WaveformViewer";
import PianoRoll from "@/components/PianoRoll";

export default function Home() {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [detectedBPM, setDetectedBPM] = useState<number | null>(null);
  const [detectedKey, setDetectedKey] = useState<string | null>(null);

  const keyToScaleMap: { [key: string]: string[] } = {
    // Major Keys
    "C Major": ["C", "D", "E", "F", "G", "A", "B"],
    "C# Major": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
    "D Major": ["D", "E", "F#", "G", "A", "B", "C#"],
    "Eb Major": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "E Major": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "F Major": ["F", "G", "A", "Bb", "C", "D", "E"],
    "F# Major": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
    "G Major": ["G", "A", "B", "C", "D", "E", "F#"],
    "Ab Major": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "A Major": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "Bb Major": ["Bb", "C", "D", "Eb", "F", "G", "A"],
    "B Major": ["B", "C#", "D#", "E", "F#", "G#", "A#"],

    // Minor Keys (Natural Minor / Aeolian Mode)
    "A Minor": ["A", "B", "C", "D", "E", "F", "G"],
    "A# Minor": ["A#", "B#", "C#", "D#", "E#", "F#", "G#"],
    "B Minor": ["B", "C#", "D", "E", "F#", "G", "A"],
    "C Minor": ["C", "D", "Eb", "F", "G", "Ab", "Bb"],
    "C# Minor": ["C#", "D#", "E", "F#", "G#", "A", "B"],
    "D Minor": ["D", "E", "F", "G", "A", "Bb", "C"],
    "D# Minor": ["D#", "E#", "F#", "G#", "A#", "B", "C#"],
    "E Minor": ["E", "F#", "G", "A", "B", "C", "D"],
    "F Minor": ["F", "G", "Ab", "Bb", "C", "Db", "Eb"],
    "F# Minor": ["F#", "G#", "A", "B", "C#", "D", "E"],
    "G Minor": ["G", "A", "Bb", "C", "D", "Eb", "F"],
    "G# Minor": ["G#", "A#", "B", "C#", "D#", "E", "F#"],
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAudioUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:5001/analyze", {
      method: "POST",
      body: formData,
    });

    const { bpm, key } = await res.json();

    console.log("Detected:", bpm, key);
    setDetectedBPM(bpm);
    setDetectedKey(key);
  };

  return (
    <main className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">🎼 Key & BPM Visualizer</h1>
      <input type="file" accept="audio/*" onChange={handleUpload} />
      {audioUrl && (
        <>
          <WaveformViewer audioUrl={audioUrl} bpm={detectedBPM ?? 140} />
          {detectedKey && (
            <PianoRoll scaleNotes={keyToScaleMap[detectedKey] || []} />
          )}
        </>
      )}
    </main>
  );
}
