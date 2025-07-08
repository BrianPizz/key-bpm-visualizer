type Props = {
    scaleNotes: string[];
  };
  
  const ALL_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B'];
  
  export default function PianoRoll({ scaleNotes }: Props) {
    return (
      <div className="flex w-full justify-center mt-2">
        {ALL_NOTES.map((note) => (
          <div
            key={note}
            className={`w-10 h-16 text-sm flex items-center justify-center border
                        ${scaleNotes.includes(note) ? 'bg-green-400' : 'bg-gray-200'}`}
          >
            {note}
          </div>
        ))}
      </div>
    );
  }
  