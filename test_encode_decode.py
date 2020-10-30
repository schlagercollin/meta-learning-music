import music21 as m21
from collections import defaultdict
from dataset.data_utils import encode, decode, get_vocab


def new_encode(stream, note_min=60, note_max=108):

    # Stores tuples of (instrument, note(s), offset, duration)
    all_elements = []

    instruments = set()

    partitioned = m21.instrument.partitionByInstrument(stream)
    for idx, part in enumerate(partitioned):
        part.write('midi', "part-{}.mid".format(idx+1))
        # print(f"NEW PART with {len(part.flat)} and instrument {next(part.flat).instrumentName}")
        for element in part.flat:
            if isinstance(element, m21.instrument.Instrument):
                cur_instrument = str(element.instrumentName).replace(" ", "")
                instruments.add(cur_instrument)

            elif isinstance(element, m21.note.Note):
                if cur_instrument != "None":
                    all_elements.append((cur_instrument, int(element.pitch.midi), element.offset, element.duration.quarterLength))

            elif isinstance(element, m21.note.Rest):
                if cur_instrument != "None":
                    all_elements.append((cur_instrument, 0, element.offset, element.duration.quarterLength))

            elif isinstance(element, m21.chord.Chord):
                if cur_instrument != "None":
                    all_elements.append((cur_instrument, [int(pitch.midi) for pitch in element.pitches], element.offset, element.duration.quarterLength))

    print(instruments)

    # Now we have to sort all of the notes by offset
    sorted_elements = sorted(all_elements, key=lambda x: x[2])
    encoding = []

    instruments = set()

    for idx, element in enumerate(sorted_elements[:-1]):
        instr, pitches, offset, duration = element
        advance = sorted_elements[idx+1][2] - offset

        instruments.add(instr)

        if isinstance(pitches, list):
            allowed_pitches = [pitch for pitch in pitches if pitch in range(note_min, note_max+1)]

            for pitch in pitches:
                encoding += [instr, pitch, duration, 0]

            # Manually change the last note's advance value, if we actually added any notes
            if len(allowed_pitches) > 0:
                encoding[-1] = advance

        elif isinstance(pitches, int):
            if pitches == 0 or pitches in range(note_min, note_max+1):
                encoding += [instr, pitches, duration, advance]

    print(instruments)

    return encoding

def new_decode(encoding):
    assert len(encoding)%4 == 0

    full_stream = m21.stream.Stream()

    # encoding_by_instrument = defaultdict(list)

    # quadruplets = (encoding[i:i+4] for i in range(0, len(encoding), 4))
    # for instrument_name, pitch, duration, advance in quadruplets:
    #     encoding_by_instrument[instrument_name].append((pitch, duration, advance))

    # for instrument_name, encoding in encoding_by_instrument.items():
    #     inst_stream = m21.stream.Stream()
    #     cur_offset = 0.0

    #     if instrument_name == "Voice":
    #         instrument = m21.instrument.Vocalist()
    #     else:
    #         instrument = eval("m21.instrument.{}()".format(instrument_name))

    #     inst_stream.insert(cur_offset, instrument)

    #     for pitch, duration, advance in encoding:
    #         m21_duration = m21.duration.Duration(duration)
    #         if pitch == 0:
    #             note = m21.note.Rest(duration=m21_duration)
    #         else:
    #             note = m21.note.Note(pitch, duration=m21_duration)

    #         inst_stream.insert(cur_offset, note)
    #         cur_offset += advance

    #     full_stream.append(inst_stream)




    # The offset value of the current note / rest / chord
    cur_offset = 0.0
    prev_offset = 0.0

    instruments = set()

    quadruplets = (encoding[i:i+4] for i in range(0, len(encoding), 4))
    for instrument_name, pitch, duration, advance in quadruplets:
        m21_duration = m21.duration.Duration(duration)

        instruments.add(instrument_name)

        if instrument_name == "Voice":
            instrument = m21.instrument.Vocalist()
        else:
            instrument = eval("m21.instrument.{}()".format(instrument_name))

        if pitch == 0:
            note = m21.note.Rest(duration=m21_duration)
        else:
            note = m21.note.Note(pitch, duration=m21_duration)

        # if cur_offset == 0 or cur_offset != prev_offset:
        full_stream.insert(cur_offset, instrument)
        full_stream.insert(cur_offset, note)

        prev_offset = cur_offset
        cur_offset += advance

    print(instruments)
    return full_stream


stream = m21.converter.parse("./TRBFQAY128E078F6C1-all.mid")
encoding = new_encode(stream)
decoded = new_decode(encoding)
decoded.write('midi', 'new_decode_test.mid')