from pathlib import Path
import argparse
import shutil
import itertools
from pydub import AudioSegment


def process(args):

    # Check if the audio and segments directories exist
    audio_dir = Path(args.europarl_data_root) / args.src_lang / "audios"
    segments_dir = Path(args.europarl_data_root) / args.src_lang / \
        args.tgt_lang / args.split
    assert audio_dir.exists(), f"Audio directory {audio_dir} does not exist"
    assert segments_dir.exists(), f"Segments directory {segments_dir} does not exist"

    # Create the output directory
    output_dir = Path(args.output_data_root) / \
        f"{args.src_lang}-{args.tgt_lang}" / "data" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the segments.lst file
    with open(segments_dir / "segments.lst", "r", encoding="utf-8") as file:
        segments_data = file.readlines()

    # Convert segments.lst data to the MuST-C's yaml format
    converted_data = []
    for line in segments_data:
        tokens = line.strip().split()
        audio_file = tokens[0]
        start_time = float(tokens[1])
        end_time = float(tokens[2])
        duration = end_time - start_time

        audio_file_name = f"{audio_file}.{args.audio_format}"

        entry = {
            "duration": duration,
            "offset": start_time,
            "speaker_id": audio_file,
            args.audio_format: audio_file_name,
        }
        
        converted_data.append(entry)

    formatted_data_str_list = []
    for entry in converted_data:
        formatted_line = (
            f"- {{duration: {entry['duration']:.6f}, "
            f"offset: {entry['offset']:.6f}, "
            f"speaker_id: {entry['speaker_id']}, "
            f"{args.audio_format}: {entry[args.audio_format]}}}"
        )
        formatted_data_str_list.append(formatted_line)

    # Save yaml and text files
    output_text_dir = output_dir / "txt"
    output_text_dir.mkdir(parents=True, exist_ok=True)
    output_yaml_filename = output_text_dir / f"{args.split}.{args.audio_format}.yaml"
    output_src_filename = output_text_dir / f"{args.split}.{args.src_lang}"
    output_tgt_filename = output_text_dir / f"{args.split}.{args.tgt_lang}"
    formatted_content = "\n".join(formatted_data_str_list)
    with open(output_yaml_filename, "w", encoding="utf-8") as file:
        file.write(formatted_content)
    shutil.copyfile(segments_dir / f"segments.{args.src_lang}", output_src_filename)
    shutil.copyfile(segments_dir / f"segments.{args.tgt_lang}", output_tgt_filename)

    # Save the audio files
    output_audio_dir = output_dir / f"{args.audio_format}"
    output_audio_dir.mkdir(parents=True, exist_ok=False)
    orig_audio_format = audio_dir.glob("*.*").__next__().suffix[1:]
    segments_data = [line.strip().split() for line in segments_data]
    for speech_id, group in itertools.groupby(segments_data, lambda x: x[0]):
        audio_filename = audio_dir / f"{speech_id}.{orig_audio_format}"
        output_audio_filename = output_audio_dir / f"{speech_id}.{args.audio_format}"
        if orig_audio_format != args.audio_format:
            audio = AudioSegment.from_file(audio_filename, orig_audio_format)
            audio = audio.set_frame_rate(args.frame_rate)
            if not args.keep_channels:
                audio = audio.set_channels(1)
            audio.export(output_audio_filename, format=args.audio_format)
        else:
            shutil.copyfile(audio_filename, output_audio_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--europarl-data-root", "-d", type=str, required=True)
    parser.add_argument("--output-data-root", "-o", type=str, required=True)
    parser.add_argument("--src-lang", "-s", type=str, required=True)
    parser.add_argument("--tgt-lang", "-t", type=str, required=True)
    parser.add_argument("--split", "-sp", type=str, required=True)
    parser.add_argument("--audio-format", "-a", type=str, default="m4a", choices=["m4a", "wav"])
    parser.add_argument("--frame-rate", "-r", type=int, default=16000)
    parser.add_argument("--keep-channels", "-k", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == '__main__':
    main()
