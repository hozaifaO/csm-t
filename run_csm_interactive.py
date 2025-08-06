import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass
import time

# --- Logging and Timing Utilities ---
def log(message):
    """Prints a message with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

class Timer:
    """A simple timer class to measure execution time."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        log(f"Operation completed in {self.interval:.4f} seconds.")

# --- Main Application Logic ---

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
with Timer():
    log("Downloading speaker prompt...")
    prompt_filepath_conversational_a = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    )

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    with Timer():
        log(f"Loading prompt audio from {audio_path}...")
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        log("Resampling prompt audio...")
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
        )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    with Timer():
        log("Preparing speaker prompt...")
        audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def main():
    log("Starting interactive CSM script.")

    with Timer():
        log("Selecting device...")
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        log(f"Using device: {device}")

    with Timer():
        log("Loading CSM model...")
        generator = load_csm_1b(device)

    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )

    log("Model loaded and ready. Enter text to generate speech or 'exit' to quit.")

    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                log("Exiting...")
                break

            with Timer():
                log(f"Generating speech for: '{user_input}'")
                audio_tensor = generator.generate(
                    text=user_input,
                    speaker=0,
                    context=[prompt_a],
                    max_audio_length_ms=10_000,
                )

            output_filename = f"generated_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            with Timer():
                log(f"Saving audio to {output_filename}...")
                torchaudio.save(
                    output_filename,
                    audio_tensor.unsqueeze(0).cpu(),
                    generator.sample_rate
                )
            log(f"Successfully generated {output_filename}")

        except (KeyboardInterrupt, EOFError):
            log("\nExiting...")
            break

if __name__ == "__main__":
    main()
