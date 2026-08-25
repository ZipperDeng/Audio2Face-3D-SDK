import argparse
from .api import audio_to_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audio to an Audio2Face vertex cache")
    parser.add_argument("audio", help="Input WAV file")
    parser.add_argument("output", help="Output .npz cache")
    parser.add_argument("--model", dest="model_path", help="Path to diffusion model.json")
    parser.add_argument("--device", dest="device_id", type=int, default=0)
    parser.add_argument("--identity", dest="identity_index", type=int, default=0)
    parser.add_argument("--constant-noise", action="store_true")
    args = parser.parse_args()
    print(audio_to_cache(args.audio, args.output, model_path=args.model_path,
                         device_id=args.device_id, identity_index=args.identity_index,
                         constant_noise=args.constant_noise))


if __name__ == "__main__":
    main()
