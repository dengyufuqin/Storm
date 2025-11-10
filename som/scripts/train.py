import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from trainning.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train SAM Multiview model on YCB-V dataset"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Path to the training config YAML file"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from"
    )
    args = parser.parse_args()

    trainer = Trainer(cfg_path=args.config, resume_ckpt=args.resume)
    trainer.run()


if __name__ == "__main__":
    main()

