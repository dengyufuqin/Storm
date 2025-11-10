from transformers import pipeline, AutoProcessor
import torch
import time
import json
import os 
import argparse

class ImageDescriptor:
    """
    Encapsulates the GEMMA image-to-text pipeline for concise keyword-based descriptors.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        device: int | str = 0,
        dtype=torch.bfloat16,
        precision: str = 'high'
    ):
        # Set matmul precision
        torch.set_float32_matmul_precision(precision)
        # Resolve device
        if isinstance(device, int):
            self.device = f"cuda:{device}"
        else:
            self.device = device
        # Load processor and pipeline
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.pipe = pipeline(
            task="image-text-to-text",
            model=model_name,
            processor=self.processor,
            device=self.device,
            torch_dtype=dtype
        )

    def describe(self, image_path: str, max_new_tokens: int = 100) -> str:
        """
        Generate a concise, high-confidence, keyword-only description for the object in the image.

        Args:
            image_path: Path to the local image file.
            max_new_tokens: Maximum tokens for generation.

        Returns:
            Space-separated keywords describing the object.
        """
        # Define an optimized prompt for confident keyword output
        topics = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an image descriptor. Given an image of a single object, "
                            "output only space-separated keywords representing its key attributes "
                            "(e.g. red metal cylinder) without any punctuation or full sentences. "
                            "Include only attributes you are highly confident about; if uncertain, omit them. "
                            "Provide all attributes you are confident in."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": "Provide space-separated keywords only, include only attributes you are certain of."}
                ]
            }
        ]

        # Inference
        start = time.time()
        output = self.pipe(text=topics, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        inf_time = time.time() - start
        print(f"Inference time: {inf_time:.3f}s")

        # Extract and return keywords
        gen = output[0].get("generated_text", [])
        if isinstance(gen, list) and gen:
            return gen[-1].get("content", "").strip()
        return str(output[0].get("generated_text", "")).strip()

def batch_describe_datasets(
    data_root: str,
    model_name: str,
    device,
    max_new_tokens: int
):
    """
    For each subdataset under data_root, looks in its 'render' folder,
    describes each object by its angle_003.png, and writes
    descriptions.json into that subdataset folder.
    """
    descriptor = ImageDescriptor(model_name=model_name, device=device)
    for ds_name in sorted(os.listdir(data_root)):
        ds_path     = os.path.join(data_root, ds_name)
        render_root = os.path.join(ds_path, "render")
        if not os.path.isdir(render_root):
            print(f"Skipping '{ds_name}', no render directory.")
            continue

        results = {}
        # each object folder under render
        for obj_name in sorted(os.listdir(render_root)):
            img_path = os.path.join(render_root, obj_name, "nature", "angle_003.png")
            if not os.path.isfile(img_path):
                print(f"  Missing angle_003.png for object '{obj_name}', skipping.")
                results[obj_name] = None
                continue
            try:
                desc = descriptor.describe(img_path, max_new_tokens=max_new_tokens)
                results[obj_name] = desc
                print(f"  [{ds_name}/{obj_name}] {desc}")
            except Exception as e:
                print(f"  Failed '{ds_name}/{obj_name}': {e}")
                results[obj_name] = None

        out_json = os.path.join(ds_path, "descriptions.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote descriptions for {len(results)} objects to {out_json}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str,
                        required=True,
                        help="Root directory containing subdatasets")
    parser.add_argument("--model_name",   type=str,
                        default="google/gemma-3-12b-it",
                        help="GEMMA model name")
    parser.add_argument("--device",       type=str,
                        default="cuda:2" if torch.cuda.is_available() else "cpu",
                        help="CUDA device or 'cpu'")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens for description generation")
    args = parser.parse_args()

    batch_describe_datasets(
        data_root=args.data_root,
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )

if __name__ == "__main__":
    main()