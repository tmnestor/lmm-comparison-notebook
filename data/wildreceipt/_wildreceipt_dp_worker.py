
#!/usr/bin/env python3
"""WildReceipt benchmark DP worker — processes a shard of images on one GPU."""
import argparse
import base64
import io
import json
import os
import sys
import time as _time

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    print(f"[GPU {args.gpu_id}] Loading model ...", flush=True)
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=False,
        enable_prefix_caching=True,
    )

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    with open(args.image_list) as f:
        image_paths = json.load(f)
    with open(args.prompt_file) as f:
        prompt = f.read()

    results = []
    total = len(image_paths)
    print(f"[GPU {args.gpu_id}] Processing {total} images ...", flush=True)

    for idx, img_path in enumerate(image_paths):
        try:
            t0 = _time.time()
            image = Image.open(img_path).convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data_uri = (
                f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            )
            buf.close()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            outputs = model.chat(
                messages=messages, sampling_params=sampling, use_tqdm=False
            )
            text = outputs[0].outputs[0].text.strip()
            elapsed = _time.time() - t0
            del outputs, messages, data_uri, image
            results.append({"image_path": img_path, "response": text, "time": round(elapsed, 3)})
        except Exception as e:
            results.append({"image_path": img_path, "response": "", "error": str(e), "time": 0.0})

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"[GPU {args.gpu_id}] {idx + 1}/{total}", flush=True)

    with open(args.output_file, "w") as f:
        json.dump(results, f)

    del model
    print(f"[GPU {args.gpu_id}] Done.", flush=True)


if __name__ == "__main__":
    main()
