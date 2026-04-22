import argparse
import json
import os

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps, StateDictRegistry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.media_io import encode_video


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-json", required=True,
                    help="JSON list of {seed, output_path, prompt} dicts")
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--num-workers", type=int, required=True)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--num-frames", type=int, default=97)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--num-inference-steps", type=int, default=30)
    ap.add_argument("--streaming-prefetch-count", type=int, default=8)
    ap.add_argument("--max-batch-size", type=int, default=4)
    args = ap.parse_args()

    model_root = "/mnt/disk1/models/LTX-2"
    ckpt = f"{model_root}/ltx-2-19b-dev-fp8.safetensors"

    # 构造 pipeline 一次，之后所有 prompt 复用
    registry = StateDictRegistry()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=ckpt,
        distilled_lora=[LoraPathStrengthAndSDOps(
            f"{model_root}/ltx-2-19b-distilled-lora-384.safetensors",
            0.8,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )],
        spatial_upsampler_path=f"{model_root}/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        gemma_root=model_root,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
        registry=registry,                   # ← 关键：注入共享 registry
    )

    with open(args.prompts_json) as f:
        all_jobs = json.load(f)
    my_jobs = [j for i, j in enumerate(all_jobs) if i % args.num_workers == args.worker_id]

    tiling = TilingConfig.default()
    video_guider = MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
                                          modality_scale=3.0, skip_step=0, stg_blocks=[29])
    audio_guider = MultiModalGuiderParams(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                                          modality_scale=3.0, skip_step=0, stg_blocks=[29])

    for job in my_jobs:
        video, audio = pipeline(
            prompt=job["prompt"],
            negative_prompt=job.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            seed=job["seed"],
            height=args.height, width=args.width,
            num_frames=args.num_frames, frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
            images=[],
            tiling_config=tiling,
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=args.max_batch_size,
        )
        encode_video(
            video=video, fps=args.frame_rate, audio=audio,
            output_path=job["output_path"],
            video_chunks_number=get_video_chunks_number(args.num_frames, tiling),
        )
        print(f"[worker {args.worker_id}] done: {job['output_path']}", flush=True)


if __name__ == "__main__":
    main()
