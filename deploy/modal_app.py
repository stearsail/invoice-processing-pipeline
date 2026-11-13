import os
import subprocess
import modal

app = modal.App(
    "vllm-serve-qwen",
    image=modal.Image.from_registry(
        "vllm/vllm-openai:v0.26.0",
        add_python="3.11",
    ).entrypoint([]),
)

volume = modal.Volume.from_name("qwen3-vl-ft-weights", create_if_missing=True)
MODEL_DIR = "/models/qwen3-vl-fullds-merged-r8"


@app.function(
    gpu=["L4", "A10G"],
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("vllm-api-key")],
    scaledown_window=3600,
    timeout=3600,
    max_containers=1,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_DIR,
            "--served-model-name",
            "qwen3-vl-fullds-merged-r8",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--limit-mm-per-prompt",
            '{"image":1,"video":0}',
            "--max-model-len",
            "4096",
            "--api-key",
            os.environ["VLLM_API_KEY"],
        ]
    )
