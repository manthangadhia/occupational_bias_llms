"""
vLLM smoke test — uses Qwen2.5-0.5B-Instruct (~1GB), a small but capable model.
Just verifies that vLLM loads, generates, and returns sensible output.
"""

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading model: {MODEL}")
print("(First run will download ~1GB to HF cache — subsequent runs are instant)\n")

llm = LLM(
    model=MODEL,
    swap_space=0,
)

params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

prompts = [
    "What is the capital of France? Answer in one sentence.",
    "What is 2 + 2? Answer in one sentence.",
]

print("Running inference...\n")
outputs = llm.generate(prompts, params)

for output in outputs:
    prompt = output.prompt
    response = output.outputs[0].text.strip()
    print(f"Prompt:   {prompt}")
    print(f"Response: {response}")
    print()

print("Smoke test complete. vLLM is working.")