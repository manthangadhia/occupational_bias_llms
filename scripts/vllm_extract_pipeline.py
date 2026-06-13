from pathlib import Path
import gc
import re

# -------------------------
# Configuration
root_dir = Path(__file__).parent.parent.parent                  # .py < scripts < occ_bias < root > models         # structure on euler
project_dir = root_dir / "occ_bias"
results_dir = project_dir / "data" / "olmo7b_results"
frog_results = results_dir / "olmo7b_results_frog.jsonl"
generic_results = results_dir / "olmo7b_results_generic.jsonl"

import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

# Target schema
class NarrativeExtractionResult(BaseModel):
    name: Optional[str] = Field(None, description="The name mentioned in the narrative, if any.")
    gender: Optional[str] = Field(None, description="The gender mentioned in the narrative, if any.")
    occupation: Optional[str] = Field(None, description="The occupation mentioned in the narrative, if any.")
    attended_university: Optional[bool] = Field(None, description="Whether the narrative mentions attending university.")

def create_prompt(narrative: str, system_prompt: str, tokenizer) -> str:
    instruction = """
    In an experiment I set up, two groups either wrote about a fictitious person they chose, or about an anthropomorphic frog in a children's story.
    You will receive a sample of these texts in markdown, and I want you to output a json object containing the name, gender (male/female/other), occupation of the character (try to be succinct and generate a common real-world occupation, something like: "performer", "teacher", "doctor", and so on), and whether or not they attended university (yes/no/null). Default any of those values to `null` if the information is missing.
    SAMPLE:\n
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}{narrative}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    # Load the narratives from frog and generic into one df

    # Define system prompt
    system_prompt = """# HOW YOU SHOULD RESPOND
    You are a helpful and precise assistant for extracting information from narratives. Please follow the instructions below to provide the information in a structured JSON format.
    You should reason step-by-step about your answers, and adhere to the output labels provided in the instructions.
    """ 

    # 2. Load your data
    df_frog = pd.read_json(frog_results, lines=True)  # Adjust if lines=True is needed
    df_generic = pd.read_json(generic_results, lines=True)  # Adjust if lines=True is needed

    # combine the two dataframes
    df = pd.concat([df_frog, df_generic], ignore_index=True)
    del df_frog, df_generic  # free memory
    gc.collect()
    print(f"Loaded {len(df)} narratives for extraction.")

    # 4. Initialize vLLM with Guided Decoding
    MODEL = "mistralai/Ministral-3-14B-Instruct-2512"  # Upgraded for better JSON adherence
    thinking = True if "Reasoning" in MODEL else False  # Use the reasoning version if available for better structured output
    max_len = 8192 if thinking else 4096  # Longer context for reasoning models
    max_sampling_tokens = 1024 if thinking else 256
    llm = LLM(
        model=MODEL, 
        swap_space=0, 
        max_model_len=max_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"Initialized vLLM  with model {MODEL} (thinking={thinking}) and tokenizer.")
    
    prompts = [create_prompt(text, system_prompt, tokenizer) for text in df["response"]]
    # Enforce JSON output matching the Pydantic schema
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_sampling_tokens,  # 300 token input + ~50 token JSON output is safe
        guided_decoding=GuidedDecodingParams(json=NarrativeExtractionResult.model_json_schema())
    )

    # 5. Run Inference (vLLM handles optimal batching under the hood)
    print(f"Processing {len(prompts)} bios...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Parse results back into the DataFrame
    extracted_data = []
    for output in outputs:
        raw = output.outputs[0].text
        if thinking:
            # Thinking models typically wrap thoughts in <tool_call>...<tool_call>
            think_match = re.search(r'<tool_call>(.*?)</tool_call>', raw, re.DOTALL)
            thinking_output = think_match.group(1).strip() if think_match else None
        
        # The JSON comes after the thinking block
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)

        try:
            # vLLM guarantees the output matches the schema string structure
            json_data = json.loads(json_match.group())
            json_data['thinking'] = thinking_output if thinking else None  # Optionally keep the thinking process for analysis
            extracted_data.append(json_data)
        except Exception as e:
            # Fallback in case of unexpected generation issues
            extracted_data.append({"name": None, "gender": None, "occupation": None, "attended_university": None, "thinking": thinking_output if thinking else None})

    # Merge extracted fields back to original dataframe
    df_ext = pd.DataFrame(extracted_data)
    df = pd.concat([df, df_ext], axis=1)

    # Save results
    output_path = results_dir / "baseline_results.json"
    df.to_json(output_path, orient="records", indent=2)
    print(f"Extraction complete. Saved results to {output_path}")

if __name__ == "__main__":
    main()