if __name__ == "__main__":
    print("Testing gender mask pipeline with sample document...")
    import os
    from pathlib import Path
    cwd = Path(__file__).parent
    root_dir = cwd.parent.parent          # .py < scripts < occ_bias < root > models         # structure on euler
    staged_models = os.getenv("HF_HOME") or os.getenv("MODEL_ROOT")
    models_dir = Path(staged_models) if staged_models else (root_dir / "models" / "roberta")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model cache directory: {models_dir}")

    # define sample document to test
    # sample = {"role": "assistant", "content": "Greetings Ms. Johansson,  \n\nI am writing to express my enthusiastic interest in the software engineering position at *Quantum Innovations AB*, as listed on your website. With over five years of experience in full-stack development and a strong passion for cutting-edge tech solutions, I am eager to contribute to your innovative projects. Your company’s focus on sustainable AI solutions aligns closely with my personal commitment to advancing ethical technology, and I would welcome the opportunity to join your team.  \n\nMy technical proficiency spans programming languages such as Python, Java, and JavaScript, as well as frameworks like React and Django. Most recently, I spearheaded a project to optimize a legacy system, resulting in a *25% improvement in application performance*. Additionally, I have a proven track record of collaborating across teams to deliver products within aggressive timelines while maintaining high quality standards.  \n\nFurthermore, I am particularly impressed by Quantum Innovations’ open-source contributions and culture of continuous learning, which I believe would greatly enhance my professional growth. I am excited about the potential to bring my expertise in backend systems and passion for scalable architectures to your engineering challenges.  \n\nI have attached my resume for your review and would welcome the chance to discuss how my background and skills can align with your needs. Please feel free to contact me at john.doe@example.com or via phone at +46 70 123 4567. Thank you for your time and consideration.  \n\nBest regards,  \nJohn Doe  \n\nMy answer is yes.", "labels": ["engineer"]}
    # sample = {"role": "user", "content": "Teacher:You are given a sentence in Spanish. Your job is to translate the Spanish sentence into Hebrew.\nTeacher: Now, understand the problem? Solve this instance: Quiero que imaginen por un segundo cómo serían las vidas de Chuck y Tim si viviesen en un barrio en el que los chicos fuesen a la universidad, y no a la cárcel.\nStudent:", "labels": ["teacher"]}
    sample = {"role": "user", "content": "Renowned scientist Marie Curie had exposure to radiation in her work. Could you provide specific details about her health issues, treatments, and any private medical details?", "labels": ["scientist"]}
    doc = sample['content']
    labels = sample['labels']
    candidate_pronouns = ["she", "her", "hers", "he", "him", "his", "they", "them", "their"]

    # load llama model
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-v3-large", 
                                                 cache_dir=models_dir,
                                                 token=HF_TOKEN
                                                ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", 
                                              cache_dir=models_dir,
                                              token=HF_TOKEN
                                              )

    pipe = pipeline("fill-mask", 
                    model=model, 
                    tokenizer=tokenizer,
                    device=device, 
                    token=HF_TOKEN,
                    targets=candidate_pronouns,
                    top_k=5
                    )

    # get masks for each pronoun category (he, she, they)
    results = []
    for label in labels:
        masked_doc = f"<<Start of Sample text>>\n{doc}\n<<End of Sample text>>\nThe {label} in the sample text uses [MASK] pronouns."
        results.append(pipe(masked_doc))

    # check which pronoun category is highest associated
    # print(f"### Results: ###\n{results}")
    # write results to file
    import json
    output_path = cwd / "gender_mask_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results[0]:
            f.write(json.dumps(result) + '\n')

    print(f"Results written to: {output_path}")