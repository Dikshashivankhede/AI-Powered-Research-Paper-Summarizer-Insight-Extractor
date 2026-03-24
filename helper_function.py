import torch 

def summeriser(text, tokenizer, model):
    # tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length= 1024, truncation=True)

    # generate summary

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens = 1000,
            min_length=300,
            length_penalty=0.8,
            num_beams=5,
            early_stopping=True
        )
        
    # decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens = True)
    print("summary: ", summary)
    return summary


# insigth extraction 

from groq import Groq
from dotenv import load_dotenv
import os, json
load_dotenv()

def insigth_extraction(summary):
    
    client = Groq(api_key=os.getenv("groq_api_key"))

    prompt = f"""
    Extract structured insights from the abstract below.

    Return ONLY valid JSON.
    Do not add explanations.
    Do not add markdown.
    Do not add text before or after JSON.

    Use this exact format:

    {{
    "domain": [],
    "research_problem": "",
    "methods": [],
    "datasets": [],
    "metrics": [],
    "key_findings": "",
    "limitations": "",
    "future_directions": ""
    }}

    Abstract:
    {summary}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    #print(response.choices[0].message.content)
    
    return json.loads(response.choices[0].message.content)


