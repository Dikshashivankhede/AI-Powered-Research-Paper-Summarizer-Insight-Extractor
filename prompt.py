def build_prompt(content, query):
    return f"""
    You are a research assistant.

    Use ONLY the provided research paper content to answer the question.

    Rules:
    1. If the answer exists in the provided papers, generate the answer.
    2. Mention ONLY the title of the research paper used for the answer.
    3. If the answer is not present in the papers, respond exactly:

    Answer: Not found in the retrieved papers.
    Research Paper: None

    Response format:

    Answer:
    <answer>

    Research Paper:
    <paper title>, <paper title>

    Content:
    {content}

    Question:
    {query}
    """