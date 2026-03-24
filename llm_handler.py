from google import genai
from google.genai import types
from groq import Groq
import os
from dotenv import load_dotenv
from prompt import build_prompt

load_dotenv()

# Initialize clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(content, query):

    prompt = build_prompt(content, query)

    # -------- GEMINI FIRST --------
    
    try:
        print("Using Gemini...")

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3
            ),
        )

        text = response.text

        # fallback if empty/bad response
        if not text or "Error" in text:
            raise Exception("Invalid Gemini response")

        return text

    except Exception as e:
        print("Gemini failed:", e)

    # -------- GROQ FALLBACK --------
    try:
        print("Switching to Groq...")

        response = groq_client.chat.completions.create(
          model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("Groq also failed:", e)

        return "Error: Both Gemini and Groq failed."