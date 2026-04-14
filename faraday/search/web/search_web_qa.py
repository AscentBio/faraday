# pip install exa-py openai
from exa_py import Exa
from openai import OpenAI
from agents import function_tool

from faraday.openai_clients import llm_client
import json
import os
from datetime import datetime

# print(f'os.getenv("EXA_API_KEY"): {os.getenv("EXA_API_KEY")[:5]}...')
exa = Exa(api_key=os.getenv("EXA_API_KEY"))


def general_web_search_question_answering(query: str) ->  str:

    client = OpenAI(
        base_url = "https://api.exa.ai",
        api_key = os.getenv("EXA_API_KEY"),
    )
    SYSTEM_PROMPT = """
    You are a scientific research assistant specializing in providing evidence-based answers using web search. Your responses must meet rigorous scientific standards.

    SCIENTIFIC RIGOR REQUIREMENTS:
    - Prioritize peer-reviewed publications, government agencies, and established research institutions
    - Include quantitative data, experimental results, and validated methodologies when available
    - Distinguish between established scientific consensus and emerging research
    - Clearly indicate confidence levels and limitations of available evidence
    - Include in-text citations and supporting information if possible.
    - If query if about a molecule, include all molecule identifiers and properties from sources
    """

    messages = [{"role":"system","content":SYSTEM_PROMPT},
             {"role":"user","content":query}]

    completion = client.chat.completions.create(
        model = "exa",
        messages = messages,
    )
    completion_text = completion.choices[0].message.content


    output_dict = {
        "markdown_output": completion_text
    }
    return output_dict

GENERAL_WEB_SEARCH_QUESTION_ANSWERING_DESCRIPTION = """
Question answering tool that searches the web and provides natural language answers to queries.
Uses the Exa API with scientific focus and reputable source citations.

Input arguments:
- query
    - type: string
    - required: true
    - description: The question or query to answer using web search.
    - example: "What are the latest developments in LLM capabilities?"

Recommendations:
- Ask direct, specific questions in natural language.
- Tool provides detailed answers with citations and supporting information.
"""


general_web_search_question_answering_tool = {
  "type": "function",
  "name": "general_web_search_question_answering",
  "description": GENERAL_WEB_SEARCH_QUESTION_ANSWERING_DESCRIPTION,
  "parameters": {
      "type": "object",
      "strict": True,
      "properties": {
          "query": {
              "type": "string",
              "description": "The query to search the web.",
              },    
          },
          
      },    
      "required": ["query"]
  }



if __name__ == "__main__":
    # Test web search
    query = "what is the SMILES of sotorasib"
    result = general_web_search_question_answering(query)
    print(result['markdown_output'])
    
