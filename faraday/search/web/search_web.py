

# pip install exa-py openai
from exa_py import Exa
from agents import function_tool

from faraday.openai_clients import llm_client
import json
import os
from datetime import datetime



# print(f'os.getenv("EXA_API_KEY"): {os.getenv("EXA_API_KEY")[:5]}...')
exa = Exa(api_key=os.getenv("EXA_API_KEY"))




def get_excerpts_from_webpage(query: str, webpage_contents: str) ->  str:

    PROMPT = """
    You are a helpful assistant that parse through dense web scraping output

    Your task is to find relevant excerpts that are most relevant to the query.
    In addition to the excerpt, provide a few lines above and below the excerpt to get the context.     
    This will make up the excerpt content.

    Format your response in the following way:
    - excerpt 1
    - <excerpt content with a few lines above and below>
    - reason for why the excerpt is relevant

    - excerpt 2
    - <excerpt content with a few lines above and below>
    - reason for why the excerpt is relevant

    - excerpt 3
    - <excerpt content with a few lines above and below>
    - reason for why the excerpt is relevant

    Format the excerpt content as markdown code block.
    """


    response = llm_client.responses.create(
        model="gpt-4.1",
        instructions=PROMPT,
        input=f'query: {query}\n\nwebpage_contents: {webpage_contents}',
    )
    return response.output_text
    
    


def general_web_search(query: str) ->  str:

    # all_search_results = exa.search_and_contents(
    #     query,
    #     # text = True,
    #     type = "auto",
    #     num_results = 5,
    #     highlights = True,
    #     context = {
    #         "max_characters": 1000000,
    #     },
    #     extras = {
    #         "links": 1
    #     }
    #     )

    QUERY_PROMPT = """
    You are a research assistant specializing in creating comprehensive, information-rich summaries from web content.

    Your task is to create a detailed, information-dense summary that directly addresses the query. Focus on:
    - Authoritative and credible sources with clear source attribution
    - Factual information, data, and verified claims
    - Recent information and current developments
    - Multiple perspectives when relevant
    - Specific details, numbers, dates, and concrete information
    - Key organizations, companies, or individuals involved

    Create an information-rich summary that consolidates all relevant information rather than providing sparse excerpts. Include:
    - Comprehensive synthesis of factual information with specific details
    - Historical context and recent developments
    - Quantitative data, statistics, and measurements when available
    - Multiple viewpoints or approaches when they exist
    - Cross-references between related information and findings
    - Source credibility assessment with clear attribution

    Format your response as a comprehensive, flowing summary that maximizes information density while maintaining accuracy. Include specific source details (authors, organizations, publication dates, URLs when relevant) integrated naturally within the text rather than as separate citations.

    Prioritize information from:
    1. Authoritative sources (established organizations, reputable publications, official sources)
    2. Recent and up-to-date information
    3. Primary sources over secondary interpretations
    4. Multiple corroborating sources when possible

    Aim for maximum information content per word while ensuring accuracy and source transparency.
    """

    all_search_results = exa.search_and_contents(
        query,
        type = "auto",
        num_results = 5,
        highlights = True,
        summary = {
            "query": QUERY_PROMPT
        }
    )

    print(f'number of results: {len(all_search_results.results)}')


    ordered_results = """"""
    for indx, result in enumerate(all_search_results.results):
        ordered_results += f"**Result {indx+1}: {result.title}**\n\n"
        ordered_results += f"**Source:** [{result.url}]({result.url})\n\n"
        # print(f'result.highlights: {result.highlights}')
        # summary = get_excerpts_from_webpage(query, result.text)
        summary = result.summary

        ordered_results += f"**Summary:**\n\n{summary}\n\n"
        ordered_results += f"*[Read full content →]({result.url})*\n\n"
        ordered_results += "---\n\n"



    # all_search_results.context

    output_dict = {
        "markdown_output": ordered_results
    }
    return output_dict

GENERAL_WEB_SEARCH_DESCRIPTION = """
This function searches the web for information based on a query and returns relevant web results with excerpts.
It uses the Exa API to search the web and return a curated set of results with contextual excerpts.
The exa api is set to use auto search to find the most relevant pages.

Output format:
- Returns multiple web results (typically 5)
- Each result includes page title, URL, and relevant excerpts
- Excerpts are contextual snippets that directly relate to your query
- Results are formatted as markdown for easy reading

Safety:
- Results are filtered for safety and relevance.

Input arguments:
- query
    - type: string
    - required: true
    - description: The query string for the earch.
    - example: "Latest developments in LLM capabilities"

Recommendations on how to use this tool:
- Ask your query in a direct manner. 
- Your query can be up to a sentence long but aim to be concise and specific.
- The tool will return relevant excerpts from web pages that match your query.
"""


general_web_search_tool = {
  "type": "function",
  "name": "general_web_search",
  "description": GENERAL_WEB_SEARCH_DESCRIPTION,
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
    query = "bioavailability of sotorasib"
    result = general_web_search(query)
    print(result['markdown_output'])