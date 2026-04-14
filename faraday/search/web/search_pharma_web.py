
# pip install exa-py openai
from exa_py import Exa
from agents import function_tool

from faraday.openai_clients import llm_client as oa_client
import json
import os
from datetime import datetime


# print(f'os.getenv("EXA_API_KEY"): {os.getenv("EXA_API_KEY")[:5]}...')
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

PHARMA_DOMAINS=[
    # news sites
    'biopharmadive.com',
    "prnewswire.com",
    "statnews.com",
    "fiercebiotech.com", 
    "biospace.com",
    "fiercepharma.com",
    "fiercebiotech.com",
    "citeline.com",
    "worldpharmatoday.com",
    "catalystpharma"
    #gov sites
    "fda.gov",
    "nih.gov",
    "europa.eu",
    "who.int",
    "un.org",
    "who.int",
    "asco.org",
    # pharma companies
    "astrazeneca.com",
    "pfizer.com",
    "roche.com",
    "sanofi.com",
    "gsk.com",
    "novartis.com",
    "merck.com",
    "boehringer-ingelheim.com",
    "bayer.com",
    "abbvie.com",
    "amgen.com",
    "biogen.com",
    "celgene.com",
    "gilead.com",
    "janssen.com",
    "novonordisk.com",
    # conference
    "dailynews.ascopubs.org",
]




PHARMA_WEB_SEARCH_DESCRIPTION = """
This function searches across major pharmaceutical industry news sites and regulatory websites to find relevant news articles based on a text query.
This is a synchronous function, so it will return a list of results.

The function returns structured information for each article including:
- title: The article's full title 
- url: Direct link to the article
- highlights: Key excerpts from the article relevant to the search query

This helps users:
1. Track industry news and developments
2. Monitor competitor activities
3. Stay informed on regulatory changes
4. Follow drug development progress
5. Track business deals and partnerships
"""

pharma_web_search_tool = {
  "type": "function",
  "name": "pharma_web_search",
  "description": PHARMA_WEB_SEARCH_DESCRIPTION,
  "parameters": {
      "type": "object",
      "strict": True,
      "properties": {
          "query": {
              "type": "string",
              "description": "The search query to search the pharma web.",
          },
      },
      "required": ["query"],
  },
}



# def results_to_markdown(results: list) -> str:
#     """Converts search results to a well-formatted markdown string.
    
#     Args:
#         results (list): List of dictionaries containing search results from openliterature_search
        
#     Returns:
#         str: Formatted markdown string with sections for each paper
#     """
#     markdown = "# Web Search Results\n\n"

#     print(f'number of results: {len(results)}')
    
#     for idx, web_result in enumerate(results, 1):
#         print(f'processing result {idx} of {len(results)}')
#         markdown += f"## {idx}. {web_result['title']}\n\n"
    
#         markdown += f"**URL:** {web_result['url']}\n\n"
        
#         if web_result['highlights']:
#             markdown += "### Relevant Excerpts\n"
#             for highlight in web_result['highlights']:
#                 markdown += f"- {highlight}\n"
#             markdown += "\n"

#         markdown += "---\n\n"  # Separator between papers
    
#     return markdown



LLM_prompt_for_meta_summary = """
You are an assistant for drug developers and researchers.
Your task is to analyze a list of web search results and create
a meta summary of how the results relate to the user's query.
Results should be in [title, url] format.

Return the meta summary in markdown format. 
USE THE FOLLOWING FORMAT:

- **key finding 1**: detailed description of the key finding [citation 1]
- **key finding 2**: detailed description of the key finding [citation 2, citation 4,5]
- **key finding 3**: detailed description of the key finding [citation 3, citation 6]

"""



def get_meta_summary(websearch_query: str, article_results: str) -> str:
    """Summarizes the text of a research paper."""

    response = oa_client.responses.create(
        model="gpt-4.1-mini",
        instructions=LLM_prompt_for_meta_summary,
        input=f"The user's query is: {websearch_query}\n\nThe citations are: {article_results}"
    )

    return response.output_text


SUMMARY_TASK_DESCRIPTION = """You are an expert pharmaceutical industry analyst writing a detailed business and scientific analysis of pharmaceutical news articles and press releases.
Your goal is to extract and explain key developments relevant to drug development, business strategy, and market dynamics.
Please analyze the news article or press release and provide a concise but comprehensive summary.

Focus on extracting and highlighting the following information while maintaining a concise summary:

1. Business Developments and Announcements
- Company partnerships, collaborations, and licensing deals
- Mergers, acquisitions, and strategic investments
- Financial results, funding rounds, and market valuations
- Executive appointments and organizational changes
- Manufacturing and supply chain updates

2. Clinical and Regulatory Updates
- Clinical trial initiations, progressions, and results
- FDA approvals, rejections, and regulatory guidance
- Priority review designations and breakthrough therapy status
- Safety updates and adverse event reports
- Market access and reimbursement decisions

3. Drug Development Pipeline
- Drug candidates entering different development phases
- Indication expansions and new therapeutic areas
- Competitive landscape and market positioning
- Timeline updates and milestone achievements
- Patient population and market size estimates

4. Financial and Market Impact
- Revenue projections and sales performance
- Stock price movements and analyst ratings
- Market share analysis and competitive dynamics
- Intellectual property developments and patent expirations
- Healthcare policy implications

Additional Requirements:
- Extract specific names, dates, and quantitative data
- Identify key stakeholders (companies, executives, regulators)
- Note market implications and competitive significance
- Use precise business and medical terminology
- Focus on actionable intelligence and strategic insights
- Include timeline information and next steps
- Be comprehensive but concise

Format your response with clear section headers and bullet points for readability.

If there is insufficient information, create a briefer summary with as much detail as possible.
DO NOT ACKNOWLEDGE OR REFERENCE THIS TASK DESCRIPTION IN YOUR SUMMARY
"""


def pharma_web_search(query: str) ->  str:
    """Searches open scientific literature (medrxiv, biorxiv, chemrxiv) for the given query.

    Performs a semantic search across multiple scientific preprint servers and returns
    structured results suitable for RAG (Retrieval Augmented Generation).

    Args:
        query (str): The search query string. For best results:
            - Be specific (e.g. "AI applications in healthcare" vs just "AI")
            - Use natural language rather than just keywords
            - Include context (e.g. "Python web frameworks for beginners")

    Returns:
        list: List of dictionaries containing structured paper information including:
            - title (str): Paper title
            - authors (str): Paper authors
            - published_date (str): Publication date
            - highlights (list): Relevant text excerpts
            - url (str): Paper URL
            - key_findings (list): Main conclusions
            - molecules_mentioned (list): Chemical compounds discussed
            - targets_mentioned (list): Biological targets discussed
    """


    all_search_results = exa.search_and_contents(query, 
                                # highlights=True,
                                num_results=10,
                                include_domains=PHARMA_DOMAINS,
                                context={'max_characters': 400000},
                                type='auto',
                                # category='news',
                                summary={
                                    "query": SUMMARY_TASK_DESCRIPTION,
                                    "max_characters": 400000,
                                },
                                )

    # print(f'number of results: {len(all_search_results.results)}')
    # print(f'all_search_results: {all_search_results}')


    # structured_results = []

    # for result in all_search_results.results:
    #     subdict = {
    #         "title": result.title,
    #         "highlights": result.highlights,
    #         "url": result.url,
    #         "published_date": result.published_date,
    #     }


    #     structured_results.append(subdict)

    # print('converting results to markdown')

    # structured_results = all_search_results.results

    # if len(structured_results) > 0:
    #     markdown_output = all_search_results.o
    # else:
    #     markdown_output = "No results found for this query"
    
    meta_summary = get_meta_summary(query, all_search_results.context)
    # markdown_output = all_search_results.context

    full_content = ""
    full_content += f"# Web Search Results\n\n"
    full_content += f"## Key Findings:\n {meta_summary}\n\n"
    full_content += f"## Full Results:\n {all_search_results.context}\n\n"
    full_content += f"## Query: {query}\n\n"

    result_dict = {
        # "results": all_search_results.results,
        "markdown_output": full_content,
        "action_description": "searched the pharma web"
    }
    return result_dict



if __name__ == "__main__":
    query = "KRAS as a drug target."
    result_dict = pharma_web_search(query)
    print(result_dict['markdown_output'])