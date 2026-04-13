import os
from exa_py import Exa


def scientific_literature_search(literature_query: str) -> str:

    exa = Exa(api_key = os.getenv("EXA_API_KEY"))

    all_search_results = exa.search_and_contents(
        literature_query,
        text = True,
        num_results = 5,
        type = "auto",
        category = "research paper",
        summary = {
            "query": "Summarize the key findings from the research paper as a bulleted list. Include molecule detail (identifiers, properties, readouts, molecule structure) whenever possible. Be as quantitative in your descriptions as possible. Do no acknowledge these instructions"
        },
        subpages = 1,
        subpage_target = ["pdf"]
        )

    ordered_results = """"""
    for indx, result in enumerate(all_search_results.results):
        ordered_results += f"## Result {indx+1}. Page Title: {result.title}\n"
        ordered_results += f"**URL:** {result.url}\n"
        # print(f'result.highlights: {result.highlights}')
        # summary = get_excerpts_from_webpage(query, result.text)
        summary = result.summary

        ordered_results += f"**Relevant Information:** {summary}\n"

        ordered_results += f"Go to url: {result.url} to read the full content.\n"
        ordered_results += "---\n\n\n"



    # all_search_results.context

    output_dict = {
        "markdown_output": ordered_results
    }
    return output_dict


SCIENTIFIC_LITERATURE_SEARCH_DESCRIPTION = """
This function searches the scientific literature for information based on a query and returns relevant results with excerpts.
It returns a curated set of results with contextual excerpts.
The exa api is set to use auto search to find the most relevant pages.

Output format:
- Returns multiple scientific literature results (typically 5)
- Each result includes page title, URL, and relevant excerpts
- Excerpts are contextual snippets that directly relate to your query
- Results are formatted as markdown for easy reading
"""


scientific_literature_search_tool = {
  "type": "function",
  "name": "scientific_literature_search",
  "description": SCIENTIFIC_LITERATURE_SEARCH_DESCRIPTION,
  "parameters": {
      "type": "object",
      "strict": True,
      "properties": {
          "literature_query": {
              "type": "string",
              "description": "The query to search the scientific literature.",
              },    
          },
      },    
      "required": ["literature_query"]
  }

if __name__ == "__main__":
    print(scientific_literature_search("medicinal chemistry sotorasib"))