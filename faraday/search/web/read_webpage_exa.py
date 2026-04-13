from exa_py import Exa
import os
from typing import List, Any, Dict

from faraday.openai_clients import llm_client

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


def _coerce_to_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _format_exa_results(results: List[Any]) -> str:
    formatted = ""
    for item in results:
        if isinstance(item, dict):
            url = item.get("url", "unknown_url")
            highlights = item.get("highlights")
        else:
            url = getattr(item, "url", "unknown_url")
            highlights = getattr(item, "highlights", None)
        if isinstance(highlights, list):
            highlights_text = "\n".join(str(h) for h in highlights if h)
        else:
            highlights_text = str(highlights) if highlights else "No highlights returned."
        formatted += f"**Web Page Contents for {url}**\n"
        formatted += f"**{highlights_text}**\n\n\n"
    if formatted:
        return formatted
    return "**No content retrieved. Exa returned no result payload.**\n\n\n"


def _format_exa_raw_response(data: Dict[str, Any] | Any, url_list: List[str]) -> str:
    if not isinstance(data, dict):
        formatted = ""
        for url in url_list:
            formatted += f"**Web Page Contents for {url}**\n"
            formatted += (
                "**No content retrieved. Exa returned an unexpected non-dict response.**\n\n\n"
            )
        return formatted

    results = _coerce_to_list(data.get("results", []))
    statuses = _coerce_to_list(data.get("statuses", []))

    results_by_url: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        if url:
            results_by_url[url] = result

    statuses_by_url: Dict[str, Dict[str, Any]] = {}
    for status in statuses:
        if not isinstance(status, dict):
            continue
        status_id = status.get("id")
        if status_id:
            statuses_by_url[status_id] = status

    formatted = ""
    for url in url_list:
        formatted += f"**Web Page Contents for {url}**\n"

        result = results_by_url.get(url)
        if result:
            highlights = result.get("highlights")
            if isinstance(highlights, list):
                highlights_text = "\n".join(h for h in highlights if h)
            else:
                highlights_text = str(highlights) if highlights else "No highlights returned."
            formatted += f"**{highlights_text}**\n\n\n"
            continue

        status = statuses_by_url.get(url)
        if status:
            status_value = status.get("status", "unknown")
            error_field = status.get("error")
            if isinstance(error_field, dict):
                error_tag = error_field.get("tag")
            elif error_field:
                error_tag = str(error_field)
            else:
                error_tag = None
            if error_tag:
                formatted += f"**No content retrieved. Exa status: {status_value} (tag: {error_tag})**\n\n\n"
            else:
                formatted += f"**No content retrieved. Exa status: {status_value}**\n\n\n"
        else:
            formatted += "**No content retrieved. Exa returned no result/status for this URL.**\n\n\n"
    return formatted


def read_webpage(user_query: str, url_list: List[str]) ->  str:
    highlights_payload = {
        "query": f"key excerpts that are relevant to the user's query: {user_query}",
        "max_characters": 15000,
    }

    get_contents_error = None
    try:
        result = exa.get_contents(urls=url_list, highlights=highlights_payload)
        formatted = _format_exa_results(_coerce_to_list(getattr(result, "results", [])))
        # Fall back to the raw API response if parsed results are empty.
        if "No content retrieved" not in formatted:
            return formatted
    except Exception as exc:
        get_contents_error = exc

    try:
        raw_data = exa.request(
            "/contents",
            {
                "urls": url_list,
                "highlights": {
                    "query": highlights_payload["query"],
                    "maxCharacters": highlights_payload["max_characters"],
                },
            },
        )
        return _format_exa_raw_response(raw_data, url_list)
    except Exception:
        if get_contents_error is not None:
            raise get_contents_error
        raise

READ_WEB_PAGE_DESCRIPTION = """
This function reads the content of a webpage and returns the relevant information.

IMPORTANT:
- title: The webpage's full title
- highlights: Key excerpts from the webpage relevant to the search query

This helps the agent:
1. Read the content of a webpage
2. Use information from a webpage to answer the user's query
3. Check generated information against the content of the webpage
"""



read_webpage_tool = {
  "type": "function",
  "name": "read_webpage",
  "description": READ_WEB_PAGE_DESCRIPTION,
  "parameters": {
      "type": "object",
      "strict": True,
      "properties": {
          "user_query": {
              "type": "string",
              "description": "The user's query that may contain @url references.",
          },
          "url_list": {
              "type": "array",
              "description": "The list of URLs to read.",
              "items": {
                  "type": "string"
              }
          },
      },
      "required": ["user_query", "url_list"],
  },
}




if __name__ == "__main__":

    print("Reading webpage...")
    user_query = "What is the molecular structure of sotorasib?"
    url_list = ["https://pubchem.ncbi.nlm.nih.gov/compound/Sotorasib"]
    result = read_webpage(user_query, url_list)
    print(result)

    print("now reading multiple webpages...")
    user_query = "What is the molecular structure of sotorasib?"
    url_list = ["https://pubchem.ncbi.nlm.nih.gov/compound/Sotorasib", "https://www.drugbank.ca/drugs/DB01111"]
    result = read_webpage(user_query, url_list)
    print(result)