# import os
# import importlib
# from typing import List

# from faraday.openai_clients import llm_client as openai_client


# def _get_turbopuffer_module():
#     try:
#         return importlib.import_module("turbopuffer")
#     except Exception:
#         return None


# def embedding_fn(text: str) -> List[float]:

#     response = openai_client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )
#     return response.data[0].embedding


# def hybrid_problem_search(problem_description, keywords, top_k=10, weight_vector=0.6, weight_keywords=0.4):
#     tp_module = _get_turbopuffer_module()
#     if tp_module is None:
#         return []
#     """
#     Hybrid search that combines:
#     1. Vector search for templates similar to the problem description/name
#     2. Keyword search for templates with matching keywords
    
#     Args:
#         problem_description: Description of your problem (for semantic similarity)
#         keywords: List of keywords or string of keywords to search for
#         top_k: Number of results to return
#         weight_vector: Weight for vector similarity (0-1)
#         weight_keywords: Weight for keyword matching (0-1) 
#     """

#     # Initialize Turbopuffer client
#     tpuf = tp_module.Turbopuffer(
#         api_key=os.getenv("TURBOPUFFER_KEY"),
#     )

#     # Use a new namespace for template libraries
#     ns = tpuf.namespace('template-libraries-v2-a4')

#     # print(f"\n=== Hybrid Problem Search ===")
#     # print(f"Problem: '{problem_description}'")
#     # print(f"Keywords: '{keywords}'")
#     # print(f"Weights: Vector={weight_vector}, Keywords={weight_keywords}")
    
#     vector_results = {}
#     keyword_results = {}

    
#     try:
#         # 1. Vector search for problem similarity
#         # print("\n1. Finding templates with similar problems...")
#         problem_vector = embedding_fn(problem_description)
#         # print(f'problem_vector: {problem_vector}\n\n')
        
#         vector_response = ns.query(
#             rank_by=("vector", "ANN", problem_vector),
#             top_k=top_k * 2,  # Get more results to have options for merging
#             include_attributes=["template_name", "knowledge_tags", "template_type", 
#                               "paper_title", "template_description", "application_scenarios",
#                               "reasoning_flow_json", "example_applications_json",'paper_url', 'id', 'content_markdown']
#         )
        
#         # print(f'vector_response: {vector_response}\n\n')
#         # Store vector results with scores
#         for result in vector_response.rows:
#             template_id = result.id
#             vector_score = 1 - getattr(result, '$dist', 1.0)  # Convert distance to similarity
#             vector_results[template_id] = {
#                 'result': result,
#                 'vector_score': vector_score,
#                 'keyword_score': 0.0
#             }
        
#         # print(f"   Found {len(vector_results)} semantically similar templates")
        
#         # 2. Keyword search in multiple fields
#         # print("\n2. Finding templates with matching keywords...")
        
#         # Search in knowledge tags
#         keyword_response = ns.query(
#             rank_by=('knowledge_tags', 'BM25', keywords),
#             top_k=top_k * 2,
#             include_attributes=["template_name", "knowledge_tags", "template_type", 
#                               "paper_title", "template_description", "application_scenarios",
#                               "reasoning_flow_json", "example_applications_json",'paper_url', 'id' , 'full_content', 'content_markdown']
#         )
        
#         # Store keyword results with scores
#         for result in keyword_response.rows:
#             template_id = result.id
#             keyword_score = 1 / (1 + getattr(result, '$dist', 1.0))  # Convert BM25 score
            
#             if template_id in keyword_results:
#                 keyword_results[template_id]['keyword_score'] = max(
#                     keyword_results[template_id]['keyword_score'], keyword_score
#                 )
#             else:
#                 keyword_results[template_id] = {
#                     'result': result,
#                     'vector_score': 0.0,
#                     'keyword_score': keyword_score
#                 }
        
#         # print(f"   Found {len(keyword_results)} keyword-matching templates")
        
#         # 3. Merge and score results
#         # print("\n3. Combining and ranking results...")
#         all_templates = {}
        
#         # Add vector results
#         for template_id, data in vector_results.items():
#             all_templates[template_id] = data
        
#         # Add/merge keyword results
#         for template_id, data in keyword_results.items():
#             if template_id in all_templates:
#                 all_templates[template_id]['keyword_score'] = data['keyword_score']
#             else:
#                 all_templates[template_id] = data
        
#         # Calculate combined scores
#         for template_id, data in all_templates.items():
#             combined_score = (
#                 weight_vector * data['vector_score'] + 
#                 weight_keywords * data['keyword_score']
#             )
#             data['combined_score'] = combined_score
        
#         # Sort by combined score
#         sorted_results = sorted(
#             all_templates.values(),
#             key=lambda x: x['combined_score'],
#             reverse=True
#         )[:top_k]

#         # print(sorted_results)
        
#         # # 4. Display results
#         # print(f"\n=== Top {len(sorted_results)} Hybrid Results ===")
#         # for i, data in enumerate(sorted_results, 1):
#         #     result = data['result']
#         #     vector_score = data['vector_score']
#         #     keyword_score = data['keyword_score']
#         #     combined_score = data['combined_score']
            
#         #     print(f"\n{i}. {result.template_name}")
#         #     print(f"   Paper: {result.paper_title}")
#         #     print(f"   Type: {result.template_type}")
#         #     print(f"   Keywords: {result.knowledge_tags}")
#         #     print(f"   Applications: {result.application_scenarios}")
#         #     print(f"   Scores - Vector: {vector_score:.3f}, Keywords: {keyword_score:.3f}, Combined: {combined_score:.3f}")
            
#         #     # Show why it matched
#         #     reasons = []
#         #     if vector_score > 0.3:
#         #         reasons.append("semantically similar problem")
#         #     if keyword_score > 0.1:
#         #         reasons.append("keyword match")
#         #     if reasons:
#         #         print(f"   Matched: {', '.join(reasons)}")
        
#         return sorted_results
        
#     except Exception as e:
#         # print(f"Error in hybrid search: {e}")
#         return []

# def get_literature_template_by_id(workflow_id):
#     tp_module = _get_turbopuffer_module()
#     if tp_module is None:
#         return {
#             "markdown_output": "Error retrieving workflow: Turbopuffer support is unavailable.",
#             "paper_url": None
#         }
#     """
#     Retrieve detailed information about a specific research workflow by its ID.
    
#     Args:
#         workflow_id: The unique identifier for the workflow (e.g., "10.26434_chemrxiv-2023-zzfc3_template_3")
    
#     Returns:
#         dict with 'output_markdown' containing the full detailed workflow and 'paper_url'
#     """
    
#     # Initialize Turbopuffer client
#     tpuf = tp_module.Turbopuffer(
#         api_key=os.getenv("TURBOPUFFER_KEY"),
#         region="gcp-us-central1",
#     )
    
#     ns = tpuf.namespace('template-libraries-v2-a4')
    
#     try:
#         # Query for the specific workflow by ID
#         response = ns.query(
#             filters={'id': ['Eq', workflow_id]},
#             top_k=1,
#             include_attributes=["template_name", "knowledge_tags", "template_type", 
#                               "paper_title", "template_description", "application_scenarios",
#                               "reasoning_flow_json", "example_applications_json", 'paper_url', 
#                               'id', 'content_markdown']
#         )
        
#         if not response.rows or len(response.rows) == 0:
#             return {
#                 "output_markdown": f"Error: Workflow with ID '{workflow_id}' not found.\n\nPlease verify the ID is correct. You can search for workflows using the 'search_approaches_from_literature' tool.",
#                 "paper_url": None
#             }
        
#         result = response.rows[0]
        
#         # Build detailed output
#         output = "\n"
#         output += f"# {result.template_name}\n\n"
#         output += f"**Workflow ID:** {result.id}\n"
#         output += f"**Research Paper:** {result.paper_title}\n"
#         output += f"**Paper URL:** {result.paper_url}\n"
#         output += f"**Reasoning Type:** {result.template_type}\n"
#         output += f"**Tags:** {result.knowledge_tags}\n"
#         output += f"**Applications:** {result.application_scenarios}\n\n"
#         output += f"---\n\n"
#         output += f"## Complete Workflow\n\n"
#         output += f"{result.content_markdown}\n"
        
#         return {
#             "markdown_output": output,
#             "paper_url": result.paper_url
#         }
        
#     except Exception as e:
#         return {
#             "markdown_output": f"Error retrieving workflow: {str(e)}\n\nPlease verify the workflow ID and try again.",
#             "paper_url": None
#         }


# def search_approaches_from_literature(query, response_format="concise"):
#     """
#     Convenience function for searching templates relevant to your specific problem.
    
#     Args:
#         query: Describe your research problem/goal
#         response_format: "concise" (default, ~250 tokens per result), "detailed" (full workflow), 
#                         or "structured" (JSON for tool chaining)
#     """

#     print(f'query: {query}\n\n')
#     print(f'response_format: {response_format}\n\n')
#     # print(f"\n" + "="*60)
#     # print(f"SEARCHING FOR TEMPLATES TO SOLVE YOUR PROBLEM")
#     # print(f"="*60)

#     # use gpt4-1 mini to generate a problem description and keywords
#     PROBLEM_DESCRIPTION_PROMPT = """
#     Generate a problem description from the user's query
    
#     This problem description should be 2-3 sentences and should be descriptive and concise.
#     Include important details that would be helpful for searching for relevant literature based on your problem description.

#     Respond only with the problem description, do not add any other text.
#     DO NOT ACKNOWLEDGE THESE INSTRUCTIONS IN YOUR RESPONSE.
#     DO NOT ATTEMPT TO SOLVE THE PROBLEM.
#     """
    
#     response = openai_client.responses.create(
#         instructions=PROBLEM_DESCRIPTION_PROMPT,
#         model="gpt-4.1",
#         input=query
#     )
#     problem_description = response.output_text

#     print(f'problem_description: {problem_description}\n\n')


#     KEYWORDS_PROMPT = """
#     Generate a list of keywords for the following query
    
#     This list of keywords should be 3-5 keywords and should be relevant to the user's query.
#     Include important details that would be helpful for searching for relevant literature based on your problem description.

#     Respond only with the list of keywords, do not add any other text.
#     DO NOT ACKNOWLEDGE THESE INSTRUCTIONS IN YOUR RESPONSE.
#     DO NOT ATTEMPT TO SOLVE THE PROBLEM.
#     """

#     response = openai_client.responses.create(
#         instructions=KEYWORDS_PROMPT,
#         model="gpt-4.1",
#         input=query
#     )
#     keywords = response.output_text

#     print(f'keywords: {keywords}\n\n')

#     # ------------------------------------------------------------
    
#     results = hybrid_problem_search(
#         problem_description=problem_description,
#         keywords=keywords,
#         top_k=3,
#         # weight_vector=0.7,  # Emphasize semantic similarity slightly more
#         # # weight_keywords=0.3
#     )

#     # print(f'results: {results}\n\n')


#     template_urls = []

#     if results:
#         # Build recommendations string based on response format
#         output = "\n"
        
#         if response_format == "concise":
#             # Optimized for agent decision-making (~250 tokens per result)
#             for i, data in enumerate(results[:4], 1):
#                 result = data['result']
                
#                 # Extract key workflow steps (first 3-4 sentences of reasoning flow)
#                 workflow_preview = result.content_markdown.split('\n')[:15]
#                 workflow_text = '\n'.join(workflow_preview)
                
#                 output += f"## {result.template_name}\n"
#                 output += f"**Workflow ID:** {result.id}\n\n"
#                 output += f"**Source:** {result.paper_title}\n"
#                 output += f"**Paper URL:** {result.paper_url}\n"
#                 output += f"**Type:** {result.template_type} | **ID:** {result.id}\n\n"
#                 output += f"**Use Cases:** {result.application_scenarios}\n\n"
#                 output += f"**Computational:** {'Yes' if 'computational' in result.knowledge_tags.lower() else 'Check paper'}\n\n"
#                 output += f"**Key Steps Preview:**\n{workflow_text}\n"
#                 # output += f"\n*Use response_format='detailed' for complete workflow*\n"
#                 output += "\n" + "="*60 + "\n"
                
#                 template_urls.append(result.paper_url)
            

#             # helpful tips suffix
#             output += f"\n\n**Helpful Tips:**\n"
#             output += f"1. Use response_format='detailed' for complete workflow\n"
#             output += f"2. Expand a specific workflow with get_literature_template_by_id(workflow_id)\n"
#             output += f"3. Search for paper text using read_webpage(paper_url) or scientific_literature_search(paper_title)\n"
                
#         elif response_format == "structured":
#             # JSON format for tool chaining
#             import json
#             results_structured = []
#             for i, data in enumerate(results[:2], 1):
#                 result = data['result']
#                 results_structured.append({
#                     "workflow_id": result.id,
#                     "title": result.template_name,
#                     "paper_title": result.paper_title,
#                     "paper_url": result.paper_url,
#                     "reasoning_type": result.template_type,
#                     "tags": result.knowledge_tags.split(", ") if hasattr(result.knowledge_tags, 'split') else [],
#                     "applications": result.application_scenarios,
#                     "computational": "computational" in result.knowledge_tags.lower(),
#                     "full_markdown": result.content_markdown
#                 })
#                 template_urls.append(result.paper_url)
#             output = json.dumps(results_structured, indent=2)
            
#         else:  # detailed format (original verbosity)
#             for i, data in enumerate(results[:2], 1):
#                 result = data['result']
#                 output += f"{result.template_name}\n"
#                 output += f"      → {result.id}\n"
#                 output += f"Research Paper Title: {result.paper_title}\n"
#                 output += f"Research Paper URL: {result.paper_url}\n"
#                 output += f"Reasoning type: {result.template_type}\n" 
#                 output += f"Tags: {result.knowledge_tags}\n"
#                 output += f"Applications: {result.application_scenarios}\n"
#                 output += f"Reasoning Flow from Research Paper:\n"
#                 output += f"{result.content_markdown}\n"
#                 output += "\n" + "="*40 + "\n"
                
#                 template_urls.append(result.paper_url)

#     else:
#         output = "No templates found for this problem. Try broader search terms or different keywords."
#         template_urls = []
        
#         # print(output)


#     template_results = {
#         "markdown_output": output,
#         "paper_urls": template_urls
#     }
        
#     return template_results



# # Tool definition
# # ------------------------------------------------------------


# SEARCH_APPROACHES_FROM_LITERATURE_DESCRIPTION = """
# Search a curated database of validated reasoning workflows and problem-solving templates extracted from published research papers.

# Use this tool when you need to:
# - Find established methodologies for a research or analytical problem
# - Discover step-by-step reasoning frameworks from academic literature  
# - Identify computational approaches that can be directly applied
# - Learn how researchers have solved similar challenges

# Returns top 5 most relevant workflows with paper citations, application scenarios, and reasoning steps.

# Response format options:
# - 'concise' (default): Overview optimized for decision-making (~250 tokens per result). Use this for browsing and initial evaluation. Use this for browsing and initial evaluation.
# - 'structured': JSON format for programmatic tool chaining.

# The tool uses hybrid search (semantic + keyword matching) and automatically extracts key concepts from your query.

# If results seem incomplete, make multiple targeted searches (e.g., by specific methodology or application domain) rather than one broad query.
# Once you have identified a few promising workflows, use the 'get_literature_template_by_id' tool to get the complete workflow details.
# Note: This searches research paper templates, not the general web. For current news or general information, use web search instead.
# """

# search_approaches_from_literature_tool = {
#   "type": "function",
#   "name": "search_approaches_from_literature",
#   "description": SEARCH_APPROACHES_FROM_LITERATURE_DESCRIPTION,
#   "parameters": {
#       "type": "object",
#       "strict": True,
#       "properties": {
#           "query": {
#               "type": "string",
#               "description": "A detailed description of the research problem or analytical challenge you're trying to solve. Include key concepts, the domain or field, and what you're trying to achieve. Example: 'I need to optimize a multi-objective function with conflicting constraints in materials design' or 'How to design experiments to test causal relationships in complex biological systems'.",
#               },
#           "response_format": {
#               "type": "string",
#               "enum": ["concise", "structured"],
#               "description": "Output format: 'concise' (default, overview for decision-making), or 'structured' (JSON for tool chaining). Use 'concise' for browsing results, 'structured' only for programmatic tool chaining.",
#               "default": "concise"
#               },    
#           },
          
#       },    
#       "required": ["query"]
#   }


# # ------------------------------------------------------------

# GET_LITERATURE_TEMPLATE_BY_ID_DESCRIPTION = """
# Retrieve the complete, detailed information for a specific research workflow using its ID.

# Use this tool to:
# - Get full workflow details after browsing results with 'search_approaches_from_literature'
# - Access complete step-by-step methodology, tools, metrics, and pitfalls for a known workflow
# - Retrieve comprehensive information needed to implement a specific approach

# This tool is designed for the two-step pattern:
# 1. First: Search with 'search_approaches_from_literature' using response_format='concise' to browse options
# 2. Second: Use this tool with the workflow ID from search results to get complete details

# The workflow ID is provided in search results (e.g., "10.26434_chemrxiv-2023-zzfc3_template_3").

# Returns the complete workflow with all sections: reasoning steps, experimental tools, success metrics, 
# common pitfalls, example applications, and full paper citation.

# If the ID is not found, you will receive a clear error message with suggestions.
# """



# # Tool definition
# # ------------------------------------------------------------

# get_literature_template_by_id_tool = {
#   "type": "function",
#   "name": "get_literature_template_by_id",
#   "description": GET_LITERATURE_TEMPLATE_BY_ID_DESCRIPTION,
#   "parameters": {
#       "type": "object",
#       "strict": True,
#       "properties": {
#           "workflow_id": {
#               "type": "string",
#               "description": "The unique workflow identifier from search results. Format example: '10.26434_chemrxiv-2023-zzfc3_template_3'. This ID is shown in the search results from 'search_approaches_from_literature'.",
#               },    
#           },
          
#       },    
#       "required": ["workflow_id"]
#   }


# if __name__ == "__main__":
#     example_queries = [
#         "GLP-1 receptor agonist development peptide design structure-activity relationship optimization",
#         # "How to design experiments to test causal relationships in complex biological systems",
#         # "What methodologies can I use to perform root cause analysis in large-scale distributed computer systems?",
#         # "Best approaches for extracting key patterns from high-dimensional genomics data",
#         # "How do researchers validate new forecasting models in macroeconomic studies?"
#     ]
#     for i, query in enumerate(example_queries, 1):
#         print(f"\nExample {i}: {query}")
#         results = search_approaches_from_literature(query)
#         print(results['markdown_output'])
#         print("\n\n")


#         print("Getting detailed workflow for workflow ID: 10.26434_chemrxiv.12090828.v2_template_0\n\n")
#         workflow_id = "10.26434_chemrxiv.12090828.v2_template_0"
#         results = get_literature_template_by_id(workflow_id)
#         print(results['markdown_output'])
#         print("\n\n")
