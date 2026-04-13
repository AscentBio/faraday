# """
# Search files by semantic similarity to a query.
# """

# from typing import List, Dict
# from faraday.memory.in_memory_rag import (
#     is_in_memory_rag_enabled,
#     search_files as search_files_in_memory,
# )


# def search_filebase_fn(
#     user_id: str,
#     chat_id: str,
#     search_query: str,
#     top_k: int = 5
# ) -> Dict[str, str]:
#     """
#     Search files by lexical/semantic overlap in in-memory file RAG.
    
#     Args:
#         user_id: User identifier
#         chat_id: Chat identifier (optional filter)
#         search_query: Text query to search for
#         top_k: Maximum number of files to return
        
#     Returns:
#         Dict with markdown_output key containing formatted markdown
#     """
#     if not is_in_memory_rag_enabled():
#         return {
#             "markdown_output": (
#                 "# File Search Results\n\n"
#                 f"**Query:** {search_query}\n\n"
#                 "*In-memory file search is currently disabled.*"
#             )
#         }
#     files = search_files_in_memory(
#         user_id=user_id,
#         chat_id=chat_id,
#         search_query=search_query,
#         top_k=top_k,
#     )
#     return {"markdown_output": _format_file_results_markdown(search_query, files)}


# def _format_file_results_markdown(search_query: str, files: List[Dict]) -> str:
#     """
#     Format file search results as markdown.
    
#     Args:
#         search_query: The original search query
#         files: List of file dictionaries with metadata
        
#     Returns:
#         Formatted markdown string
#     """
#     if not files:
#         return f"# File Search Results\n\n**Query:** {search_query}\n\n*No matching files found.*"
    
#     md = f"# File Search Results\n\n"
#     md += f"**Query:** {search_query}\n"
#     md += f"**Results:** {len(files)} file(s)\n\n"
#     md += "---\n\n"
    
#     for idx, file in enumerate(files, 1):
#         md += f"## {idx}. {file['filename']}\n\n"
        
#         # Filepath
#         if file['filepath']:
#             md += f"**Path:** `{file['filepath']}`\n\n"
        
#         # File type
#         if file['file_type']:
#             md += f"**Type:** {file['file_type']}\n\n"
        
#         # Description
#         if file['description']:
#             md += f"**Description:** {file['description']}\n\n"
        
#         # Parent directory
#         if file['parent_directory']:
#             md += f"**Directory:** `{file['parent_directory']}`\n\n"
        
#         # Relevance score
#         if file['score'] is not None:
#             md += f"**Relevance Score:** {file['score']:.4f}\n\n"
        
#         # Created at
#         if file['created_at']:
#             md += f"**Created:** {file['created_at']}\n\n"
        
#         md += "---\n\n"
    
#     return md


# SEARCH_FILEBASE_DESCRIPTION = """
# This function searches the filebase for files by semantic similarity to a query and returns relevant results with excerpts.
# It returns a curated set of results with contextual excerpts.

# Use this tool to search across all files in the filebase for relevant files to the user's query.
# You will not get file contents from this tool but you will get a high level description.
# You will also get filepaths and some helpful metadata about the files

# Output format:
# - Returns multiple file results (typically 5)
# - Each result includes file name, path, file type, description, and relevance score
# - Results are formatted as markdown for easy reading
# """




# search_filebase_tool = {
#     "type": "function",
#     "name": "search_filebase",
#     "description": SEARCH_FILEBASE_DESCRIPTION,
#     "parameters": {
#         "type": "object",
#         "strict": True,
#         "properties": {
#             "search_query": {
#                 "type": "string", 
#                 "description": "The query to search the filebase for relevant files.",
#             },
#         },
#         "required": ["search_query"],
#     },
# }