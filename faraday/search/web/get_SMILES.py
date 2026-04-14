

# pip install exa-py openai
from exa_py import Exa
import os

from faraday.openai_clients import llm_client
from datetime import datetime



# print(f'os.getenv("EXA_API_KEY"): {os.getenv("EXA_API_KEY")[:5]}...')
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

def get_excerpts_from_webpage(query: str, webpage_contents: str) ->  str:

    PROMPT = """
    You are an expert chemical informatics specialist with deep knowledge of SMILES notation, chemical databases, and molecular structure representation. Your task is to extract and analyze chemical information from web content with high precision and reliability.

    ## PRIMARY OBJECTIVE
    Extract SMILES strings and comprehensive chemical information for the requested molecule, prioritizing accuracy and completeness.

    ## SEARCH PRIORITIES (in order of importance)
    1. **SMILES Notation Strings**: Look for canonical, isomeric, or any valid SMILES representation
       - Pattern recognition: C, N, O, S, P atoms with bonds (-, =, #), branches (), rings (numbers), stereochemistry (@, @@)
       - Examples: "CC(=O)OC1=CC=CC=C1C(=O)O" (aspirin), "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" (caffeine)
    
    2. **Chemical Identifiers**: 
       - CAS Registry Numbers (format: XXXXXX-XX-X)
       - InChI strings (starts with "InChI=")
       - InChI Keys (27-character strings)
       - PubChem CID numbers
       - ChemSpider ID numbers
    
    3. **Molecular Data**:
       - Molecular formulas (e.g., C9H8O4)
       - Molecular weights/masses
       - Chemical names (IUPAC, common names, synonyms)
    
    4. **Structural Information**:
       - Chemical properties (melting point, boiling point, solubility)
       - Pharmacological data (if applicable)
       - Safety information

    ## VALIDATION CRITERIA
    Before including any SMILES string, verify:
    - Contains valid chemical symbols (C, N, O, S, P, F, Cl, Br, I, etc.)
    - Uses proper SMILES syntax (parentheses for branches, numbers for rings)
    - Reasonable length for the target molecule
    - Appears in chemical context (not random text)

    ## OUTPUT FORMAT
    For each relevant finding, structure your response as follows:

    **EXCERPT [N]**: [Brief descriptor of what was found]
    ```
    [Include 2-3 lines of context above]
    [THE ACTUAL CHEMICAL DATA LINE - highlight SMILES if present]
    [Include 2-3 lines of context below]
    ```
    **ANALYSIS**: [Detailed explanation of relevance]
    - **Chemical Data Found**: [List specific identifiers/SMILES found]
    - **Confidence Level**: [High/Medium/Low based on source reliability]
    - **Data Type**: [SMILES/CAS/InChI/Formula/etc.]
    - **Notes**: [Any validation concerns or additional context]

    ## QUALITY STANDARDS
    - Prioritize excerpts containing SMILES strings over other chemical data
    - Include context that helps validate the chemical information
    - Flag any suspicious or potentially incorrect data
    - If multiple SMILES are found, include the most complete/canonical version
    - Always explain WHY each excerpt is chemically relevant

    ## ERROR HANDLING
    - If no SMILES found, focus on other chemical identifiers
    - If data seems inconsistent, note the discrepancy
    - If source appears unreliable, mention this in your analysis
    """


    response = llm_client.responses.create(
        model="gpt-4.1-mini",
        instructions=PROMPT,
        input=f'query: {query}\n\nwebpage_contents: {webpage_contents}',
    )
    return response.output_text
    

def get_filtered_results_from_webpage_results(query: str, webpage_contents: str) ->  str:

    PROMPT = """
    You are an expert chemical informatics specialist with deep knowledge of SMILES notation, chemical databases, and molecular structure representation. Your task is to analyze web search results and filter out non-useful content for chemical information extraction.

    ## PRIMARY OBJECTIVE
    Evaluate the provided webpage contents and determine if they contain useful chemical information relevant to the query. Filter out irrelevant, low-quality, or non-chemical content.

    ## FILTERING CRITERIA
    **INCLUDE results that contain:**
    - SMILES notation strings or other chemical identifiers
    - Chemical database entries (PubChem, ChemSpider, DrugBank, etc.)
    - Peer-reviewed scientific literature with chemical data
    - Authoritative chemical reference materials
    - Molecular formulas, structures, or properties
    - Pharmaceutical or chemical compound information

    **EXCLUDE results that are:**
    - General news articles without specific chemical data
    - Commercial product listings without chemical identifiers
    - Blog posts or forums without authoritative chemical information
    - Irrelevant content that doesn't match the chemical query
    - Duplicate or redundant information already covered
    - Low-quality sources with questionable chemical accuracy
    - Content that mentions the molecule name but provides no useful chemical data

    ## OUTPUT FORMAT
    Return one of the following:

    **If content is USEFUL:**
    ```
    URL: [URL of the content]
    TITLE: [Title of the content]
    CHEMICAL SMILES: [SMILES string of the molecule]
    ADDITIONAL CHEMICAL DATA: [List key chemical identifiers/data found]
    SOURCE_RELIABILITY: [High/Medium/Low]
    ANY OTHER USEFUL INFORMATION: [2-3 sentence summary of the most relevant chemical information such as molecule identifier, chemical data, etc.]
    ```

    **If content is NOT USEFUL:**
    - do not include it in your output.

    ## QUALITY ASSESSMENT
    - Prioritize authoritative chemical databases and peer-reviewed sources
    - Consider the specificity and accuracy of chemical information provided
    - Evaluate if the content directly addresses the chemical query
    - Check for presence of validated chemical identifiers (SMILES, CAS, InChI, etc.)
    """

    response = llm_client.responses.create(
        model="gpt-4.1-mini",
        instructions=PROMPT,
        input=f'query: {query}\n\nwebpage_contents: {webpage_contents}',
    )
    return response.output_text
    


def name_to_smiles(query: str) ->  str:


    # First, try to get results from established chemical databases
    database_search_results = exa.search_and_contents(
        query,
        type = "auto",
        num_results = 5,
        # highlights = True,
        include_domains = [
            "pubchem.ncbi.nlm.nih.gov",
            "chemspider.com", 
            "drugbank.ca",
            "ebi.ac.uk",
            "rcsb.org",
            "comptox.epa.gov",
            "nist.gov",
            "wikipedia.org"
        ],
        summary = {
            "query": f"""Retrieve the SMILES string for the molecule with all supporting molecule data.
             Include all important information about the molecule, it form, and any properties.
             Clarify if the SMILES string is canonical, isomeric, or any other type.
             Return these as well formatted list and be scientific."""
        }
    )
    

    all_search_results = database_search_results

    print(f'number of results: {len(all_search_results.results)}')

    
    
    ordered_results = """"""
    for indx, result in enumerate(all_search_results.results):
        # Check if result is from an authoritative database        
        ordered_results += f"## Result {indx+1}. Page Title: {result.title}\n"
        ordered_results += f"**URL:** {result.url}\n"
        
        # print(f'result.highlights: {result.highlights}')
        # summary = get_excerpts_from_webpage(query, result.text)
        summary = result.summary

        ordered_results += f"**Relevant Excerpts:** {summary}\n"
        ordered_results += f"Go to url: {result.url} to read the full content.\n"
        ordered_results += "---\n\n\n"

    filtered_results = get_filtered_results_from_webpage_results(query, ordered_results)

    output_dict = {
        "markdown_output": filtered_results
    }
    return output_dict


NAME_TO_SMILES_DESCRIPTION = """
Advanced chemical information extraction tool powered by GPT-5, designed to find SMILES strings and comprehensive molecular data with expert-level accuracy and validation.

## Core Capabilities
- **SMILES Extraction**: Identifies, validates, and extracts canonical/isomeric SMILES notation
- **Chemical Validation**: Expert-level syntax checking and consistency verification
- **Multi-Database Search**: Prioritizes authoritative sources (PubChem, ChemSpider, DrugBank, EBI, RCSB, EPA CompTox, NIST)
- **Intelligent Analysis**: Provides confidence assessments and source reliability evaluations

## Search Strategy
1. **Primary Search**: Established chemical databases with authority verification
2. **Supplementary Search**: General web sources when database results are insufficient  
3. **Quality Control**: GPT-5 powered validation of chemical syntax and data consistency
4. **Source Assessment**: Reliability scoring based on database authority and peer-review status

## Output Features
- Structured excerpts with contextual information
- Confidence levels for each chemical identifier found
- Source reliability assessments
- Validation notes for SMILES syntax and consistency
- Clear marking of authoritative database sources

## Input Arguments
- query (string, required): Molecule name or chemical compound identifier
  - Examples: "aspirin", "SMILES of sotorasib", "caffeine molecular structure"
  - Tip: Use specific chemical names for optimal results

## Best Practices
- Provide precise molecule names when possible
- Tool automatically validates SMILES syntax and flags inconsistencies  
- Prioritizes peer-reviewed database content over general web sources
- Results include expert-level chemical analysis and validation
"""

name_to_smiles_tool = {
    "type": "function",
    "name": "name_to_smiles",
    "description": NAME_TO_SMILES_DESCRIPTION,
    "parameters": {
        "type": "object",
        "strict": True,
        "properties": {
            "query": {
                "type": "string", 
                "description": "The molecule name or chemical compound to search for SMILES string and chemical data.",
            },
        },
        "required": ["query"],
    },
}

if __name__ == "__main__":
    # Test some example molecules
    test_molecules = [
        "aspirin",
        "caffeine", 
        "ibuprofen",
        "sotorasib",
        "acetaminophen"
    ]
    
    for molecule in test_molecules:
        print(f"\nTesting molecule: {molecule}")
        result = name_to_smiles(molecule)
        print(f"Result: {result['markdown_output']}")