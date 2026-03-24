import json


documents = []
metadatas = []

# step 1: add data from parsed_output folder
import os

parsed_folder = "parsed_output_pdf"

for filename in os.listdir(parsed_folder):
    if filename.endswith(".json"):
        
        file_path = os.path.join(parsed_folder, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            
            # ⚠️ your key is "insigth" (typo in JSON)
            insight = data.get("insigth", {})  

            text = f"""
            Title: {metadata.get('title', '')}
            Authors: {", ".join(metadata.get('authors', []))}
            Publication Year: {metadata.get('publication_year', '')}
            DOI: {metadata.get('doi', '')}
            Keywords: {", ".join(metadata.get('keywords', []))}
            
            Domain: {", ".join(insight.get("domain", []))}
            Research Problem: {insight.get("research_problem", "")}
            Methods: {", ".join(insight.get("methods", []))}
            Datasets: {", ".join(insight.get("datasets", []))}
            Metrics: {", ".join(insight.get("metrics", []))}
            Key Findings: {insight.get("key_findings", "")}
            Limitations: {insight.get("limitations", "")}
            Future Directions: {insight.get("future_directions", "")}
            
            Abstract:
            {data.get("abstract", "")}
            
            Summary:
            {data.get("summary", "")}
            """

            documents.append(text)

            metadatas.append({
                "paper_id": data.get("document_id"),
                "title": metadata.get("title"),
                "source": data.get("source_file"),
                "publication_year": metadata.get("publication_year"),
                "domain": insight.get("domain", []),
            })

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# step 2: add data from arxiv papers
with open("arxiv_papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
    


for paper in papers:
    
    insight = paper.get("insight", {})
    
    text = f"""
    Title: {paper.get('title', '')}
    Authors: {paper.get('authors', '')}
    Published: {paper.get('published', '')}
    Categories: {paper.get('categories', '')}
    
    Domain: {", ".join(insight.get("domain", []))}
    Research Problem: {insight.get("research_problem", "")}
    Methods: {", ".join(insight.get("methods", []))}
    Datasets: {", ".join(insight.get("datasets", []))}
    Metrics: {", ".join(insight.get("metrics", []))}
    Key Findings: {insight.get("key_findings", "")}
    Limitations: {insight.get("limitations", "")}
    Future Directions: {insight.get("future_directions", "")}
    
    Abstract:
    {paper.get("abstract", "")}
    """
    
    documents.append(text)
    
    metadatas.append({
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "source": paper.get("source"),
        "categories": paper.get("categories"),
        "domain": insight.get("domain", []),
    })

# step 3: add data from pubmed papers

with open("pubmed_multiple_queries1.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
    


for paper in papers:
    
    insight = paper.get("insight", {})
    print(insight)
    text = f"""
    Title: {paper.get('title', '')}
    Authors: {paper.get('authors', '')}
    Journal: {paper.get('journal', '')}
    Keywords: {paper.get('keywords', [])}
    
    Domain: {", ".join(insight.get("domain", [])) if insight else None} 
    Research Problem: {insight.get("research_problem", "")  if insight else None}
    Methods: {", ".join(insight.get("methods", []))  if insight else None}
    Datasets: {", ".join(insight.get("datasets", []))  if insight else None}
    Metrics: {", ".join(insight.get("metrics", []))  if insight else None}
    Key Findings: {insight.get("key_findings", "")  if insight else None}
    Limitations: {insight.get("limitations", "")  if insight else None}
    Future Directions: {insight.get("future_directions", "")  if insight else None}
    
    Abstract:
    {paper.get("abstract", "")}
    """
    
    documents.append(text)
    
    metadatas.append({
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "source": "pubmed",
        "journal": paper.get('journal', ''),
        "domain": insight.get("domain", [])  if insight else None,
    })
    
    
print("length of all documents", len(documents))

# add data in database     
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS

vector_db = FAISS.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas
)

print(f"Number of vectors in index: {vector_db.index.ntotal}")

vector_db.save_local("research_papers_faiss")  

print("FAISS index documents saved successfully")