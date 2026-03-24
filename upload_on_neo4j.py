import json, os
from neo4j import GraphDatabase


# # 1. Conect to Neo4j

driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=('neo4j','myinstance123')
)



# load data 

with open(r'arxiv_papers.json','r', encoding='utf-8') as f:
    arxiv_data = json.load(f)
    
    
with open(r'pubmed_multiple_queries1.json','r', encoding='utf-8') as f:
    pubmed_data = json.load(f)    
    
    
data = arxiv_data + pubmed_data

# load parsed json files 
folder_path = r'parsed_output_pdf'

for file in os.listdir(folder_path):
    if file.endswith(".json"):
        
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            parsed = json.load(f)
            
            paper={
                "title": parsed.get('metadata',{}).get('title'),
                "authors": parsed.get('metadata',{}).get('authors',[]),
                "insight": parsed.get("insigth", {})
            }
            
            data.append(paper)
            
            
# insert Graph 

def create_graph(tx, paper):
    
    title = paper.get("title")   
    
    tx.run(
      """ 
      MERGE (p:Paper {title:$title})
      """,
      title =title
    )
    
    # authors
    
    authors = paper.get("authors",[])
    
    if isinstance(authors, str):
        authors = authors.split(",")
        
    for author in authors:
         tx.run(
            
            """
            MERGE (a:Author {name:$author})
            MERGE (p:Paper {title:$title})
            MERGE (a)-[:WROTE]->(p)
            """,
            author = author.strip(),
            title=title
        )
         
    insight = paper.get("insight", {})
    
    if insight:
        # domains
        for domain in insight.get("domain",[]):
            tx.run(
                
                """
                MERGE (d:Domain {name:$domain})
                MERGE (p:Paper {title:$title})
                MERGE (p)-[:BELONGS_TO]->(d)
                """,
                domain = domain,
                title=title
            )
        
        # methods
        for method in insight.get("methods",[]):
            tx.run(
                
                """
                MERGE (m:Method {name:$method})
                MERGE (p:Paper {title:$title})
                MERGE (p)-[:USES]->(m)
                """,
                method = method,
                title=title
            )
            
        # metrics
        for metric in insight.get("metrics",[]):
            tx.run(
                
                """
                MERGE (m:Metric {name:$metric})
                MERGE (p:Paper {title:$title})
                MERGE (p)-[:EVALUATED_BY]->(m)
                """,
                metric = metric,
                title=title
            )
    
# 4 Insert data into Neo4j

with driver.session() as session:
    for paper in data:
        session.execute_write(create_graph, paper)

    # RUN QUERY 
    result = session.run("MATCH (p:Paper) RETURN p.title ")

    print("\nSample Papers from Neo4j:")
    for record in result:
        print(record["p.title"])

print('Knowledge Graph Created')    
#print("Total papers loaded:", len(data))