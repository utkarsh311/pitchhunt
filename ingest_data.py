import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

csv_file = "pitchhunt.csv"  # update path if needed
df = pd.read_csv(csv_file)

#Connect to Elasticsearch 
es = Elasticsearch("http://localhost:9200")

index_name = "pitchhunt"

#Define Index Mapping 
mapping = {
    "mappings": {
        "properties": {
            "Name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "Email": {"type": "keyword"},
            "Phone number": {"type": "keyword"},
            "City": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "Profession": {"type": "keyword"},
            "Field of Interest": {"type": "text"},
            "field_of_interest_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": "true",
                "similarity": "cosine",
            },
            "Why would you be interested in a platform that helps you build meaningful professional connections?": {
                "type": "text"
            },
            "How often do you network professionally?": {"type": "keyword"}
        }
    }
}



if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
es.indices.create(index=index_name, body=mapping)

# Load Embedding Model 
model = SentenceTransformer('all-MiniLM-L6-v2')


actions = []
for i, row in df.iterrows():
    # Generate embedding for "Field of Interest"
    embedding = model.encode(row["Field of Interest"]).tolist()

    doc = {
        "Name": row["Name"],
        "Email": row["Email"],
        "Phone number": str(row["Phone number"]),
        "City": row["City"],
        "Profession": row["Profession"],
        "Field of Interest": row["Field of Interest"],
        "field_of_interest_vector": embedding,
        "Why would you be interested in a platform that helps you build meaningful professional connections?": row["Why would you be interested in a platform that helps you build meaningful professional connections?"],
        "How often do you network professionally?": row["How often do you network professionally?"]
    }

    actions.append({
        "_index": index_name,
        "_id": i + 1,
        "_source": doc
    })


helpers.bulk(es, actions)

print(f"Indexed {len(actions)} documents into index '{index_name}'")
