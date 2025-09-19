from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# === Setup ===
app = Flask(__name__)

es = Elasticsearch("http://localhost:9200")  # no security
index_name = "pitchhunt"

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# === 1) Search by Profession (keyword match) ===
@app.route("/search/profession", methods=["POST"])
def search_profession():
    data = request.get_json()
    profession = data.get("q") if data else None
    if not profession:
        return jsonify({"error": "Missing 'q' in request body"}), 400

    query = {
        "_source": {"excludes": ["field_of_interest_vector"]},
        "query": {
            "match": {
                "Profession": profession
            }
        }
    }

    res = es.search(index=index_name, body=query)
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    return jsonify(hits)


# === 2) Search by City (keyword match) ===
@app.route("/search/city", methods=["POST"])
def search_city():
    data = request.get_json()
    city = data.get("q") if data else None
    if not city:
        return jsonify({"error": "Missing 'q' in request body"}), 400

    query = {
        "_source": {"excludes": ["field_of_interest_vector"]},
        "query": {
            "match": {
                "City": city
            }
        }
    }

    res = es.search(index=index_name, body=query)
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    return jsonify(hits)


# === 3) Semantic Search on Field of Interest (kNN) ===
@app.route("/search/field_of_interest", methods=["POST"])
def search_field_of_interest():
    data = request.get_json()
    user_query = data.get("q") if data else None
    if not user_query:
        return jsonify({"error": "Missing 'q' in request body"}), 400

    # Encode query using sentence-transformers
    query_vector = model.encode(user_query).tolist()

    query = {
        "_source": {"excludes": ["field_of_interest_vector"]},
        "knn": {
                "field":"field_of_interest_vector",
                "query_vector": query_vector,
                "k": 2 ,
                "num_candidates":5 # top results
            }
        }
    

    res = es.search(index=index_name, body=query)
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    return jsonify(hits)


# === Run Flask App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
