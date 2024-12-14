import os
import time
from datetime import datetime, timedelta, timezone
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

def split_doc(filename_):
    print(f'Reading - {filename_}')
    loader = TextLoader(filename_, encoding="utf-8")
    documents = loader.load()
    text_splitter = SemanticChunker(
       embed, breakpoint_threshold_type="gradient")
    docs = text_splitter.create_documents([doc.page_content for doc in documents])
    return docs

def add_metadata(data,time):
    for chunk in data:
        chunk.metadata['last_update'] = time
    return data


def create_index(index_name,pc):
    name = index_name
    if name not in pc.list_indexes().names():
        pc.create_index(
            name="example-index",
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment="us-west1-gcp",
                pod_type="p1.x1",
                pods=1,
                metadata_config = {
                "indexed": ["genre"]
                }
            ),
            deletion_protection="disabled"
        )   

    while not pc.describe_index(name).status['ready']:
        time.sleep(1)

    return 0

def upload_documents(documents,index_name,namespace):
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding = embed,
        namespace=namespace
    )
    print("data is upload to index")

    return docsearch

def main():
    print("Main method started..!")
    msft_q1 = split_doc('Data\\MSFT_q1_2024.txt')
    msft_q2 = split_doc('Data\\MSFT_q2_2024.txt')

    q2_time = (datetime.now(timezone.utc) - timedelta(days=90)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
    )
    q1_time = (datetime.now(timezone.utc) - timedelta(days=180)).strftime(
        "%Y-%m-%dT%H:%M:%S-00:00"
    )

    msft_q1 = add_metadata(msft_q1,q1_time)
    msft_q2 = add_metadata(msft_q2,q2_time)

    documents = msft_q1 + msft_q2
    print("Document is ready ..!")
    index_name = "new"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("pc object created, ready to create index now ..!")
    index = create_index(index_name,pc)
    print("Index got created..!")
    namespace = "microsoft"
    data =upload_documents(documents,index_name,namespace)
    print("upload_documents got created..!")
    print(pc.Index(index_name).describe_index_stats())

    results = data.similarity_search_with_score(query="How is Windows OEM revenue growth?",k=1)
    for doc, score in results:
        print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

if __name__ == "__main__":
    main()