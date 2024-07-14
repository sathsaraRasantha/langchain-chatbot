from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

docs_path = 'docs/'
vectorstore_path = 'vectorstore/FAISS'

def vectorstore():
    loader = DirectoryLoader(docs_path,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500,chunk_overlap= 100)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(vectorstore_path)

if __name__=='__main__':
    vectorstore()
