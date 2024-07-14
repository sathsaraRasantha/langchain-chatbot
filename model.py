from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

vectorstore_path = 'vectorstore/FAISS'
model_name = 'llama-2-7b-chat.ggmlv3.q8_0.bin'
model_type = 'llama'
max_new_tokens = '500'
temperature = '0.5'

prompt_template = """Use the given information to answer users querries. If you don't know the answers do not make up an answer, 
just say that you do not have enough information about the question in a polite way.

Context: {context}
Question: {question}

Only return well strructured and helpful answers below and nothing else.
Answer:
"""

def set_prompt():
    
    prompt = PromptTemplate(template= prompt_template,input_variables=['context','question'])
    return prompt

def load_llm(model_name, model_type,max_new_tokens,temperature):
    llm = CTransformers(
        model= model_name,
        model_type=model_type,
        max_new_tokens = max_new_tokens,
        temperature= temperature
    )
    return llm


def QAChain(llm,prompt,vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def Bot(model_name, model_type,max_new_tokens,temperature):
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
    vectorstore = FAISS.load_local(vectorstore_path,embeddings,allow_dangerous_deserialization=True)
    llm = load_llm(model_name, model_type,max_new_tokens,temperature)
    qa_prompt = set_prompt()
    qa = QAChain(llm,qa_prompt,vectorstore)
    return qa

def final_response(query):
    qa_result = Bot(model_name, model_type,max_new_tokens,temperature)
    response = qa_result({'query':query})
    return response


@cl.on_chat_start
async def start():
    chain = Bot(model_name, model_type,max_new_tokens,temperature)
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
