import os
import tempfile
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 本地模型設定
MODEL_PATH = "C:/Users/user/Desktop/chatpdf/llama-3-chinese-8b-instruct-v3"

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Streamlit Showcase：釋放 RAG 與 LangChain 的強大力量")

# 載入本地模型
@st.cache_resource
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model

try:
    tokenizer, model = load_llama_model()
except Exception as e:
    st.error(f"模型載入失敗：{e}")
    st.stop()

# 自定義 LLM 類別
class LocalLLM:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt, max_length=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"], max_length=max_length, temperature=temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

local_llm = LocalLLM(tokenizer, model)

# 載入文件
def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

# 分割文件
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts

# 嵌入向量處理
def embeddings_on_local_vectordb(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

# 自定義問答邏輯
def query_llm(retriever, query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"以下是一些相關內容：\n{context}\n\n使用者問題：{query}\n請根據上述內容回答問題。"
    response = local_llm.generate(prompt)
    return response

# 文件上傳區
def input_fields():
    st.session_state.source_docs = st.file_uploader(label="上傳文件", type="pdf", accept_multiple_files=True)

# 處理文件
def process_documents():
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)  # 確保臨時目錄存在
        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())

        documents = load_documents()
        for _file in TMP_DIR.iterdir():
            temp_file = TMP_DIR.joinpath(_file)
            temp_file.unlink()

        texts = split_documents(documents)
        st.session_state.retriever = embeddings_on_local_vectordb(texts)
    except Exception as e:
        st.error(f"文件處理過程中出錯：{e}")

# 啟動應用
def boot():
    input_fields()
    st.button("提交文件", on_click=process_documents)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])

    if query := st.chat_input():
        st.chat_message("human").write(query)

        if "retriever" in st.session_state:
            response = query_llm(st.session_state.retriever, query)
        else:
            response = local_llm.generate(query)

        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()
