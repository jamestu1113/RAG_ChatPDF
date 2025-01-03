# RAG Chatbot with LLaMA Model

## 說明
此專案展示了使用 LLaMA 模型與 LangChain 的檔案檢索問答（RAG）應用。使用者可以上傳 PDF 文件，系統會解析文件內容並根據問題給出回答。

## 功能
- 上傳 PDF 文件並解析內容
- 分割文本並嵌入向量
- 基於 LLaMA 模型的問答能力
- 使用 Streamlit 提供互動式介面

---

## 安裝

### 1. 創建虛擬環境並安裝依賴

```bash
conda create -n rag python=3.9 -y
conda activate rag
pip install -r requirements.txt
```
### 2. 下載並解壓縮模型
下載連接: [Llama 3 8B Chinese](<https://jamestu-my.sharepoint.com/:u:/g/personal/jamestu13_jamestu_onmicrosoft_com/EYl3uCVKlwxNs89KcoHPvtYBr_z-JdbtQ9gb8a7lra_HyA?e=YTVSVQ>)
```bash
下載完成後解壓縮到此專案的資料夾內
```
### 3. 模型配置
```bash
確保您的本地 LLaMA 模型目錄中包含 config.json, tokenizer.json, 以及權重文件等必要內容。
```

## 使用方式
### 1. 啟動應用
在終端執行以下命令：

```bash
streamlit run pdfChatbotV1.py
