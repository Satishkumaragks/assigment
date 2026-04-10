# =========================================
# 0. Imports
# =========================================
from template import get_models , list_models
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

# =========================================
# 1. Document Loading
# =========================================
print("\n========== DOCUMENT LOADING ==========\n")

all_docs = []

# ---- TXT LOADING ----
txt_path = "sample.txt"

if os.path.exists(txt_path):
    txt_loader = TextLoader(txt_path, encoding="utf-8")
    txt_docs = txt_loader.load()
    all_docs.extend(txt_docs)
    print(f"Loaded TXT: {txt_path}")
else:
    print(f" TXT not found: {txt_path}")

# ---- PDF LOADING ----
pdf_path = "sample.pdf"

if os.path.exists(pdf_path):
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f" Loaded PDF: {pdf_path}")
    except Exception as e:
        print(f" Invalid PDF skipped: {e}")
else:
    print(f" PDF not found: {pdf_path}")

# ---- WEB LOADING ----
web_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

try:
    web_loader = WebBaseLoader(web_url)
    web_docs = web_loader.load()
    all_docs.extend(web_docs)
    print(f" Loaded Web: {web_url}")
except Exception as e:
    print(f" Web load failed: {e}")

# ---- PRINT SAMPLE ----
for i, doc in enumerate(all_docs[:3]):
    print(f"\n--- Document {i+1} ---")
    print("Content:", doc.page_content[:300])
    print("Metadata:", doc.metadata)


# =========================================
# 2. Text Splitting
# =========================================
print("\n========== TEXT SPLITTING ==========\n")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

recursive_chunks = recursive_splitter.split_documents(all_docs)

char_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

char_chunks = char_splitter.split_documents(all_docs)

print(f"Recursive chunks count: {len(recursive_chunks)}")
print(f"Character chunks count: {len(char_chunks)}")

print("\n--- Observations ---")
print("1. Recursive splitter preserves semantic structure (better context).")
print("2. Character splitter splits blindly at fixed size.")
print("3. Recursive chunks improve LLM understanding.")


# =========================================
# 3. LCEL Chain
# =========================================
print("\n========== LCEL CHAIN ==========\n")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document analyst."),
    ("human", "Summarize the following text:\n\n{text}")
])



llm = get_models(model="nvidia/nemotron-3-nano-4b", temperature=0.7, max_tokens=100)

parser = StrOutputParser()

chain = prompt | llm | parser


# =========================================
# 4. Batch Processing
# =========================================
print("\n========== BATCH PROCESSING ==========\n")

batch_inputs = [{"text": chunk.page_content} for chunk in recursive_chunks[:5]]

batch_results = chain.batch(batch_inputs)

for i, result in enumerate(batch_results):
    print(f"\n--- Summary {i+1} ---")
    print(result)


# =========================================
# 5. Streaming
# =========================================
print("\n========== STREAMING ==========\n")

if recursive_chunks:
    sample_text = recursive_chunks[0].page_content

    stream = chain.stream({"text": sample_text})

    print("Streaming response:\n")

    for token in stream:
        print(token, end=" | ")
else:
    print(" No chunks available")

print("\n\n========== DONE ==========")
