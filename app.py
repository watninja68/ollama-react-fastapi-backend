import os
import re
import shutil
import uuid
from zipfile import ZipFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pdfplumber
from PIL import Image
from docx import Document  # For DOCX text extraction
from pptx import Presentation  # For PPTX text extraction

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
# Import the LangChain Document schema (aliasing to avoid conflict with python-docx Document)
from langchain.schema import Document as LC_Document

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration and variables
folder_path = "db"
cached_llm = Ollama(model="deepseek-r1:latest")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

def clean_text(text):
    """Utility function to clean extracted text."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_documents(input_directory, image_output_dir="extracted_images"):
    """
    Walk through input_directory, process each PDF/DOCX/PPTX file,
    extract text (and images), and return a dict of {file_name: extracted_text}.
    Images are saved in image_output_dir.
    """
    # Recreate image output folder
    if os.path.exists(image_output_dir):
        shutil.rmtree(image_output_dir)
    os.makedirs(image_output_dir, exist_ok=True)

    extracted_data = {}
    for root, _, files in os.walk(input_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)

            if file_ext.lower() == '.pdf':
                text = extract_text_and_images_from_pdf(file_path, file_name, image_output_dir)
                extracted_data[file_name] = text
            elif file_ext.lower() == '.docx':
                text = extract_text_and_images_from_word(file_path, file_name, image_output_dir)
                extracted_data[file_name] = text
            elif file_ext.lower() == '.pptx':
                text = extract_text_and_images_from_ppt(file_path, file_name, image_output_dir)
                extracted_data[file_name] = text

    return extracted_data

def extract_text_and_images_from_pdf(pdf_path, file_name, image_output_dir):
    """
    Extract text from a PDF using pdfplumber.
    For each 'image' bounding box in the PDF, render the page and crop that region.
    """
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text() or ""
            extracted_text += clean_text(text) + " "

            # Attempt to extract "images" by rendering the page and cropping bounding boxes
            for img_index, img_dict in enumerate(page.images):
                x0 = img_dict["x0"]
                top = img_dict["top"]
                x1 = img_dict["x1"]
                bottom = img_dict["bottom"]

                # Render entire page to an image (set resolution as needed)
                page_image = page.to_image(resolution=150)

                # Crop the bounding box where the image is located
                cropped = page_image.crop((x0, top, x1, bottom))

                # Save the cropped region as PNG
                image_filename = os.path.join(
                    image_output_dir,
                    f"{file_name}_page{page_number+1}_img{img_index+1}.png"
                )
                cropped.save(image_filename, format="PNG")
    return extracted_text

def extract_text_and_images_from_word(docx_path, file_name, image_output_dir):
    """
    Extract text from a Word document.
    Additionally, extract images by unzipping the DOCX and saving files from the "word/media" folder.
    """
    extracted_text = ""
    doc = Document(docx_path)
    for para in doc.paragraphs:
        extracted_text += clean_text(para.text) + " "

    # Extract images from DOCX
    try:
        with ZipFile(docx_path, 'r') as docx_zip:
            for item in docx_zip.namelist():
                if item.startswith("word/media/"):
                    image_filename = os.path.join(image_output_dir, f"{file_name}_{os.path.basename(item)}")
                    with docx_zip.open(item) as image_file, open(image_filename, 'wb') as f:
                        shutil.copyfileobj(image_file, f)
    except Exception as e:
        print(f"Error extracting images from DOCX {docx_path}: {e}")

    return extracted_text

def extract_text_and_images_from_ppt(ppt_path, file_name, image_output_dir):
    """
    Extract text from a PowerPoint presentation.
    Additionally, extract images by unzipping the PPTX and saving files from the "ppt/media" folder.
    """
    extracted_text = ""
    pres = Presentation(ppt_path)
    for i, slide in enumerate(pres.slides):
        slide_text = ""
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text += clean_text(shape.text) + " "
        extracted_text += slide_text

    # Extract images from PPTX
    try:
        with ZipFile(ppt_path, 'r') as pptx_zip:
            for item in pptx_zip.namelist():
                if item.startswith("ppt/media/"):
                    image_filename = os.path.join(image_output_dir, f"{file_name}_{os.path.basename(item)}")
                    with pptx_zip.open(item) as image_file, open(image_filename, 'wb') as f:
                        shutil.copyfileobj(image_file, f)
    except Exception as e:
        print(f"Error extracting images from PPTX {ppt_path}: {e}")

    return extracted_text

raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are a helpful assistant tasked with answering questions based on the provided context.
    Please follow these guidelines:
    - Answer only based on the information provided in the context
    - If the information is not in the context, say "I don't have enough information to answer this question"
    - Be concise and direct in your answers
    - Do not make assumptions beyond what is stated in the context
    Question: {input}
    Context: {context}
    Answer: [/INST]</s>
    """
)

# Pydantic model for incoming JSON requests
class QueryRequest(BaseModel):
    query: str

@app.post("/ai")
async def ai_post(query_req: QueryRequest):
    print("POST /ai called")
    query = query_req.query
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    return {"answer": response}

@app.post("/ask_content")
async def ask_pdf_post(query_req: QueryRequest):
    print("POST /ask_pdf called")
    query = query_req.query
    print(f"query: {query}")
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    print(result)
    sources = []
    for doc in result["context"]:
        sources.append({
            "source": doc.metadata["source"],
            "page_content": doc.page_content
        })
    return {"answer": result["answer"], "sources": sources}

@app.post("/pdf")
async def pdf_post(file: UploadFile = File(...)):
    """
    Endpoint to handle a PDF (or DOCX/PPTX) upload, extract text & images,
    store embeddings in the Chroma DB, and return the extracted text as JSON.
    """
    upload_dir = "temp_uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Process the folder with the new file
    extracted_data = process_documents(upload_dir, "extracted_images")

    # Prepare documents for embedding by splitting the extracted text into chunks.
    docs = []
    for filename, text in extracted_data.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            docs.append(LC_Document(page_content=chunk, metadata={"source": filename}))

    # If the vector store exists, add the new documents; otherwise, create a new store.
    if os.path.exists(folder_path) and os.listdir(folder_path):
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
        vector_store.add_documents(docs)
        vector_store.persist()
    else:
        vector_store = Chroma.from_documents(docs, embedding, persist_directory=folder_path)

    return JSONResponse({
        "filename": file.filename,
        "extracted_data": extracted_data
    })

@app.delete("/delete_embeddings")
async def delete_embeddings():
    """
    DELETE endpoint that removes all embeddings stored in the Chroma DB by deleting the folder.
    """
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)  # Recreate an empty folder for future use
            print("Embeddings deleted and folder recreated.")
            return {"status": "Successfully deleted all embeddings from Chroma DB"}
        else:
            print("Embeddings folder does not exist.")
            return {"status": "No embeddings to delete"}
    except Exception as e:
        print(f"Error deleting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, debug=True)

