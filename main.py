import base64
import io
import json
import os
import logging
import random
import dotenv
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

from src.state_manager.state_manager import StateManager
from src.graph.build_graph import build_graph
from langchain_community.callbacks import get_openai_callback
from config import (AZURE_DEPLOYMENT_NAME_GPT_4, AZURE_OPENAI_API_BASE,
                    AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_TYPE,
                    AZURE_OPENAI_API_VERSION, OPENAI_GPT_MODEL)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S",  
)

dotenv.load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "PersonalInfoBot"

class ConverseAIBot(BaseModel):
    user_text:str
    user_id: int
    
@app.get("/health")
async def health_check():
    "Basic health check endpoint to verify if the app is running."
    return {"status": "healthy"}

@app.post("/converse")
async def converse_with_retailbuddy(request: ConverseAIBot):   
    try:
        user_text = request.user_text
        user_id = request.user_id
        random_user_id = random.randint(100, 999)
        logging.info(f"\n================ User Question ==================\n{user_text}\n")
        StateManager().set_company_and_user(user_id, user_text)
        config = {
            "configurable": {
                "user_id": random_user_id,
                "thread_id": random_user_id,
                "company_id": 1,
            }
        }
        logging.info(f"\n================ Building AI-bot graph... ==================\n")
        graph = build_graph()
        if not graph:
            logging.error("Failed to build the graph.")
            raise HTTPException(status_code=500, detail="Graph building failed.")
        
        logging.info(f"\n================ AI-bot graph built successfully. ==================\n")
        final_output = None

        with get_openai_callback() as cost:
            events = graph.stream(
                {"messages": ("user", user_text)}, config, stream_mode="values"
            )
            for event in events:
                messages = event.get("messages", [])
                if not messages:
                    logging.warning(f"No messages found in event: {event}")
                    continue

                last_message = messages[-1]
                if last_message.content:
                    final_output = last_message.content
        logging.info(f"\n======================= AI Response ========================\n{final_output}\n")
        logging.info(f"\n================ AI interaction completed. ==================\n")
        return {"response": final_output}

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")
    
    
def encode_image(file: UploadFile):
    try:
        image_content = file.file.read()
        return base64.b64encode(image_content).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error encoding image: " + str(e))

@app.post("/extract-text-from-image")
async def extract_text_from_image(file: UploadFile = File(...)):
    base64_image = encode_image(file)
    groq_api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=groq_api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": """Extract the text from the provided image exactly as it appears without answering any questions or interpreting the content.
                                    Your task is to simply return the extracted text as-is."""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
        )
        return {"extracted_text": chat_completion.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the image: " + str(e))
    
@app.post("/extract_text_from_pdf")
async def extract_text_from_pdf(pdf: UploadFile = File(...)):
    try:
        if not pdf.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")
        
        pdf_file = await pdf.read()
        reader = PdfReader(io.BytesIO(pdf_file))
        
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
        
        extracted_json_data = genterate_json_data(extracted_text.strip())
        return {"extracted_text": extracted_text.strip(), "extracted_json_data": extracted_json_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def genterate_json_data(user_input):
    try:
        # llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
        # llm = ChatOpenAI(model="gpt-3.5-turbo")
        llm = AzureChatOpenAI(
            openai_api_base= AZURE_OPENAI_API_BASE,
            openai_api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_type= AZURE_OPENAI_API_TYPE,
            model_name=AZURE_DEPLOYMENT_NAME_GPT_4,
            temperature=0.0
        )
        prompt = PromptTemplate(input_variables=["user_input"], template="""
            Correct the spelling errors in the following sentence: '{user_input}'
            If there are no spelling mistakes, return the input as it is, without any changes.
            Only correct the spelling mistakes; do not add any extra content or alter the meaning.
            Return the response in a simple, clean JSON format without escape characters for clarity.
            Please ensure that the output JSON is properly formatted and easy to read, following standard JSON syntax.
        """)       
        runnable = prompt | llm
        corrected_input = runnable.invoke({"user_input": user_input})
        result_content = corrected_input.content.strip('```json').strip('```').strip()
        return result_content
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise