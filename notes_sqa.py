import boto3
import json
import logging
import botocore
import time
import whisper
import fastapi
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from dotenv import load_dotenv
load_dotenv()
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
aws_region_name = os.environ["AWS_REGION"]

# s3 = boto3.client('s3',aws_access_key_id=aws_access_key_id,
#                       aws_secret_access_key=aws_secret_access_key,
#  region_name='us-east-1'                     region_name="us-east-1")

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
def setup_bedrock_client():
    try:
        client = boto3.client(service_name='bedrock-runtime',aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region_name)
        logging.info("Successfully set up Bedrock client.")
        return client
    except Exception as e:
        logging.error(f"Failed to set up Bedrock client: {e}")
        raise

def transcribe_audio_with_whisper(file_path, model_size="base"):
    # Measure model load time
    start_time = time.time()
    model = whisper.load_model(model_size)
    end_time = time.time()
    model_load_time = end_time - start_time
    print('Model load time:', model_load_time)

    # Measure transcription time
    st = time.time()
    transcript = model.transcribe(file_path)
    et = time.time()
    transcription_time = et - st
    print('Transcription time:', transcription_time)
    transcript_text = transcript['text']
    print(transcript_text)
    # Return the transcribed text
    return transcript_text


def notes_prompt(transcript_text):
    prompt = """
        Your task is to convert the provided transcript into well-organized Markdown notes formatted for Jupyter notebooks. Please follow these guidelines to ensure clarity, accuracy, and consistency in rendering mathematical content:
 
        Headings:
        
        Use headings with varying numbers of # symbols to denote different heading levels.
        Inline Math:
        
        Use single dollar signs $ for inline math expressions. Example: $e^{i\pi} + 1 = 0$.
        Display Math:
        
        Use double dollar signs $$ for display math expressions. Example: $$ e^{i\pi} + 1 = 0 $$.
        Uniform Syntax:
        
        Maintain uniform syntax throughout the response to ensure consistency and readability.
        Example:
        
        Here is an example of how a heading and math expression should look:
        # Heading 1
        ## Heading 2
        $ e^{i\pi} + 1 = 0 $
        $$ e^{i\pi} + 1 = 0 $$
        Please convert the following transcript accordingly:
        
        Transcript:
        """
    notes_prompt =prompt+transcript_text
    return notes_prompt

def summary_qa_prompt(transcript_text):
    prompt = f"""
    As a teacher who tries to make sure that a student understands each and every topic. 
    Identify the topics and subtopics in the text.
    can you generate summary.
    can you generate questions based on the identified topics and subtopics.
    Remove all the preambles.
    Here is the transcript text.
    TEXT: {transcript_text} 
    
    Validate the questions and the options, and make sure they are based on factually correct information. 
    """
    format_prompt = """
    It is very important that the response is a json with following format:
        {"Summary": generated summary,
        "Questions":[
            {{"Question": "Question",
            "Options": ["option1", "option2", "option3", "option4"],
            "CorrectAnswer": 1}},
            {{"Question": "Question",
            "Options": ["option1", "option2", "option3", "option4"],
            "CorrectAnswer": 2}},
            .....]
        }  
    ###INSTRUCTIONS:
    1.Only four options for each question (option 1, option 2, option 3, option 4) 
    2.Correct answers can only be 1, 2, 3 or 4 
    (   1 = option1
        2 = option2
        3 = option3
        4 = option4 )
    3.Make sure the questions are based on factually correct information.
    4.Formulate the questions such that they have only one correct and clear option as the answer.
    """
    summary_qa_prompt = prompt + format_prompt
    
    return summary_qa_prompt
    

def bedrock_response(prompt):
    bedrock_llm = setup_bedrock_client()
    # Define the messages payload for the Bedrock API
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    }]

    # Define the request body for the Bedrock API
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "messages": messages
    })

    try:
        # Invoke the Bedrock API
        response = bedrock_llm.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=body
        )
    except botocore.exceptions.BotoCoreError as e:
        logger.error(f"Error invoking Bedrock API: {e}")
        raise
    # Parse the response from the Bedrock API
    response_body = json.loads(response.get('body').read())
    response_text = response_body['content'][0]['text']
    print(response_text)
    return response_text


# initialize FASTAPI
app = FastAPI()

UPLOAD_FOLDER = "uploads"

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Save the uploaded video locally
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Call a function to generate transcript (replace with your actual function)
        transcript = transcribe_audio_with_whisper(file_path)
        prompt_for_notes = notes_prompt(transcript)
        print(prompt_for_notes)
        prompt_for_sqa = summary_qa_prompt(transcript)
        print(prompt_for_sqa)
        notes= bedrock_response(prompt_for_notes)
        print(notes)
        sqa= bedrock_response(prompt_for_sqa)
        print(sqa)
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    return {"filename": file.filename, "transcript": transcript,"notes":notes, "sqa":sqa}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)