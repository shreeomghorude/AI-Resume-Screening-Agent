from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.parser import parse_resume
from app.ranking import rank_resumes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/rank")
async def rank(job_description: str = Form(...), files: List[UploadFile] = File(...)):
    resumes = []

    for f in files:
        contents = await f.read()
        text = parse_resume(f.filename, contents)
        resumes.append({
            "filename": f.filename,
            "text": text
        })

    results = rank_resumes(job_description, resumes)
    return {"results": results}
