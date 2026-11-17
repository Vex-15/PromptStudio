"""Start Prompt Studio:  python run.py  (or: uvicorn server:app --port 8000)"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
