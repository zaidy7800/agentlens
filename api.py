from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def main_root():
    return {"Hello": "World"}
    