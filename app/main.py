from fastapi import FastAPI

app = FastAPI(
    title="Clinical Evidence Assistant",
    description="A citation-grounded assistant for public clinical documents.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Clinical Evidence Assistant",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}