import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)


def get_error_message(response: httpx.Response) -> str:
    try:
        response_data = response.json()
        return str(response_data.get("detail", response.text))
    except Exception:
        return response.text


st.set_page_config(
    page_title="Clinical Evidence Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("Clinical Evidence Assistant")

st.caption(
    "Upload public clinical documents and ask questions "
    "using locally generated, citation-grounded answers."
)

with st.sidebar:
    st.header("System status")

    try:
        health_response = httpx.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        if health_response.status_code == 200:
            st.success("FastAPI backend connected")
        else:
            st.error("FastAPI backend returned an error")
    except httpx.RequestError:
        st.error("FastAPI backend is not running")

    st.info(
        "The current vector store is temporary. "
        "Indexed documents are removed whenever FastAPI restarts."
    )

st.header("1. Upload and index a document")

uploaded_file = st.file_uploader(
    "Choose a public, text-based PDF",
    type=["pdf"],
)

if st.button(
    "Index document",
    disabled=uploaded_file is None,
):
    if uploaded_file is not None:
        with st.spinner(
            "Extracting, chunking and embedding the document..."
        ):
            try:
                index_response = httpx.post(
                    f"{API_BASE_URL}/documents/index",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=300,
                )

                if index_response.status_code == 200:
                    index_result = index_response.json()

                    st.session_state["document_indexed"] = True

                    st.success(
                        f"Indexed {index_result['filename']} "
                        f"into {index_result['chunk_count']} chunks."
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Pages",
                        index_result["page_count"],
                    )
                    col2.metric(
                        "New chunks",
                        index_result["chunk_count"],
                    )
                    col3.metric(
                        "Total indexed chunks",
                        index_result["total_indexed_chunks"],
                    )
                else:
                    st.error(
                        get_error_message(index_response)
                    )

            except httpx.RequestError as exc:
                st.error(
                    f"Could not reach the FastAPI backend: {exc}"
                )

st.divider()
st.header("2. Ask a question")

question = st.text_area(
    "Question",
    placeholder=(
        "For example: What was the primary objective "
        "of the study?"
    ),
)

top_k = st.slider(
    "Number of source passages",
    min_value=1,
    max_value=5,
    value=3,
)

if st.button("Generate grounded answer"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner(
            "Retrieving evidence and generating an answer..."
        ):
            try:
                answer_response = httpx.post(
                    f"{API_BASE_URL}/answer",
                    json={
                        "query": question,
                        "top_k": top_k,
                    },
                    timeout=300,
                )

                if answer_response.status_code == 200:
                    answer_result = answer_response.json()

                    st.subheader("Answer")
                    st.markdown(answer_result["answer"])

                    if answer_result["citation_validation_passed"]:
                        st.success("Citation format validated")
                    else:
                        st.warning(
                            "Citation validation warning: "
                            + " ".join(answer_result["citation_warnings"])
                        )

                    st.subheader("Supporting evidence")

                    for citation in answer_result["citations"]:
                        label = (
                            f"[{citation['source_number']}] "
                            f"{citation['filename']} — "
                            f"page {citation['page_number']} — "
                            f"score {citation['similarity_score']:.4f}"
                        )

                        with st.expander(label):
                            st.write(citation["text"])
                else:
                    st.error(
                        get_error_message(answer_response)
                    )

            except httpx.RequestError as exc:
                st.error(
                    f"Could not reach the FastAPI backend: {exc}"
                )

st.divider()

st.caption(
    "For research assistance only. Answers must be verified "
    "against the cited source documents."
)