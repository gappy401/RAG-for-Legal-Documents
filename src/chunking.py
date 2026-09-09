"""
Hierarchical parent-child chunking for legal contracts.

Uses LangChain's RecursiveCharacterTextSplitter for boundary-aware splitting
(tries paragraph -> sentence -> word -> character breaks, in that order,
rather than blindly cutting at a fixed character count). Parent-child
pairing and absolute-offset tracking is handled directly here rather than
via LangChain's ParentDocumentRetriever, which was moved to the legacy
langchain_classic package as of LangChain 1.0.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_contract(
    text: str,
    parent_size: int = 2000,
    parent_overlap: int = 200,
    child_size: int = 400,
    child_overlap: int = 50,
) -> list[dict]:
    """
    Returns a flat list of child chunk dicts, each carrying a reference
    to its parent's full text and both offsets as absolute positions in
    the original document.
    """
    chunks = []

    # add_start_index=True makes the splitter record where each chunk
    # starts within the text it was given -- this is what lets us
    # reconstruct absolute offsets after splitting twice (once for
    # parents, again for children within each parent).
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        add_start_index=True,
    )

    parent_docs = parent_splitter.create_documents([text])

    for parent_idx, parent_doc in enumerate(parent_docs):
        parent_id = f"p{parent_idx}"
        parent_text = parent_doc.page_content
        parent_offset = parent_doc.metadata["start_index"]

        child_docs = child_splitter.create_documents([parent_text])

        for child_idx, child_doc in enumerate(child_docs):
            child_text = child_doc.page_content
            # start_index here is relative to parent_text -- add the
            # parent's own offset to get the absolute position in the
            # original document
            absolute_start = parent_offset + child_doc.metadata["start_index"]

            chunks.append({
                "child_id": f"{parent_id}_c{child_idx}",
                "child_text": child_text,
                "child_start": absolute_start,
                "parent_id": parent_id,
                "parent_text": parent_text,
                "parent_start": parent_offset,
            })

    return chunks