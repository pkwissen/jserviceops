from docx import Document

def read_document(file):
    if file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    # if file.name.endswith(".pdf"):
    #     reader = PdfReader(file)
    #     return "\n".join(p.extract_text() for p in reader.pages)

    raise ValueError("Unsupported file type")
