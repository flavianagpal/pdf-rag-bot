from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
reader = PdfReader("data/isl.pdf")

# Combine all pages into one text
all_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        all_text += text + "\n"

print(f"Total characters: {len(all_text)}")

# Create chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split text into chunks
chunks = splitter.split_text(all_text)

print(f"\nTotal chunks created: {len(chunks)}")

print("\nFIRST CHUNK:\n")
print(chunks[0])

print("\n" + "="*80 + "\n")

print("SECOND CHUNK:\n")
print(chunks[1])