from pypdf import PdfReader

reader = PdfReader("data/isl.pdf")

all_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        all_text += text + "\n"

print(f"Total characters: {len(all_text)}")

print("\nFirst 1000 characters:\n")
print(all_text[:1000])