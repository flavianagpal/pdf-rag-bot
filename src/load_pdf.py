from pypdf import PdfReader

reader = PdfReader("data/isl.pdf")

num_pages = len(reader.pages)

print(f"Total pages: {num_pages}")

page_20 = reader.pages[20]

text = page_20.extract_text()

print(text[:2000])