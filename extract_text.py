import pdfplumber

pdf_path = r"C:\Users\Nayak\Downloads\WorldBank_Findex.pdf"

all_text = ""

with pdfplumber.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            all_text += text + "\n"
        else:
            print(f"Warning: Page {page_number} has no extractable text.")

with open("financial_literacy_data.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("Text extraction complete. Total length:", len(all_text))

