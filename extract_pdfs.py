import os
from pdfminer.high_level import extract_text

pdf_dir = "/scratch/leahk/eduardo.donestevez/fem-shell/elements_theory/"
files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

for pdf_file in files:
    try:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        text = extract_text(pdf_path)
        txt_path = pdf_path + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted: {pdf_file}")
    except Exception as e:
        print(f"Error extracting {pdf_file}: {e}")
