from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import os

def create_pdf(filename):
    c = canvas.Canvas(filename, pagesize=LETTER)
    width, height = LETTER
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Claude Agent Skills - PDF Translator Test")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "1. Introduction")
    c.drawString(100, 680, "This is a test PDF document created to demonstrate the PDF Translator Skill.")
    c.drawString(100, 660, "The goal is to extract this text and translate it.")
    
    c.drawString(100, 620, "2. Technical Details")
    c.drawString(100, 600, "The skill uses PyPDF2 for extraction and Claude for translation.")
    c.drawString(100, 580, "It demonstrates the progressive disclosure pattern.")
    
    c.drawString(100, 540, "3. Conclusion")
    c.drawString(100, 520, "Skills are a powerful way to extend Claude's capabilities.")
    
    c.save()
    print(f"PDF created at {filename}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "..", "test_sample.pdf")
    create_pdf(output_path)
