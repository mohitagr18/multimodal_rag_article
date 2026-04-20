import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image, ImageDraw

def create_sample_image(filename):
    """Create a sample image mimicking a chart."""
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Draw axes
    d.line([(50, 150), (350, 150)], fill=(0, 0, 0), width=2)
    d.line([(50, 150), (50, 20)], fill=(0, 0, 0), width=2)
    # Draw bars
    colors_list = [(200, 50, 50), (50, 200, 50), (50, 50, 200)]
    heights = [50, 100, 80]
    for i, h in enumerate(heights):
        x0 = 80 + i * 80
        y0 = 150 - h
        x1 = x0 + 40
        y1 = 150
        d.rectangle([x0, y0, x1, y1], fill=colors_list[i])
    img.save(filename)

def build_pdf(output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Heading1']
    title_style.alignment = 1
    story.append(Paragraph("Deep Learning for Document Understanding", title_style))
    story.append(Spacer(1, 20))
    
    # Abstract
    heading_style = styles['Heading2']
    story.append(Paragraph("Abstract", heading_style))
    story.append(Spacer(1, 10))
    abstract_text = ("We propose a new multimodal document representation pipeline. "
                     "By preserving physical layout traits such as bounding boxes and parsing structural elements "
                     "like figures and tables coherently, downstream retrieval tasks are dramatically improved "
                     "compared to naive text extraction.")
    story.append(Paragraph(abstract_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Text Section
    story.append(Paragraph("1. Introduction", heading_style))
    story.append(Spacer(1, 10))
    intro_text = ("Most RAG approaches treat documents as a flat sequence of words. This destroys vital "
                  "context, such as caption linkage to images, tabular column alignment, and visual emphasis. "
                  "Here, we demonstrate a structure-aware approach.")
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Image Section
    story.append(Paragraph("2. Proposed Architecture", heading_style))
    story.append(Spacer(1, 10))
    img_path = "sample_chart.png"
    create_sample_image(img_path)
    story.append(RLImage(img_path, width=400, height=200))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Figure 1: Performance comparison of various retrieval architectures.", styles['Italic']))
    story.append(Spacer(1, 20))
    
    # Table Section
    story.append(Paragraph("3. Results", heading_style))
    story.append(Spacer(1, 10))
    data = [
        ['Method', 'F1 Score', 'Latency (ms)'],
        ['Naive Text RAG', '65.2', '150'],
        ['Structure-Aware RAG', '82.5', '350'],
        ['Multimodal RAG (Ours)', '94.6', '800']
    ]
    t = Table(data, colWidths=[150, 100, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 5))
    story.append(Paragraph("Table 1: Quantitative results on OmniDocBench V1.5.", styles['Italic']))
    
    doc.build(story)
    
if __name__ == '__main__':
    build_pdf('sample_document.pdf')
    print("Created sample_document.pdf")
