import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def generate_pdf_report(query: str, synthesis: str, route: str) -> bytes:
    """
    Takes the orchestrator's output and formats it into a professional PDF brief.
    Returns the PDF file as raw bytes for Streamlit to download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        spaceAfter=14,
        textColor="#1E3A8A" # A nice corporate blue
    )
    heading_style = styles['Heading2']
    body_style = styles['BodyText']
    
    Story = []
    
    # 1. Title
    Story.append(Paragraph("⚡ PromptForge AI - Executive Brief", title_style))
    Story.append(Spacer(1, 12))
    
    # 2. Metadata / Route
    Story.append(Paragraph(f"<b>Execution Pipeline:</b> {route.upper()} Route", body_style))
    Story.append(Spacer(1, 12))
    
    # 3. The Original Query
    Story.append(Paragraph("<b>Business Inquiry:</b>", heading_style))
    Story.append(Paragraph(query, body_style))
    Story.append(Spacer(1, 12))
    
    # 4. The Synthesized Answer
    Story.append(Paragraph("<b>AI Synthesis & Analysis:</b>", heading_style))
    
    # ReportLab needs <br/> for line breaks instead of \n
    # Also, we do a quick strip of markdown asterisks (**) so it renders cleanly
    clean_synthesis = synthesis.replace('\n', '<br/>').replace('**', '')
    
    Story.append(Paragraph(clean_synthesis, body_style))
    
    # Build the PDF
    doc.build(Story)
    
    # Extract bytes and close buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes