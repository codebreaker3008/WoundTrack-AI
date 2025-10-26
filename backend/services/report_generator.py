"""
PDF Report Generator for Wound Analysis
Creates detailed medical reports with charts and images
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import io
import json

class WoundReportGenerator:
    """Generate PDF reports for wound analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.create_custom_styles()
    
    def create_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Normal text style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=6
        )
    
    def generate_report(self, wound_data, output_path):
        """
        Generate complete wound analysis report
        
        Args:
            wound_data: Dictionary containing wound analysis data
            output_path: Path to save PDF report
        
        Returns:
            Path to generated report
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Wound Healing Analysis Report", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Patient Information
        story.extend(self._add_patient_info(wound_data))
        story.append(Spacer(1, 0.3*inch))
        
        # Wound Measurements
        story.extend(self._add_measurements(wound_data))
        story.append(Spacer(1, 0.3*inch))
        
        # Infection Risk Assessment
        story.extend(self._add_infection_assessment(wound_data))
        story.append(Spacer(1, 0.3*inch))
        
        # Wound Images
        story.extend(self._add_images(wound_data))
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.extend(self._add_recommendations(wound_data))
        
        # Disclaimer
        story.append(PageBreak())
        story.extend(self._add_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _add_patient_info(self, wound_data):
        """Add patient information section"""
        elements = []
        
        elements.append(Paragraph("Patient Information", self.heading_style))
        
        data = [
            ['Report Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Analysis Date:', datetime.fromisoformat(wound_data['timestamp']).strftime('%B %d, %Y at %I:%M %p')],
            ['Wound ID:', wound_data['wound_id']],
        ]
        
        if wound_data.get('patient_id'):
            data.append(['Patient ID:', wound_data['patient_id']])
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        return elements
    
    def _add_measurements(self, wound_data):
        """Add wound measurements section"""
        elements = []
        
        elements.append(Paragraph("Wound Measurements", self.heading_style))
        
        measurements = wound_data.get('measurements', {})
        
        data = [
            ['Measurement', 'Value', 'Unit'],
            ['Area', f"{measurements.get('area_cm2', 0):.2f}", 'cm²'],
            ['Perimeter', f"{measurements.get('perimeter_cm', 0):.2f}", 'cm'],
            ['Length', f"{measurements.get('length_cm', 0):.2f}", 'cm'],
            ['Width', f"{measurements.get('width_cm', 0):.2f}", 'cm'],
        ]
        
        table = Table(data, colWidths=[2*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.1*inch))
        
        note = Paragraph(
            "<i>Note: Measurements are estimates. Include a reference object (coin/ruler) in photos for accuracy.</i>",
            self.normal_style
        )
        elements.append(note)
        
        return elements
    
    def _add_infection_assessment(self, wound_data):
        """Add infection risk assessment section"""
        elements = []
        
        elements.append(Paragraph("Infection Risk Assessment", self.heading_style))
        
        infection = wound_data.get('infection', {})
        risk_level = infection.get('level', 'Unknown')
        risk_prob = infection.get('probability', 0)
        
        # Color code based on risk
        if risk_level == 'High':
            risk_color = colors.HexColor('#F44336')
        elif risk_level == 'Medium':
            risk_color = colors.HexColor('#FF9800')
        else:
            risk_color = colors.HexColor('#4CAF50')
        
        data = [
            ['Risk Level', 'Probability', 'Assessment'],
            [risk_level, f"{risk_prob*100:.1f}%", self._get_risk_interpretation(risk_level)],
        ]
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (0, 1), risk_color),
            ('TEXTCOLOR', (0, 1), (0, 1), colors.whitesmoke),
            ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _get_risk_interpretation(self, risk_level):
        """Get interpretation text for risk level"""
        interpretations = {
            'Low': 'Wound shows minimal signs of infection. Continue current care.',
            'Medium': 'Some infection indicators present. Monitor closely for changes.',
            'High': 'Significant infection risk detected. Seek medical attention promptly.'
        }
        return interpretations.get(risk_level, 'Unable to assess infection risk.')
    
    def _add_images(self, wound_data):
        """Add wound images to report"""
        elements = []
        
        elements.append(Paragraph("Wound Visualization", self.heading_style))
        
        # Get image paths
        files = wound_data.get('files', {})
        overlay_path = files.get('overlay', '').replace('/results/', 'results/')
        
        if overlay_path and Path(overlay_path).exists():
            try:
                # Add wound image
                img = Image(overlay_path, width=4*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))
                caption = Paragraph(
                    "<i>Wound segmentation overlay (red area indicates detected wound)</i>",
                    self.normal_style
                )
                elements.append(caption)
            except Exception as e:
                print(f"Error adding image: {e}")
                elements.append(Paragraph("Image not available", self.normal_style))
        else:
            elements.append(Paragraph("Image not available", self.normal_style))
        
        return elements
    
    def _add_recommendations(self, wound_data):
        """Add care recommendations"""
        elements = []
        
        elements.append(Paragraph("Care Recommendations", self.heading_style))
        
        infection = wound_data.get('infection', {})
        risk_level = infection.get('level', 'Unknown')
        
        recommendations = [
            "• Keep wound clean and dry",
            "• Change dressings as recommended by healthcare provider",
            "• Monitor for signs of infection (redness, swelling, discharge)",
            "• Take photos regularly to track healing progress",
        ]
        
        if risk_level == 'High':
            recommendations.insert(0, "• <b>IMPORTANT: Seek professional medical attention promptly</b>")
            recommendations.append("• Watch for fever or increased pain")
        elif risk_level == 'Medium':
            recommendations.append("• Schedule follow-up if symptoms worsen")
        
        for rec in recommendations:
            elements.append(Paragraph(rec, self.normal_style))
        
        return elements
    
    def _add_disclaimer(self):
        """Add medical disclaimer"""
        elements = []
        
        elements.append(Paragraph("Medical Disclaimer", self.heading_style))
        
        disclaimer_text = """
        <b>IMPORTANT:</b> This report is generated by an AI-assisted wound analysis system 
        for informational and educational purposes only. It is <b>NOT</b> a substitute for 
        professional medical advice, diagnosis, or treatment. 
        <br/><br/>
        Always seek the advice of your physician or other qualified health provider with any 
        questions you may have regarding a medical condition. Never disregard professional 
        medical advice or delay in seeking it because of information from this report.
        <br/><br/>
        This system has not been evaluated or approved by the FDA and is not intended for 
        clinical diagnosis or treatment decisions.
        <br/><br/>
        <i>Report generated on: {}</i>
        """.format(datetime.now().strftime('%B %d, %Y at %I:%M %p'))
        
        elements.append(Paragraph(disclaimer_text, self.normal_style))
        
        return elements


def generate_wound_report(wound_id, upload_dir="uploads", output_dir="reports"):
    """
    Convenience function to generate report from wound ID
    
    Args:
        wound_id: Wound identifier
        upload_dir: Directory containing wound data
        output_dir: Directory to save report
    
    Returns:
        Path to generated report or None if failed
    """
    try:
        # Load wound data
        metadata_path = Path(upload_dir) / f"{wound_id}_metadata.json"
        
        if not metadata_path.exists():
            print(f"Wound data not found: {metadata_path}")
            return None
        
        with open(metadata_path, 'r') as f:
            wound_data = json.load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_filename = f"wound_report_{wound_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = output_path / report_filename
        
        generator = WoundReportGenerator()
        generator.generate_report(wound_data, report_path)
        
        print(f"✅ Report generated: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test report generation
    import sys
    
    if len(sys.argv) > 1:
        wound_id = sys.argv[1]
        report_path = generate_wound_report(wound_id)
        
        if report_path:
            print(f"\n✅ Report saved to: {report_path}")
        else:
            print("\n❌ Failed to generate report")
    else:
        print("Usage: python report_generator.py <wound_id>")