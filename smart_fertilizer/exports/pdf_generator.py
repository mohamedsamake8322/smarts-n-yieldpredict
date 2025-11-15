from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, blue, green, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib import colors
from datetime import datetime
from typing import Dict, List, Optional
import io
import os

class FertilizerReportGenerator:
    """
    Professional PDF report generator for fertilizer recommendations
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the report"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2E8B57'),
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=HexColor('#2E8B57'),
            borderWidth=1,
            borderColor=HexColor('#2E8B57'),
            borderPadding=5
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=HexColor('#1F5F3F'),
            borderWidth=0.5,
            borderColor=HexColor('#1F5F3F'),
            leftIndent=0,
            borderPadding=3
        ))
        
        # Highlight style
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=HexColor('#0066CC'),
            leftIndent=20,
            spaceAfter=10
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=HexColor('#FF6600'),
            leftIndent=20,
            spaceAfter=8,
            borderWidth=1,
            borderColor=HexColor('#FF6600'),
            borderPadding=5
        ))
    
    def generate_fertilizer_report(self, recommendation: Dict, output_path: str = None) -> bytes:
        """
        Generate comprehensive fertilizer recommendation report
        """
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build report content
        story = []
        
        # Title page
        story.extend(self._create_title_page(recommendation))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(recommendation))
        story.append(PageBreak())
        
        # Soil analysis section
        story.extend(self._create_soil_analysis_section(recommendation))
        
        # Crop requirements section
        story.extend(self._create_crop_requirements_section(recommendation))
        
        # Fertilizer recommendations
        story.extend(self._create_fertilizer_recommendations_section(recommendation))
        
        # Application schedule
        story.extend(self._create_application_schedule_section(recommendation))
        
        # Cost analysis
        story.extend(self._create_cost_analysis_section(recommendation))
        
        # Regional considerations
        story.extend(self._create_regional_considerations_section(recommendation))
        
        # Risk assessment
        story.extend(self._create_risk_assessment_section(recommendation))
        
        # Appendices
        story.extend(self._create_appendices(recommendation))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        
        return pdf_bytes
    
    def _create_title_page(self, recommendation: Dict) -> List:
        """Create title page"""
        
        story = []
        
        # Main title
        story.append(Paragraph("Smart Fertilizer Recommendation Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Farm information table
        farm_info = [
            ['Report ID:', recommendation.get('recommendation_id', 'N/A')],
            ['Generated:', recommendation.get('generated_at', datetime.now()).strftime('%B %d, %Y at %I:%M %p')],
            ['Region:', recommendation.get('region', 'N/A')],
            ['Crop:', recommendation.get('crop_selection', {}).get('crop_type', 'N/A').title()],
            ['Variety:', recommendation.get('crop_selection', {}).get('variety', 'N/A')],
            ['Area:', f"{recommendation.get('area_hectares', 0)} hectares"],
            ['Target Yield:', f"{recommendation.get('target_yield', 0)} tons/ha"],
            ['Currency:', recommendation.get('cost_analysis', {}).get('currency', 'USD')]
        ]
        
        farm_table = Table(farm_info, colWidths=[2*inch, 3*inch])
        farm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#F0F8F0')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#2E8B57')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(farm_table)
        story.append(Spacer(1, 1*inch))
        
        # Professional disclaimer
        disclaimer = """
        <b>Professional Agricultural Recommendation</b><br/><br/>
        This report provides scientifically-based fertilizer recommendations tailored to your specific 
        soil conditions, crop requirements, and regional agricultural practices. The recommendations 
        are generated using advanced agronomic models and regional databases.<br/><br/>
        <b>Important:</b> This report should be used in conjunction with local agricultural extension 
        services and professional agronomic advice. Always conduct soil tests before application and 
        follow local regulations for fertilizer use.
        """
        
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return story
    
    def _create_executive_summary(self, recommendation: Dict) -> List:
        """Create executive summary"""
        
        story = []
        story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        
        # Key metrics
        nutrient_balance = recommendation.get('nutrient_balance', {})
        cost_analysis = recommendation.get('cost_analysis', {})
        
        summary_text = f"""
        <b>Fertilizer Program Overview:</b><br/>
        • Total Nitrogen Required: {nutrient_balance.get('total_n', 0):.1f} kg/ha<br/>
        • Total Phosphorus Required: {nutrient_balance.get('total_p', 0):.1f} kg/ha<br/>
        • Total Potassium Required: {nutrient_balance.get('total_k', 0):.1f} kg/ha<br/>
        • Total Program Cost: {cost_analysis.get('currency', 'USD')} {cost_analysis.get('total_cost', 0):.2f}<br/>
        • Cost per Hectare: {cost_analysis.get('currency', 'USD')} {cost_analysis.get('cost_per_hectare', 0):.2f}<br/>
        • Expected Yield: {recommendation.get('expected_yield', 0):.1f} tons/ha<br/>
        • Return on Investment: {recommendation.get('roi_percentage', 0):.1f}%<br/>
        """
        
        story.append(Paragraph(summary_text, self.styles['Highlight']))
        
        # Key recommendations
        story.append(Paragraph("Key Recommendations:", self.styles['SectionHeader']))
        
        application_schedule = recommendation.get('application_schedule', [])
        if application_schedule:
            rec_text = "<b>Application Timeline:</b><br/>"
            for app in application_schedule[:3]:  # Show first 3 applications
                rec_text += f"• {app.get('stage', 'N/A')}: {app.get('fertilizer_type', 'N/A')} at {app.get('amount_kg_per_ha', 0):.1f} kg/ha<br/>"
            
            story.append(Paragraph(rec_text, self.styles['Normal']))
        
        return story
    
    def _create_soil_analysis_section(self, recommendation: Dict) -> List:
        """Create soil analysis section"""
        
        story = []
        story.append(Paragraph("Soil Analysis Results", self.styles['CustomSubtitle']))
        
        soil_analysis = recommendation.get('soil_analysis', {})
        
        # Soil properties table
        soil_data = [
            ['Parameter', 'Value', 'Unit', 'Status'],
            ['pH', f"{getattr(soil_analysis, 'ph', 0):.1f}", '', self._get_ph_status(getattr(soil_analysis, 'ph', 0))],
            ['Organic Matter', f"{getattr(soil_analysis, 'organic_matter', 0):.1f}", '%', self._get_om_status(getattr(soil_analysis, 'organic_matter', 0))],
            ['Available Nitrogen', f"{getattr(soil_analysis, 'nitrogen', 0):.1f}", 'ppm', self._get_nutrient_status(getattr(soil_analysis, 'nitrogen', 0), 'N')],
            ['Available Phosphorus', f"{getattr(soil_analysis, 'phosphorus', 0):.1f}", 'ppm', self._get_nutrient_status(getattr(soil_analysis, 'phosphorus', 0), 'P')],
            ['Available Potassium', f"{getattr(soil_analysis, 'potassium', 0):.1f}", 'ppm', self._get_nutrient_status(getattr(soil_analysis, 'potassium', 0), 'K')],
            ['CEC', f"{getattr(soil_analysis, 'cec', 0):.1f}", 'cmol/kg', self._get_cec_status(getattr(soil_analysis, 'cec', 0))],
            ['Soil Texture', f"{getattr(soil_analysis, 'texture', 'N/A')}", '', 'Classification']
        ]
        
        soil_table = Table(soil_data, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch])
        soil_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E8B57')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#2E8B57')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F8F8F8')])
        ]))
        
        story.append(soil_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Soil interpretation
        story.append(Paragraph("Soil Analysis Interpretation:", self.styles['SectionHeader']))
        
        interpretation = self._generate_soil_interpretation(soil_analysis)
        story.append(Paragraph(interpretation, self.styles['Normal']))
        
        return story
    
    def _create_crop_requirements_section(self, recommendation: Dict) -> List:
        """Create crop requirements section"""
        
        story = []
        story.append(Paragraph("Crop Nutrient Requirements", self.styles['CustomSubtitle']))
        
        crop_selection = recommendation.get('crop_selection', {})
        nutrient_balance = recommendation.get('nutrient_balance', {})
        
        # Crop information
        crop_info = f"""
        <b>Crop Details:</b><br/>
        • Crop Type: {crop_selection.get('crop_type', 'N/A').title()}<br/>
        • Variety: {crop_selection.get('variety', 'N/A')}<br/>
        • Planting Season: {crop_selection.get('planting_season', 'N/A').title()}<br/>
        • Growth Duration: {crop_selection.get('growth_duration', 0)} days<br/>
        • Target Yield: {recommendation.get('target_yield', 0)} tons/ha<br/>
        """
        
        story.append(Paragraph(crop_info, self.styles['Normal']))
        
        # Nutrient requirements table
        story.append(Paragraph("Calculated Nutrient Requirements:", self.styles['SectionHeader']))
        
        nutrient_data = [
            ['Nutrient', 'Required (kg/ha)', 'Application Method'],
            ['Nitrogen (N)', f"{nutrient_balance.get('total_n', 0):.1f}", 'Split applications'],
            ['Phosphorus (P)', f"{nutrient_balance.get('total_p', 0):.1f}", 'Basal application'],
            ['Potassium (K)', f"{nutrient_balance.get('total_k', 0):.1f}", 'Split applications']
        ]
        
        # Add secondary nutrients
        secondary = nutrient_balance.get('secondary_nutrients', {})
        for nutrient, amount in secondary.items():
            if amount > 0:
                nutrient_data.append([f"{nutrient.upper()}", f"{amount:.1f}", "As needed"])
        
        nutrient_table = Table(nutrient_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        nutrient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E8B57')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#2E8B57')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(nutrient_table)
        
        return story
    
    def _create_fertilizer_recommendations_section(self, recommendation: Dict) -> List:
        """Create fertilizer recommendations section"""
        
        story = []
        story.append(Paragraph("Recommended Fertilizers", self.styles['CustomSubtitle']))
        
        recommended_fertilizers = recommendation.get('recommended_fertilizers', [])
        
        if recommended_fertilizers:
            # Fertilizer table
            fert_data = [['Fertilizer', 'N-P-K Content', 'Price/kg', 'Availability']]
            
            for fert in recommended_fertilizers:
                fert_data.append([
                    fert.name,
                    f"{fert.n_content}-{fert.p_content}-{fert.k_content}",
                    f"{fert.price_per_kg:.2f}",
                    fert.availability.title()
                ])
            
            fert_table = Table(fert_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
            fert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E8B57')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#2E8B57')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F8F8F8')])
            ]))
            
            story.append(fert_table)
        
        return story
    
    def _create_application_schedule_section(self, recommendation: Dict) -> List:
        """Create application schedule section"""
        
        story = []
        story.append(Paragraph("Fertilizer Application Schedule", self.styles['CustomSubtitle']))
        
        application_schedule = recommendation.get('application_schedule', [])
        
        if application_schedule:
            # Schedule table
            schedule_data = [['Stage', 'Days After Planting', 'Fertilizer', 'Rate (kg/ha)', 'Method']]
            
            for app in application_schedule:
                schedule_data.append([
                    app.stage.replace('_', ' ').title(),
                    str(app.days_after_planting),
                    app.fertilizer_type,
                    f"{app.amount_kg_per_ha:.1f}",
                    app.application_method.replace('_', ' ').title()
                ])
            
            schedule_table = Table(schedule_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch, 1*inch])
            schedule_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E8B57')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#2E8B57')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F8F8F8')])
            ]))
            
            story.append(schedule_table)
            
            # Application notes
            story.append(Paragraph("Application Notes:", self.styles['SectionHeader']))
            
            for app in application_schedule:
                if app.notes:
                    story.append(Paragraph(f"• <b>{app.stage.title()}:</b> {app.notes}", self.styles['Normal']))
        
        return story
    
    def _create_cost_analysis_section(self, recommendation: Dict) -> List:
        """Create cost analysis section"""
        
        story = []
        story.append(Paragraph("Economic Analysis", self.styles['CustomSubtitle']))
        
        cost_analysis = recommendation.get('cost_analysis', {})
        
        # Cost summary
        cost_summary = f"""
        <b>Cost Summary:</b><br/>
        • Total Fertilizer Cost: {cost_analysis.get('currency', 'USD')} {cost_analysis.get('total_cost', 0):.2f}<br/>
        • Cost per Hectare: {cost_analysis.get('currency', 'USD')} {cost_analysis.get('cost_per_hectare', 0):.2f}<br/>
        • Expected Yield: {recommendation.get('expected_yield', 0):.1f} tons/ha<br/>
        • Return on Investment: {recommendation.get('roi_percentage', 0):.1f}%<br/>
        """
        
        story.append(Paragraph(cost_summary, self.styles['Highlight']))
        
        # Cost breakdown
        fertilizer_breakdown = cost_analysis.get('fertilizer_breakdown', {})
        if fertilizer_breakdown:
            story.append(Paragraph("Cost Breakdown by Fertilizer:", self.styles['SectionHeader']))
            
            breakdown_data = [['Fertilizer Type', 'Cost', 'Percentage']]
            total_cost = cost_analysis.get('total_cost', 1)
            
            for fert_type, cost in fertilizer_breakdown.items():
                percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
                breakdown_data.append([
                    fert_type.replace('_', ' ').title(),
                    f"{cost_analysis.get('currency', 'USD')} {cost:.2f}",
                    f"{percentage:.1f}%"
                ])
            
            breakdown_table = Table(breakdown_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
            breakdown_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E8B57')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#2E8B57')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(breakdown_table)
        
        return story
    
    def _create_regional_considerations_section(self, recommendation: Dict) -> List:
        """Create regional considerations section"""
        
        story = []
        story.append(Paragraph("Regional Considerations", self.styles['CustomSubtitle']))
        
        climate_considerations = recommendation.get('climate_considerations', [])
        if climate_considerations:
            story.append(Paragraph("Climate Considerations:", self.styles['SectionHeader']))
            for consideration in climate_considerations:
                story.append(Paragraph(f"• {consideration}", self.styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Alternative options
        alternative_options = recommendation.get('alternative_options', [])
        if alternative_options:
            story.append(Paragraph("Alternative Options:", self.styles['SectionHeader']))
            for option in alternative_options:
                story.append(Paragraph(f"• {option}", self.styles['Normal']))
        
        return story
    
    def _create_risk_assessment_section(self, recommendation: Dict) -> List:
        """Create risk assessment section"""
        
        story = []
        story.append(Paragraph("Risk Assessment & Management", self.styles['CustomSubtitle']))
        
        risk_factors = recommendation.get('risk_factors', [])
        if risk_factors:
            story.append(Paragraph("Identified Risk Factors:", self.styles['SectionHeader']))
            for risk in risk_factors:
                story.append(Paragraph(f"⚠ {risk}", self.styles['Warning']))
        
        return story
    
    def _create_appendices(self, recommendation: Dict) -> List:
        """Create appendices"""
        
        story = []
        story.append(PageBreak())
        story.append(Paragraph("Appendices", self.styles['CustomSubtitle']))
        
        # Methodology
        story.append(Paragraph("A. Methodology", self.styles['SectionHeader']))
        methodology_text = """
        This fertilizer recommendation was generated using the Soil Test Crop Response (STCR) approach, 
        which considers soil nutrient availability, crop nutrient requirements, and fertilizer efficiency 
        factors. The recommendations are based on established agronomic principles and regional agricultural 
        research data from institutions including FAO, ESDAC, and ICAR/ICRISAT.
        """
        story.append(Paragraph(methodology_text, self.styles['Normal']))
        
        # Glossary
        story.append(Paragraph("B. Glossary", self.styles['SectionHeader']))
        glossary_terms = [
            ('CEC', 'Cation Exchange Capacity - soil\'s ability to hold nutrients'),
            ('NPK', 'Nitrogen, Phosphorus, Potassium - primary nutrients'),
            ('STCR', 'Soil Test Crop Response - fertilizer recommendation method'),
            ('ROI', 'Return on Investment - profitability measure'),
            ('ppm', 'Parts per million - concentration unit')
        ]
        
        for term, definition in glossary_terms:
            story.append(Paragraph(f"<b>{term}:</b> {definition}", self.styles['Normal']))
        
        return story
    
    def _get_ph_status(self, ph: float) -> str:
        """Get pH status classification"""
        if ph < 5.5:
            return "Acidic"
        elif ph > 7.5:
            return "Alkaline"
        else:
            return "Optimal"
    
    def _get_om_status(self, om: float) -> str:
        """Get organic matter status"""
        if om < 2.0:
            return "Low"
        elif om > 4.0:
            return "High"
        else:
            return "Medium"
    
    def _get_nutrient_status(self, value: float, nutrient: str) -> str:
        """Get nutrient status classification"""
        thresholds = {
            'N': {'low': 200, 'high': 400},
            'P': {'low': 10, 'high': 50},
            'K': {'low': 100, 'high': 400}
        }
        
        thresh = thresholds.get(nutrient, {'low': 0, 'high': 1000})
        
        if value < thresh['low']:
            return "Low"
        elif value > thresh['high']:
            return "High"
        else:
            return "Medium"
    
    def _get_cec_status(self, cec: float) -> str:
        """Get CEC status classification"""
        if cec < 10:
            return "Low"
        elif cec > 25:
            return "High"
        else:
            return "Medium"
    
    def _generate_soil_interpretation(self, soil_analysis) -> str:
        """Generate soil interpretation text"""
        
        ph = getattr(soil_analysis, 'ph', 0)
        om = getattr(soil_analysis, 'organic_matter', 0)
        
        interpretation = f"""
        The soil analysis indicates a pH of {ph:.1f}, which is {self._get_ph_status(ph).lower()}. 
        Organic matter content is {om:.1f}%, classified as {self._get_om_status(om).lower()}. 
        """
        
        if ph < 5.5:
            interpretation += "The acidic soil conditions may limit nutrient availability and require lime application. "
        elif ph > 7.5:
            interpretation += "The alkaline conditions may reduce micronutrient availability. "
        
        if om < 2.0:
            interpretation += "Low organic matter suggests need for organic amendments to improve soil structure and nutrient retention."
        
        return interpretation
