import os
from fpdf import FPDF # pyright: ignore[reportMissingModuleSource]
from datetime import datetime
import qrcode # pyright: ignore[reportMissingModuleSource]
from io import BytesIO
import textwrap
import pandas as pd

# 📁 Polices
FONTS_DIR = os.path.join(os.path.dirname(__file__), "../fonts/dejavu-fonts-ttf-2.37/ttf/")
DEJAVU_REGULAR = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
DEJAVU_BOLD = os.path.join(FONTS_DIR, "DejaVuSans-Bold.ttf")

def build_pdf(culture, surface, fertilizer_name):
    class StyledPDF(FPDF):
        def header(self):
            self.set_fill_color(0,102,204)
            self.rect(0,0,self.w,20,'F')
            self.set_font("DejaVu","B",14)
            self.set_text_color(255,255,255)
            self.set_y(6)
            self.cell(0,8,"🧪 Recommandation de fertilisation – SmartFactLaser", align="C")
            self.ln(10)
        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu","",8)
            self.set_text_color(150,150,150)
            self.cell(0,10,"Généré le " + datetime.now().strftime("%d/%m/%Y %H:%M"), 0, 0, "C")

    pdf = StyledPDF()
    if os.path.exists(DEJAVU_REGULAR):
        pdf.add_font("DejaVu","",DEJAVU_REGULAR, uni=True)
        pdf.add_font("DejaVu","B",DEJAVU_BOLD, uni=True)
    pdf.add_page()
    pdf.set_font("DejaVu" if os.path.exists(DEJAVU_REGULAR) else "Arial", "", 12)
    pdf.cell(0,10,f"🌿 Culture : {culture}", ln=True)
    pdf.cell(0,10,f"📐 Surface : {surface} ha", ln=True)
    pdf.cell(0,10,f"🧪 Fertilizer recommandé : {fertilizer_name}", ln=True)
    pdf.ln(5)

    url = f"https://sama-agrolink.com/fertiplan/{culture}"
    qr_img = qrcode.make(url)
    qr_buffer = BytesIO()
    qr_img.save(qr_buffer, format='PNG')
    qr_buffer.seek(0)
    pdf.image(qr_buffer, w=30)
    pdf.set_font("DejaVu" if os.path.exists(DEJAVU_REGULAR) else "Arial","",9)
    pdf.cell(0,10,url, ln=True)

    out = BytesIO()
    out_bytes = pdf.output(dest="S")
    out.write(out_bytes)
    out.seek(0)
    return out

def build_excel(culture, surface, fertilizer_name):
    df_plan = pd.DataFrame([{
        "Culture": culture,
        "Surface (ha)": surface,
        "Fertilizer": fertilizer_name
    }])
    excel_bytes = BytesIO()
    with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
        df_plan.to_excel(writer, index=False, sheet_name="Fertilisation")
    excel_bytes.seek(0)
    return excel_bytes
