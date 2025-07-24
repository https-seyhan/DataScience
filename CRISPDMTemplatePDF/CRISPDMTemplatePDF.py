from fpdf import FPDF

class CRISPDMTemplatePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'CRISP-DM Project Template', 0, 1, 'C')
        self.ln(4)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)

    def section_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln(2)

pdf = CRISPDMTemplatePDF()
pdf.add_page()

sections = [
    ("1. Business Understanding", 
     "Objective:\n  (What problem are you solving?)\n\n"
     "Stakeholders:\n  (Who needs the results?)\n\n"
     "Success Criteria:\n  (How will you measure impact?)"),
    
    ("2. Data Understanding",
     "Data Sources:\n  (Where is the data coming from?)\n\n"
     "Data Overview:\n  (What variables exist? How much data?)\n\n"
     "Initial Observations:\n  (Missing data, outliers, inconsistencies?)"),
    
    ("3. Data Preparation",
     "Cleaning Steps:\n  (Null handling, duplicates removal?)\n\n"
     "Feature Engineering:\n  (New columns created?)\n\n"
     "Final Dataset:\n  (Summary of rows, columns, format?)"),
    
    ("4. Modeling",
     "Algorithms Used:\n  (e.g., XGBoost, SVM?)\n\n"
     "Performance Metrics:\n  (AUC, RMSE, Accuracy?)\n\n"
     "Hyperparameters:\n  (What tuning was done?)"),
    
    ("5. Evaluation",
     "Model Performance:\n  (How well does it generalize?)\n\n"
     "Business Implications:\n  (Does it solve the original problem?)\n\n"
     "Recommendations:\n  (Go / No-Go / Improvements?)"),
    
    ("6. Deployment",
     "Output Format:\n  (App, API, dashboard?)\n\n"
     "Monitoring Plan:\n  (Track performance over time?)\n\n"
     "Documentation:\n  (Instructions, limitations, contacts?)")
]

for title, body in sections:
    pdf.section_title(title)
    pdf.section_body(body)

pdf_path = "/mnt/data/CRISP_DM_Project_Template.pdf"
pdf.output(pdf_path)
pdf_path
