import sys
sys.path.insert(0, '.')
from inference import run_inference, generate_demo_mri
from report_generator import generate_pdf_report

demo_bytes = generate_demo_mri()
result = run_inference(demo_bytes)
pdf_bytes = generate_pdf_report(result)
print(f'PDF generated: {len(pdf_bytes):,} bytes')

outpath = '/tmp/test_report.pdf'
with open(outpath, 'wb') as f:
    f.write(pdf_bytes)
print(f'Saved to {outpath}')
print('PDF TEST PASSED')
