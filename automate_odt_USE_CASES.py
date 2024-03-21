
import zipfile
import xml.etree.ElementTree as ET
import os

def get_bold_text_from_odt(odt_file):
    print(odt_file)
    bold_text = []

    # Open the ODT file as a zip archive
    with zipfile.ZipFile(odt_file, 'r') as odt_zip:
        # Extract content.xml from the ODT archive
        with odt_zip.open('content.xml') as content_file:
            # Parse content.xml using ElementTree
            tree = ET.parse(content_file)
            root = tree.getroot()

            # Find all text:span elements with a text:style-name attribute containing "emphasis"
            for span in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}span'):
                style_name = span.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}style-name', '')
                print('style_name :', style_name.lower())
                
                if 'strong' in style_name.lower():
                    # Get the text content of the span
                    text = ''.join(span.itertext())
                    bold_text.append(text)

    return bold_text

os.chdir('/home/saul/Desktop/generative-AI/RE/')
print(os.getcwd())
odt_file = 'Gen_AI_Real_estate.odt'
bold_text_in_odt = get_bold_text_from_odt(odt_file)
print(bold_text_in_odt)
