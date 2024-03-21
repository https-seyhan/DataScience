
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import os
import matplotlib.pyplot as plt

class RE:
	def __init__(self):
		self.bold_text = []
		os.chdir('/home/saul/Desktop/generative-AI/RE/')
		print(os.getcwd())
		self.__merge_files()
		
	def __merge_files(self):
		os.system('ooo_cat USE_CASE_*.odt -o ALL_USE_CASES_test3.odt')
		
	def get_bold_text_from_odt(self, odt_file):
		print(odt_file)
		#bold_text = []

		# Open the ODT file as a zip archive
		with zipfile.ZipFile(odt_file, 'r') as odt_zip:
			#print('ODT_ZIP ', odt_zip)
			# Extract content.xml from the ODT archive
			with odt_zip.open('content.xml') as content_file:
				print(content_file)
				# Parse content.xml using ElementTree
				tree = ET.parse(content_file)
				print('Tree ', tree)
				root = tree.getroot()
				print('Loc ', root[1].text)
				print('Root Tag ', root.tag)
				#print('Root Attrib ', root.attrib)
				#for body in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}body'):
				for child in root:
					print(child.tag, child.attrib)
					for body in child.iter('{urn:oasis:names:tc:opendocument:xmlns:office:1.0}body'):
						#print('BODY ', body)
						print(body.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:office:1.0}body', ''))

				for subject in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}subject'):
					suffix = span.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}suffix', '')
					print('suffix ', suffix)
					
				for tab in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}tab'):
					tab = span.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}tab-ref', '')
					print('tab ', tab)
					
				# Find all text:span elements with a text:style-name attribute containing "emphasis"
				for span in root.iter('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}span'):
					#print('span :', span)
					#print('root :', root)
					style_name = span.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:text:1.0}style-name', '')
					#unknown = root.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:office:1.0}document-content')
					#print('unkown :', unknown)
					#print('style_name :', style_name.lower())
                
					if 'strong' in style_name.lower():
						# Get the text content of the span
						
						text = ''.join(span.itertext())
						#print('Text :', text)
						self.bold_text.append(text)
		print(self.bold_text)
		#subject_freqs = self.__get_subject_freq(self.bold_text)
		
		return self.bold_text
		
	def __get_subject_freq(self, topics):

		frequency = {}

		# iterating over the list
		for item in topics:
			# checking the element in dictionary
			if item in frequency:
				# incrementing the counr
				frequency[item] += 1
			else:
				# initializing the count
				frequency[item] = 1
		#sort frequencies
		sorted_freq = sorted(frequency.items(), key=lambda x:x[1], reverse=True)
		sorted_freq  = dict(sorted_freq )
		#self.__convert_dict_to_dataframe(sorted_freq)

		#printing the frequency
		#print(sorted_freq)
		return sorted_freq
		
	def __convert_dict_to_dataframe(self, freq):
		#print(freq)
		freq_dataframe = pd.DataFrame(list(freq.items()))
		freq_dataframe.columns =['Name', 'Freq']
		print(freq_dataframe.head())
		#freq_dataframe.to_csv('subject_freq.csv', index=False)
		#freq_dataframe.to_csv('cb_freq.csv', index=False)

	#os.chdir('/home/saul/Desktop/generative-AI/RE/')
#
#odt_file = 'Gen_AI_Real_estate.odt'
#odt_file = 'use_of_cb_in_re_AU.odt'
odt_file = 'USE_CASE_1.odt'
re = RE()
strong_in_odt = re.get_bold_text_from_odt(odt_file)
#print(strong_in_odt)
