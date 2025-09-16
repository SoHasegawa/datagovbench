import json
import fitz
import docx
import pandas as pd

from trafilatura import html2txt, extract, fetch_url
from pathlib import Path
from lxml import etree

from src.agent.sequence import to_sequence_descriptive


class ExternalKnowledgeOpener:
    def __init__(self):
        pass

    def _open_json(self, df, json_file):
        knowledge = json.load(open(json_file))
        if isinstance(knowledge, list): return None
        meta = knowledge.get("meta")
        if meta is None: return None
        knowledge = knowledge["meta"]["view"]["columns"]
        columns = df.columns

        external_knowledge = {}

        for k in knowledge:
            if k["name"] in columns:
                column_name = k["name"]
                description = k.get("description")
                if description == "": continue
                if description is None: return None

                external_knowledge[column_name] = description

        return external_knowledge
    
    def _open_pdf(self, df, pdf_file):
        try:
            pdf_document = fitz.open(pdf_file)
        except Exception as e:
            return None
        num_pages = pdf_document.page_count

        table_sequence = ""
        cache = {}

        for index in range(num_pages):
            page = pdf_document[index]
            tables = page.find_tables()
            if tables:
                try:
                    table = tables[0]
                    df = table.to_pandas()
                    if index == 0:
                        columns = df.columns
                        columns = [str(c) for c in columns if not c is None or c == ""]
                        columns = "|".join(list(columns))
                        table_sequence += columns + "\n"
                    values = df.values.tolist()
                    for value in values:
                        value = [str(v).replace("\n", ",") for v in value if not v is None or v == ""]
                        value = "|".join(list(value))
                        cache_hit = cache.get(value)
                        if cache_hit is None: cache[value] = True
                        else: continue
                        table_sequence += value + "\n"
                except Exception as e:
                    continue

        return table_sequence

    @staticmethod
    def _to_sequence(df):
        if len(df.index) < 300:
            sequence = ""
            cache = {}
            columns = df.columns
            columns = [str(c) for c in columns if not c is None or c == ""]
            columns = "|".join(list(columns))
            sequence += columns + "\n"
            values = df.values.tolist()
            for value in values:
                value = [str(v).replace("\n", ",") for v in value if not v is None or v == ""]
                value = "|".join(list(value))
                cache_hit = cache.get(value)
                if cache_hit is None: cache[value] = True
                else: continue
                sequence += value + "\n"
        else:
            sequence = to_sequence_descriptive(df)

        return sequence
    
    def _open_csv(self, df, csv_file):
        try:
            knowledge = pd.read_csv(csv_file)
            return self._to_sequence(knowledge)
        except Exception as e:
            return None
    
    def _open_xml(self, df, xml_file):
        try:
            knowledge = extract_texts_with_linebreaks_any_tag(xml_file)
            return knowledge
        except Exception as e:
            return None
        
    def _open_html(self, df, html_file):
        try:
            knowledge = extract(open(html_file).read())
            return knowledge
        except Exception as e:
            return None
    
    def _open_xlsx(self, df, xlsx_file):
        try:
            knowledge = pd.read_excel(xlsx_file, engine='openpyxl')
        except Exception as e:
            print(e)
            return None

        if not "Variable_name/Variable_Nom" in knowledge.columns: return self._to_sequence(knowledge)

        column_name = knowledge["Variable_name/Variable_Nom"].values.tolist()
        column_label = knowledge["Variable_Label/Étiquette_Variable"].values.tolist()
        description = knowledge["Definition/Définition\n"].values.tolist()
        values = knowledge["Values/Valeurs"].values.tolist()
        invalid_values = knowledge["Invalid_values/Valeurs invalides"].values.tolist()

        external_knowledge = {}

        for cn, cl, des, v, iv in zip(column_name, column_label, description, values, invalid_values):
            explanation = {}
            explanation["label"] = cl
            explanation["description"] = des
            explanation["values"] = v
            explanation["invalid values"] = iv

            external_knowledge[cn] = explanation

        return external_knowledge
    
    def _open_docx(self, df, doc_file):
        doc = docx.Document(doc_file)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)

        return '\n'.join(fullText)
    
    def _open_text(self, df, text_file):
        try:
            with open(text_file, "r") as f:
                text = f.read()
            return text
        except Exception as e:
            return None
    
    def __call__(self, df, file):
        print(file)
        external_knowledges = []
        if file.endswith("json"):
            external_knowledge = self._open_json(df, file)
        elif file.endswith("csv"):
            external_knowledge = self._open_csv(df, file)
        elif file.endswith("xlsx") or file.endswith("xls"):
            external_knowledge = self._open_xlsx(df, file)
        elif file.endswith("pdf"):
            external_knowledge = self._open_pdf(df, file)
        elif file.endswith("docx") or file.endswith("doc"):
            external_knowledge = self._open_docx(df, file)
        elif file.endswith("txt"):
            external_knowledge = self._open_text(df, file)
        elif file.endswith("xml"):
            external_knowledge = self._open_xml(df, file)
        elif file.endswith("html"):
            external_knowledge = self._open_html(df, file)
        else:
            external_knowledge = None
        external_knowledges.append(external_knowledge)

        return external_knowledges
    

def extract_texts_with_linebreaks_any_tag(xml_path):
    with open(xml_path, 'rb') as f:
        tree = etree.parse(f)
    root = tree.getroot()

    def recursive_extract(elem):
        texts = []

        # Text before children
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())

        # Process children
        for child in elem:
            texts.extend(recursive_extract(child))

        # Line break after any element
        texts.append('\n')

        # Tail text after element
        if elem.tail and elem.tail.strip():
            texts.append(elem.tail.strip())

        return texts

    all_text = ''.join(recursive_extract(root))

    # Clean up: remove multiple newlines and strip whitespace
    lines = [line.strip() for line in all_text.split('\n') if line.strip()]
    return '\n'.join(lines)
