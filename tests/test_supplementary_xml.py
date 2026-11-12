"""In-memory XML/schema checks only; this test does not author XLSX files."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

PATH=Path(__file__).resolve().parents[1]/'local/supplementary.py'
SPEC=importlib.util.spec_from_file_location('supplementary_xml_module',PATH)
writer=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(writer)


class SupplementaryXmlTests(unittest.TestCase):
    def test_exact_types_and_formula_like_strings_are_preserved(self):
        raw={'title':'Tables','sheets':[{'sheet':'Values','columns':['id','n','missing','flag','fraction','literal'],
            'column_types':['text','number','number','boolean','text','text'],
            'rows':[['00102',0,None,False,'-1/48','=2+2']]}]}
        sheet=writer.normalize_payload(raw)['sheets'][0]
        xml=b''.join(writer.worksheet_chunks(sheet));root=ET.fromstring(xml)
        ns={'s':writer.NS};cells={c.attrib['r']:c for c in root.findall('.//s:c',ns)}
        self.assertEqual(cells['A2'].attrib['t'],'inlineStr')
        self.assertEqual(cells['A2'].find('.//s:t',ns).text,'00102')
        self.assertEqual(cells['B2'].find('s:v',ns).text,'0')
        self.assertNotIn('C2',cells)
        self.assertEqual(cells['D2'].attrib['t'],'b')
        self.assertEqual(cells['D2'].find('s:v',ns).text,'0')
        self.assertEqual(cells['E2'].find('.//s:t',ns).text,'-1/48')
        self.assertEqual(cells['F2'].find('.//s:t',ns).text,'=2+2')
        self.assertEqual(root.findall('.//s:f',ns),[])
        self.assertEqual(raw['sheets'][0]['rows'][0][2],None)

    def test_xml_package_relationships_and_escaped_labels(self):
        data=writer.normalize_payload({'A & B':{'columns':['<value>'],'rows':[['< & > "']],'column_types':['text']}})
        parts=writer.package_parts(data,'2026-10-05T00:00:00Z')
        for name,content in parts.items():
            with self.subTest(name=name):ET.fromstring(content)
        parsed=ET.fromstring(b''.join(writer.worksheet_chunks(data['sheets'][0])))
        ns={'s':writer.NS}
        self.assertEqual(parsed.find('.//s:c[@r="A2"]/s:is/s:t',ns).text,'< & > "')
        self.assertNotIn(b'<f>',b''.join(parts.values()))

    def test_excel_limits_invalid_xml_and_numeric_status_are_rejected(self):
        self.assertEqual(writer.column_name(16384),'XFD')
        for value in (0,16385):
            with self.assertRaises(ValueError):writer.column_name(value)
        for text in ('bad\x00text','bad\ud800','x'*32768):
            with self.assertRaises(ValueError):writer.text_value(text)
        for value in (float('nan'),float('inf'),10**15):
            with self.assertRaises(ValueError):writer.typed_value(value,'number')
        with self.assertRaises(TypeError):writer.typed_value('NA','number')
        self.assertEqual(writer.typed_value('NA','text'),'NA')
        self.assertEqual(writer.typed_value('','text'),'')
        self.assertEqual(writer.typed_value('00001','auto'),'00001')

    def test_companion_preparation_preserves_values_without_excel_row_rules(self):
        fields,rows=writer.companion_records({'columns':['id','quantity','status'],
            'column_types':['text','number','text'],'rows':[['00001',0,'observed'],['00002',None,'NA']]})
        self.assertEqual(fields,['id','quantity','status'])
        self.assertEqual(rows,[['00001',0,'observed'],['00002',None,'NA']])
        with self.assertRaises(TypeError):
            writer.companion_records({'columns':['quantity'],'column_types':['number'],'rows':[['...']]})

    def test_bad_sheet_shape_names_and_duplicate_keys_stop(self):
        for sheets in ([{'sheet':'bad/name','columns':['x'],'rows':[]}],
                       [{'sheet':'x','columns':['x','x'],'rows':[]}],
                       [{'sheet':'x','columns':['x'],'rows':[[1,2]]}],
                       [{'sheet':'x','columns':['x'],'rows':[{'unknown':1}]}],
                       [{'sheet':'x','columns':['x'],'rows':[]},{'sheet':'X','columns':['x'],'rows':[]}],
                       [{'sheet':'x','columns':['x'],'rows':[[]]*writer.MAX_ROWS}]):
            with self.assertRaises(ValueError):writer.normalize_payload({'sheets':sheets})

if __name__=='__main__':unittest.main()
