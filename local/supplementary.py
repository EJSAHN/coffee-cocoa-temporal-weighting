"""Export already-computed, typed supplementary tables on the user's computer.

Only constant values are stored. This module does not fit models, alter missing
values, infer CSV column types, or create formulas or external links.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

MAX_ROWS=1048576
MAX_COLUMNS=16384
MAX_TEXT_UNITS=32767
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG_REL='http://schemas.openxmlformats.org/package/2006/relationships'
XML_DECL=b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def text_value(value):
    if not isinstance(value,str):raise TypeError('Text values must be strings')
    if any(not (c in '\t\n\r' or '\x20'<=c<='\ud7ff' or '\ue000'<=c<='\ufffd' or '\U00010000'<=c<='\U0010ffff') for c in value):
        raise ValueError('Text contains an XML 1.0 control or invalid Unicode character')
    if len(value.encode('utf-16-le'))//2>MAX_TEXT_UNITS:raise ValueError('Cell text exceeds the Excel text limit')
    return value


def column_name(index):
    if not isinstance(index,int) or isinstance(index,bool) or not 1<=index<=MAX_COLUMNS:
        raise ValueError('Column index must be between 1 and 16384')
    name=''
    while index:
        index,remainder=divmod(index-1,26);name=chr(65+remainder)+name
    return name


def typed_value(value,kind='auto'):
    if kind not in ('auto','text','number','boolean'):raise ValueError('Unsupported column type: '+str(kind))
    if value is None:return None
    if kind=='text':return text_value(value if isinstance(value,str) else str(value))
    if kind=='boolean':
        if not isinstance(value,bool):raise TypeError('Boolean column requires bool or None')
        return value
    if kind=='number' and (isinstance(value,bool) or not isinstance(value,(int,float))):
        raise TypeError('Numeric columns require typed finite numbers or None; conversion belongs in the table collector')
    if isinstance(value,str):return text_value(value)
    if isinstance(value,bool):return value
    if isinstance(value,int):
        if abs(value)>=10**15:raise ValueError('Large integer must be supplied as text to preserve exact digits')
        return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Nonfinite numeric cell; use None with a separate status field')
        return value
    raise TypeError('Unsupported cell value type: '+type(value).__name__)


def normalize_payload(payload):
    if not isinstance(payload,dict):raise TypeError('Workbook payload must be a mapping')
    if 'sheets' in payload:
        sheets=payload['sheets'];title=payload.get('title','Supplementary tables')
    else:
        sheets=[dict(spec,sheet=name) for name,spec in payload.items()];title='Supplementary tables'
    if not isinstance(sheets,list) or not sheets:raise ValueError('At least one sheet is required')
    result=[];seen=set()
    for spec in sheets:
        name=text_value(spec['sheet'])
        if not name or len(name.encode('utf-16-le'))//2>31 or any(c in name for c in '[]:*?/\\') or name.startswith("'") or name.endswith("'"):
            raise ValueError('Invalid Excel sheet name: '+repr(name))
        if name.casefold() in seen:raise ValueError('Duplicate sheet name ignoring case: '+name)
        seen.add(name.casefold())
        columns=spec['columns'];rows=spec['rows']
        if not isinstance(columns,list) or not 1<=len(columns)<=MAX_COLUMNS:raise ValueError('Invalid column count')
        if any(not isinstance(c,str) or not c for c in columns) or len(set(columns))!=len(columns):raise ValueError('Headers must be unique nonempty strings')
        for c in columns:text_value(c)
        if not isinstance(rows,list) or len(rows)+1>MAX_ROWS:raise ValueError('Row count exceeds the Excel limit including the header')
        kinds=spec.get('column_types',['auto']*len(columns))
        if isinstance(kinds,dict):
            if set(kinds)-set(columns):raise ValueError('Types specify unknown columns')
            kinds=[kinds.get(c,'auto') for c in columns]
        if not isinstance(kinds,list) or len(kinds)!=len(columns):raise ValueError('column_types must align with columns')
        for kind in kinds:
            if kind not in ('auto','text','number','boolean'):raise ValueError('Unknown column type')
        normalized=[]
        for row in rows:
            if isinstance(row,dict):
                if set(row)-set(columns):raise ValueError('Row contains an unknown column')
                row=[row.get(c) for c in columns]
            if not isinstance(row,(list,tuple)) or len(row)!=len(columns):raise ValueError('Row width differs from header width')
            normalized.append([typed_value(value,kind) for value,kind in zip(row,kinds)])
        result.append(dict(sheet=name,columns=columns,rows=normalized,column_types=kinds))
    return dict(title=text_value(str(title)),sheets=result)


def cell_xml(address,value,header=False):
    if value is None:return b''
    attrs={'r':address,'s':'1' if header else ('2' if isinstance(value,str) else '0')}
    cell=ET.Element('c',attrs)
    if isinstance(value,str):
        cell.set('t','inlineStr');node=ET.SubElement(ET.SubElement(cell,'is'),'t',{'{http://www.w3.org/XML/1998/namespace}space':'preserve'});node.text=text_value(value)
    elif isinstance(value,bool):
        cell.set('t','b');ET.SubElement(cell,'v').text='1' if value else '0'
    else:
        number=typed_value(value,'number');ET.SubElement(cell,'v').text=str(number)
    return ET.tostring(cell,encoding='utf-8')


def worksheet_chunks(sheet):
    last=column_name(len(sheet['columns']))+str(len(sheet['rows'])+1)
    yield XML_DECL+('<worksheet xmlns="'+NS+'"><dimension ref="A1:'+last+'"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/><cols>').encode()
    for i,name in enumerate(sheet['columns'],1):
        width=min(48,max(14,len(name)+2));yield ('<col min="%d" max="%d" width="%d" customWidth="1"/>'%(i,i,width)).encode()
    yield b'</cols><sheetData>'
    for r,row in enumerate([sheet['columns'],*sheet['rows']],1):
        yield ('<row r="%d"%s>'%(r,' ht="32" customHeight="1"' if r==1 else '')).encode()
        for c,value in enumerate(row,1):yield cell_xml(column_name(c)+str(r),value,header=r==1)
        yield b'</row>'
    yield ('</sheetData><autoFilter ref="A1:'+last+'"/><pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/></worksheet>').encode()


def xml_bytes(element):return XML_DECL+ET.tostring(element,encoding='utf-8')


def package_parts(payload,created):
    """Return small package metadata parts; worksheets are streamed separately."""
    content=ET.Element('Types',xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    for ext,mime in [('rels','application/vnd.openxmlformats-package.relationships+xml'),('xml','application/xml')]:ET.SubElement(content,'Default',Extension=ext,ContentType=mime)
    overrides=[('/xl/workbook.xml','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'),('/xl/styles.xml','application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml'),('/docProps/core.xml','application/vnd.openxmlformats-package.core-properties+xml')]
    overrides += [('/xl/worksheets/sheet%d.xml'%i,'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml') for i in range(1,len(payload['sheets'])+1)]
    for name,mime in overrides:ET.SubElement(content,'Override',PartName=name,ContentType=mime)
    rels=ET.Element('Relationships',xmlns=PKG_REL)
    ET.SubElement(rels,'Relationship',Id='rId1',Type=REL+'/officeDocument',Target='xl/workbook.xml')
    ET.SubElement(rels,'Relationship',Id='rId2',Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties',Target='docProps/core.xml')
    book=ET.Element('workbook',xmlns=NS);sheets=ET.SubElement(book,'sheets');bookrels=ET.Element('Relationships',xmlns=PKG_REL)
    for i,sheet in enumerate(payload['sheets'],1):
        ET.SubElement(sheets,'sheet',name=sheet['sheet'],sheetId=str(i),attrib={'{'+REL+'}id':'rId'+str(i)})
        ET.SubElement(bookrels,'Relationship',Id='rId'+str(i),Type=REL+'/worksheet',Target='worksheets/sheet%d.xml'%i)
    ET.SubElement(bookrels,'Relationship',Id='rIdStyles',Type=REL+'/styles',Target='styles.xml')
    core=ET.Element('{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}coreProperties')
    ET.SubElement(core,'{http://purl.org/dc/elements/1.1/}title').text=payload['title']
    ET.SubElement(core,'{http://purl.org/dcterms/}created',{'{http://www.w3.org/2001/XMLSchema-instance}type':'dcterms:W3CDTF'}).text=created
    core.set('xmlns:dcterms','http://purl.org/dcterms/')
    styles=('<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="'+NS+'"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="49" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>').encode()
    return {'[Content_Types].xml':xml_bytes(content),'_rels/.rels':xml_bytes(rels),'xl/workbook.xml':xml_bytes(book),'xl/_rels/workbook.xml.rels':xml_bytes(bookrels),'xl/styles.xml':styles,'docProps/core.xml':xml_bytes(core)}


def export_workbook(tables,out_path):
    """Write one new XLSX atomically after complete input validation.

    Accepts a payload with ``sheets`` or a mapping of sheet name to table specs.
    Existing files are not overwritten. Empty numeric values must be None.
    The CSV/JSON source remains authoritative for numeric precision.
    """
    payload=normalize_payload(tables);destination=Path(out_path)
    if destination.suffix.lower()!='.xlsx':raise ValueError('Output must have .xlsx extension')
    if destination.exists():raise FileExistsError('Refusing to overwrite workbook: '+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    created=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
    handle,tmp=tempfile.mkstemp(prefix=destination.name+'.',suffix='.writing',dir=str(destination.parent));os.close(handle)
    temporary=Path(tmp)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
            for name,data in package_parts(payload,created).items():archive.writestr(name,data)
            for index,sheet in enumerate(payload['sheets'],1):
                with archive.open('xl/worksheets/sheet%d.xml'%index,'w') as stream:
                    for chunk in worksheet_chunks(sheet):stream.write(chunk)
        with zipfile.ZipFile(temporary) as archive:
            bad=archive.testzip()
            if bad:raise OSError('XLSX member CRC failure: '+bad)
        if destination.exists():raise FileExistsError('Workbook appeared during export: '+str(destination))
        os.replace(temporary,destination)
    finally:
        if temporary.exists():temporary.unlink()
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    return dict(path=str(destination),sha256=digest,created_utc=created,sheets=len(payload['sheets']),
                rows=sum(len(s['rows']) for s in payload['sheets']),formulas_written=0,
                source_values_recalculated=False)


def companion_records(spec):
    """Validate one companion table while preserving typed values and blanks."""
    columns=spec['columns'];rows=spec['rows']
    if not isinstance(columns,list) or not columns or any(not isinstance(c,str) or not c for c in columns) or len(set(columns))!=len(columns):
        raise ValueError('Companion headers must be unique nonempty strings')
    kinds=spec.get('column_types',['auto']*len(columns))
    if isinstance(kinds,dict):
        if set(kinds)-set(columns):raise ValueError('Types specify unknown companion columns')
        kinds=[kinds.get(c,'auto') for c in columns]
    if not isinstance(kinds,list) or len(kinds)!=len(columns):raise ValueError('Companion types differ from column count')
    if not isinstance(rows,list):raise TypeError('Companion rows must be a list')
    answer=[]
    for row in rows:
        if isinstance(row,dict):
            if set(row)-set(columns):raise ValueError('Companion row has an unknown column')
            row=[row.get(c) for c in columns]
        if not isinstance(row,(list,tuple)) or len(row)!=len(columns):raise ValueError('Companion row width differs')
        answer.append([typed_value(v,k) for v,k in zip(row,kinds)])
    return columns,answer


def export_companions(payload,out_dir):
    """Export indexed CSV.gz companions; does not create or modify a workbook."""
    import csv
    import gzip
    import io
    specs=payload.get('companions',[])
    if not isinstance(specs,list):raise TypeError('companions must be a list')
    destination=Path(out_dir);prepared=[];seen=set()
    for spec in specs:
        filename=spec['filename']
        if not isinstance(filename,str) or not filename.endswith('.csv.gz') or '/' in filename or '\\' in filename or ':' in filename or '\x00' in filename or filename in ('.','..'):
            raise ValueError('Companion filename must be a plain .csv.gz filename')
        if filename.casefold() in seen:raise ValueError('Duplicate companion filename')
        seen.add(filename.casefold())
        path=destination/filename
        if path.exists():raise FileExistsError('Refusing to overwrite companion: '+str(path))
        columns,rows=companion_records(spec)
        prepared.append((spec,path,columns,rows))
    destination.mkdir(parents=True,exist_ok=True);outputs=[]
    for spec,path,columns,rows in prepared:
        handle,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.writing',dir=str(destination));os.close(handle)
        temporary=Path(tmp)
        try:
            with temporary.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as compressed:
                    with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as stream:
                        writer=csv.writer(stream,lineterminator='\n');writer.writerow(columns);writer.writerows(rows)
            if path.exists():raise FileExistsError('Companion appeared during export: '+str(path))
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()
        outputs.append(dict(key=spec.get('key',path.name),filename=path.name,path=str(path),rows=len(rows),
                            columns=len(columns),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return outputs
