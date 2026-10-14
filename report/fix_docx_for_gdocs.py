import sys, re, io, zipfile
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
doc = zin.read('word/document.xml').decode()
rels = zin.read('word/_rels/document.xml.rels').decode()
# rename media to imageN.png
ren = {}
for i, m in enumerate(re.findall(r'Target="media/([^"]+)"', rels), 1):
    ren[m] = f'image{i}.png'
    rels = rels.replace(f'media/{m}', f'media/image{i}.png')
# unique drawing ids
n = [0]
def fix(m):
    n[0] += 1; k = n[0]; d = m.group(0)
    d = re.sub(r'<wp:docPr id="\d+"', f'<wp:docPr id="{k}"', d)
    d = re.sub(r'<pic:cNvPr id="\d+" name="" descr=""/>', f'<pic:cNvPr id="{k}" name="Picture {k}"/>', d)
    return d
doc = re.sub(r'<w:drawing>.*?</w:drawing>', fix, doc, flags=re.S)
zout = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    if it.is_dir(): continue
    name, data = it.filename, zin.read(it.filename)
    if name == 'word/document.xml': data = doc.encode()
    elif name == 'word/_rels/document.xml.rels': data = rels.encode()
    elif name.startswith('word/media/'):
        base = name.split('/')[-1]
        im = Image.open(io.BytesIO(data))
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGBA'); bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[3]); im = bg
        b = io.BytesIO(); im.save(b, 'PNG', optimize=True); data = b.getvalue()
        name = 'word/media/' + ren.get(base, base)
    zout.writestr(name, data)
zout.close(); print('drawings fixed:', n[0], '->', dst)
