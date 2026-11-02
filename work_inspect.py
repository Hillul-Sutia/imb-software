from pathlib import Path
import urllib.request,zipfile
p=Path('work');p.mkdir(exist_ok=True)
url='https://files.pythonhosted.org/packages/'
import json
j=json.load(urllib.request.urlopen('https://pypi.org/pypi/mtune/0.0.8/json',timeout=30))
u=next(x['url'] for x in j['urls'] if x['filename'].endswith('.whl'))
f=p/'mtune.whl';urllib.request.urlretrieve(u,f)
with zipfile.ZipFile(f) as z:
 for n in z.namelist():
  if n.endswith('.py'):
   print(n);print(z.read(n).decode())
