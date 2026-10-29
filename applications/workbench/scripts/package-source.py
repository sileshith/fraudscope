"""Create a reproducible source download without runtime state or Site identity."""
from pathlib import Path
import zipfile,json
root=Path(__file__).resolve().parents[1];out=root/'public/downloads/fraudscope-live-source.zip';out.parent.mkdir(parents=True,exist_ok=True)
excluded={'node_modules','.git','.sites-runtime','.wrangler','.vinext','dist','outputs','__pycache__','.pytest_cache','downloads','.next','.home','.cache','.npm'}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if not p.is_file() or any(part in excluded for part in rel.parts) or p.suffix in {'.pyc','.tsbuildinfo'}:continue
  if str(rel)=='.openai/hosting.json' or p.name.startswith(('.env','.dev.vars')):continue
  z.write(p,'fraudscope-live/'+rel.as_posix())
 z.writestr('fraudscope-live/.openai/hosting.example.json',json.dumps({'d1':'DB','r2':None},indent=2))
print(out)
