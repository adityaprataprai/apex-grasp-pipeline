from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

# Deterministic geometry fixture generator only. It is not a photorealistic replacement for OCID/GraspNet/custom data.
def generate(spec,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    W,H=spec['image_size']; rgb=Image.new('RGB',(W,H),tuple(spec.get('background_rgb',[40,40,40]))); draw=ImageDraw.Draw(rgb);depth=np.full((H,W),float(spec.get('background_depth',3.0)),dtype=np.float32);anns=[]
    for i,o in enumerate(spec['objects'],1):
        x1,y1,x2,y2=o['bbox_xyxy']; color=tuple(o['rgb']);draw.rectangle([x1,y1,x2,y2],fill=color);m=np.zeros((H,W),dtype=np.uint8);m[y1:y2+1,x1:x2+1]=255;mp=out/f"{o.get('id',f'obj_{i:02d}')}_mask.png";Image.fromarray(m).save(mp);depth[y1:y2+1,x1:x2+1]=float(o['depth_m']);anns.append({**o,'mask_path':str(mp)})
    rgb.save(out/'rgb.png');np.save(out/'depth.npy',depth);(out/'annotations.json').write_text(json.dumps({'objects':anns,'instruction':spec.get('instruction'),'intrinsics':spec.get('intrinsics')},indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--output-dir',required=True);a=ap.parse_args();generate(json.loads(Path(a.spec).read_text(encoding='utf-8')),a.output_dir)
if __name__=='__main__':main()
