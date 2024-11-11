import os
import leafmap
from samgeo import SamGeo, tms_to_geotiff, get_basemaps

class SAM(object):
    def __init__(self, model_type='vit_h',
                 checkpoint='sam_vit_h_4b8939.pth', 
                 sam_kwargs=None,
                 root='',
                 graficar=False):
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.sam_kwargs = sam_kwargs
        self.root=root
        self.graficar=graficar

    def get_tiles(self):
        files=os.listdir(self.root)
        files=[f for f in files if (f.endswith('.tif')) & ('tile_' in f) ]
        return files[111:]

    def plots(self,image):
        m = leafmap.Map(center=[29.676840, -95.369222], zoom=19)
        m.add_basemap("SATELLITE")
        m.layers[-1].visible = False
        m.add_raster(image, layer_name="Image")
        m

    def iterate(self):
        files=self.get_tiles()
        for file in files:
            image=os.path.join(self.root, file)

            sam = SamGeo(
                model_type=self.model_type,
                checkpoint=self.checkpoint,
                sam_kwargs=self.sam_kwargs,
            )
            mask = "segment.tif"
            sam.generate(
                image, mask, batch=True, foreground=False, erosion_kernel=(4, 4), mask_multiplier=255
            )

            vector = image.split('/')[-1]+".gpkg"
            sam.tiff_to_gpkg(mask, vector, simplify_tolerance=None)

            if self.graficar:
                self.plots(image)

def main():
    path=os.path.join('..','data','tiles')
    sam = SAM(model_type='vit_h',
              checkpoint='sam_vit_h_4b8939.pth', 
              sam_kwargs=None,
              root=path)
    sam.iterate()

if __name__ == "__main__": 
    main()