# from linknet import LinkNet34, LinkNet34MTL
from iterations.baseline import MultiTaskStackedHourglassBaseline
from iterations.clip import MultiTaskStackedHourglassClip
from iterations.spatial import MultiTaskStackedHourglassSpatial
from iterations.self import MultiTaskStackedHourglassSelfAttention
# from SPIN import spin
# from unet import unet

MODELS = {"baseline": MultiTaskStackedHourglassBaseline, "clip": MultiTaskStackedHourglassClip, "spatial": MultiTaskStackedHourglassSpatial, "self": MultiTaskStackedHourglassSelfAttention}