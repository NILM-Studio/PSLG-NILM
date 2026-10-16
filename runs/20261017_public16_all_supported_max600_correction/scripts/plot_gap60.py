import pathlib,sys,os
import pipeline
pipeline.PLAN['segmentation']['gap_seconds']=60
o=next(o for o in pipeline.PLAN['objects'] if o['id']==sys.argv[1])
pipeline.plot_object(o,pathlib.Path(os.environ['PSLG_EXECUTION_DIR']))
