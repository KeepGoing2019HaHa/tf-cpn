from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


eval_gt = COCO('/datanew/yw/cpn/tf-cpn/data/COCO/MSCOCO/annotations/person_keypoints_minival2014.json')
eval_dt = eval_gt.loadRes('/datanew/yw/cpn/tf-cpn/models/COCO.res50.256x192.CPN/log/results.json')
# eval_dt = eval_gt.loadRes('/datanew/yw/cpn/tf-cpn/models/COCO.res50.256x192.CPN/log/results-remove.json')
cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')


cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()