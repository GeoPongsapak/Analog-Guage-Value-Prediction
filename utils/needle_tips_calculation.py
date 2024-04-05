from os import getcwd
import sys
sys.path.append(getcwd())
from config.libaries import *

def distance(xy, ct):
    x1,y1 = xy
    x2,y2 = ct
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

def needle_tips_point_detect(img, model, middle):
    coordinates = []
    results = model.predict(img, 0.3, retina_masks=True)
    for result in results:
        masks = result.masks.xy
        for mask in masks[0]:
            coordinates.append((mask[0].astype(int), mask[1].astype(int)))

        max_temp = 0
        tip = (0,0)
        dist = 0
        for i in coordinates:
            
            dist = distance((i[0], i[1]), middle)
            if dist > max_temp:
                max_temp = dist
                tip = (i[0], i[1])

        return [tip[0], tip[1]]


def TND_needle_tips_point_detect(img, model, middle):
    # draw = ImageDraw.Draw(self.img)
    coordinates = []
    tip = []
    results = model.predict(img,conf=0.3,save=False,retina_masks=True)

    for result in results:
        names = result.names
        masks = result.masks.xy
        name = []

        # im_array = result.plot() 
        # im = Image.fromarray(im_array[..., ::-1]).convert('RGB')
        # plt.imshow(im)
        # plt.show()

        for num in result.boxes.cls:
            name.append(names[int(num)])

        for idx,mask in enumerate(masks):
            
            for m in mask:                 
                coordinates.append((m[0].astype(int), m[1].astype(int)))
            max_temp = 0
            dist = 0
            for i in coordinates:
                dist = distance((i[0], i[1]), middle)
                if dist > max_temp:
                    max_temp = dist
                    coor_temp = (i[0], i[1])
            tip.append(coor_temp)
            coordinates = []

        # self.d = tip[name.index('value-needle')]
        tips = []
        for idx, test in enumerate(name):
            if test == 'value-needle':
                if distance(tip[idx], middle) > distance((middle[0], middle[1]-500), middle):
                    pass
                else:
                    tips.append(tip[idx])
        
        print(tips)
        return tips[0]
    
def WNR_needle_tips_point_detect(img, model, middle):
    # draw = ImageDraw.Draw(self.img)
    coordinates = []
    tip = []
    results = model.predict(img,conf=0.3,save=False,retina_masks=True)

    for result in results:
        names = result.names
        masks = result.masks.xy
        name = []
        for num in result.boxes.cls:
            name.append(names[int(num)])

        for idx,mask in enumerate(masks):
            # draw.polygon(mask,outline=color[idx], width=5)
            
            for m in mask:                 
                coordinates.append((m[0].astype(int), m[1].astype(int)))
            max_temp = 0
            dist = 0
            for i in coordinates:
                dist = distance((i[0], i[1]), middle)
                if dist > max_temp:
                    max_temp = dist
                    coor_temp = (i[0], i[1])
            tip.append(coor_temp)
            coordinates = []
        if 'needle_w' not in name:
            return tip[name.index('needle_r')]
        else:
            return tip[name.index('needle_w')]