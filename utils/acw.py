from os import getcwd
import sys
sys.path.append(getcwd())
from config.configuration import GENERAL_CONFIG,ACW_MODEL_CONFIG
from config.libaries import *
import time

class ACWValuePrediction:

    def __init__(self, end_value : float, file_name : str = None, frame :np.ndarray = None, start_value : float = 0, conf : float = 0.3) -> None:
        self.end_value = end_value
        self.start_value = start_value
        self.conf = conf
        self.model = ACW_MODEL_CONFIG.MODEL
        self.angleb = -120
        self.anglec = 120
        self.error_state = True
        
        try:
            if file_name != None:
                self.img = Image.open(file_name)
            else:
                self.img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:
            print('Image not found!!')
            return None

        self.find_point()
        if not self.error_state:
            return None
        
        # try:
        #     self.needle_tips_point_detect()
        # except:
        #     self.predicted_value = 'Not found needle tips'
        #     return None
        
        if self.d[1] > self.b[1]:
            self.predicted_value = 0
        else:
            self.predict_value()
        # self.draw_img()


    def find_point(self):
        # ============= Detect and get coordinate =============
        r = self.model(self.img, conf=self.conf)
        for rt in r:
            name = rt.names
            names = []
            for c in rt.boxes.cls:
                names.append(name[int(c)])

            df = pd.DataFrame(np.zeros([len(names), 4]))
            df.columns = ['xmin', 'ymin', 'xmax', 'ymax']

            
            boxes = rt.boxes  
            for idx, box in enumerate(boxes):
                coordinate = box.xyxy
                row = np.array(coordinate, dtype=float)
                df.loc[idx] = row
            df['predict'] = names
            df = df.sort_values('xmin')

        # ============= Extract maximum point coordinate =============
            try:
                if names.count('max') > 1:
                    x_temp = 0
                    df_temp=df[df['predict'] == 'max']
                    for et in df_temp.iterrows():
                        if et[1]['xmax'] > x_temp:
                            x_temp = et[1]['xmax']
                            end = et[1]
                    self.c = [((end['xmax'] + end['xmin'])/2),
                        ((end['ymax'] + end['ymin'])/2)]
                else:
                    end = df[df['predict'] == 'max']
                    self.c = [((end['xmax'].values.tolist()[0] + end['xmin'].values.tolist()[0])/2),
                        ((end['ymax'].values.tolist()[0] + end['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found maximum point'
                return None
            
            # ============= Extract minimum point coordinate =============
            try:
                if names.count('min') > 1:
                    x_temp = 0
                    df_temp=df[df['predict'] == 'min']
                    for et in df_temp.iterrows():
                        if et[1]['xmax'] > x_temp:
                            x_temp = et[1]['xmax']
                            end = et[1]
                    self.b = [((start['xmax'] + start['xmin'])/2),
                        ((start['ymax'] + start['ymin'])/2)]
                else:
                    start = df[df['predict'] == 'min']
                    self.b = [((start['xmax'].values.tolist()[0] + start['xmin'].values.tolist()[0])/2),
                        ((start['ymax'].values.tolist()[0] + start['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found minimum point'
                return None
            
            # ============= Extract center point coordinate =============

            try:
                if 'center' not in names:
                    self.cal_center()
                else:
                    middle = df[df['predict'] == 'center']
                    self.a = [((middle['xmax'].values.tolist()[0] + middle['xmin'].values.tolist()[0])/2),
                        ((middle['ymax'].values.tolist()[0] + middle['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found middle point'
                return None
            
            # ============= Extract needle tips point coordinate =============

            try:
                if 'tips' not in names:
                    self.needle_tips_point_detect()
                else:
                    tips = df[df['predict'] == 'tips']
                    self.d = [((tips['xmax'].values.tolist()[0] + tips['xmin'].values.tolist()[0])/2),
                    ((tips['ymax'].values.tolist()[0] + tips['ymin'].values.tolist()[0])/2)]
            except:
                    self.error_state = False
                    self.predicted_value = 'Not found needle tips'
                    return None
            

    def distance(self,xy, ct):
        x1,y1 = xy
        x2,y2 = ct
        return math.sqrt(((x2-x1)**2)+((y2-y1)**2))
    

    def needle_tips_point_detect(self):
        # draw = ImageDraw.Draw(self.img)
        coordinates = []
        tip = []
        results = self.model.predict(self.img,conf=self.conf,save=False,retina_masks=True)
    
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
                    dist = self.distance((i[0], i[1]), self.a)
                    if dist > max_temp:
                        max_temp = dist
                        coor_temp = (i[0], i[1])
                tip.append(coor_temp)
                coordinates = []
            self.d = tip[name.index('needle')]
            

    def predict_value(self):

        im_shape = np.array(self.img).shape
        xmax = im_shape[1]
        ymin = im_shape[0]

        point_b = abs(np.arctan2((self.a[1]-self.b[1]),(self.b[0]-self.a[0]))* 180 / np.pi)
        point_c = np.arctan2((self.a[1]-self.c[1]),(self.c[0]-self.a[0]))* 180 / np.pi

        # print('Degree B :',point_b)
        # print('Degree C :',point_c)

        end_point = 360 - (abs(point_b) + abs(point_c))
        incre = (self.end_value-self.start_value)/end_point

        point_d = np.arctan2((self.d[1]-self.a[1]),(self.d[0]-self.a[0]))* 180 / np.pi

        if self.d[1] < self.a[1]:
            # point_d = np.arctan2((self.a[1]-self.d[1]),(self.d[0]-self.a[0]))* 180 / np.pi
            
            self.predicted_value = incre * abs((360-abs(point_d))-abs(point_b)) + self.start_value
            # print('Quadant 4')
        else:
            
            self.predicted_value = incre * abs((abs(point_d))-abs(point_b)) + self.start_value

    def find_intersection_point(self, m1, c1, m2, c2):
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1

        return x, y
        
    def cal_center(self):

        x=self.b[0]
        y=self.b[1]
        endxb = x + 1200 * math.cos(math.radians(self.angleb))
        
        endyb = y + 1200 * math.sin(math.radians(self.angleb))

    
        x = self.c[0]
        y = self.c[1]

        endxc = x + 600 * math.cos(math.radians(self.anglec))
        
        endyc = y + 600 * math.sin(math.radians(self.anglec)) 

        m1 = (endyc - self.c[1]) / (endxc - self.c[0])
        c1 = self.c[1] - m1 * self.c[0]

        m2 = (self.b[1] - endyb) / (self.b[0] - endxb)
        c2 = self.b[1] - m2 * self.b[0]
        self.a = self.find_intersection_point(m1, c1, m2, c2)
        # draw = ImageDraw.Draw(self.img)
        # draw.line(((self.b[0],self.b[1]),(endxb,endyb)), fill=(255,255,0),width=10)
        # draw.line(((self.c[0],self.c[1]),(endxc,endyc)), fill=(255,255,0),width=10)



    def draw_img(self):
        draw = ImageDraw.Draw(self.img)
        # draw.ellipse(((self.d[0]-10, self.d[1]-10), ((self.d[0]+10,self.d[1]+10))), fill=(0,255,0,255))
        # draw.ellipse(((self.a[0]-10, self.a[1]-10), ((self.a[0]+10,self.a[1]+10))), fill=(0,255,0,255))
        # draw.ellipse(((self.b[0]-10, self.b[1]-10), ((self.b[0]+10,self.b[1]+10))), fill=(0,255,0,255))
        # draw.ellipse(((self.c[0]-10, self.c[1]-10), ((self.c[0]+10,self.c[1]+10))), fill=(0,255,0,255))
        draw.line(((self.a[0],self.a[1]), (self.d[0],self.d[1])), fill=(0,255,0), width=10)
        

    def show_result(self, draw : bool = False, show_image :bool = False, save : bool = False):
        if draw:
            self.draw_img()

        if show_image:
            plt.imshow(self.img)
            plt.axis(False)
            plt.title("Result = {:.1f}".format(self.predicted_value))
            plt.show()

        if save:
            self.img.save('src/results/test_save.png')

pred = ACWValuePrediction(ACW_MODEL_CONFIG.MAX_VALUE, conf=GENERAL_CONFIG.CONFIDENCE, file_name=join(ACW_MODEL_CONFIG.TEST_IMAGE_DIRECTORY, 'testacw_4.jpg'))
pred.draw_img(show = True)
print(pred.predicted_value)