from os import getcwd
import sys
sys.path.append(getcwd())
from config.libaries import *
from config.configuration import GENERAL_CONFIG, ONE_QUADANT_MODEL_CONFIG


class OneQuadant:

    def __init__(self,end_value : float, file_name : str = None, frame :np.ndarray = None, start_value : float = 0, conf : float = 0.3) -> None:
        self.file_name = file_name
        self.model = ONE_QUADANT_MODEL_CONFIG.MODEL
        self.img = Image.open(file_name)
        self.conf = conf
        self.angle = 115
        self.start_value = start_value
        self.end_value = end_value
        self.needle_model = ONE_QUADANT_MODEL_CONFIG.NEEDLE_MODEL
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
        
        try:
            self.cal_center()
        except:
            self.predicted_value = 'Not found middle point'
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
        self.draw_img()
        self.show_result() 


    def find_point(self):
        r = self.model(self.file_name, self.conf)
        nr = self.needle_model(self.file_name, self.conf)
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


            # im_array = rt.plot() 
            # im = Image.fromarray(im_array[..., ::-1]).convert('RGB')


            # plt.imshow(im)
            # plt.show() 
            # display(df)
            try:
                start = df[df['predict'] == 'start']
                self.b = [((start['xmax'].values.tolist()[0] + start['xmin'].values.tolist()[0])/2),
                    ((start['ymax'].values.tolist()[0] + start['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found minimum point'
                return None
            
            try:
                end = df[df['predict'] == 'end']
                self.c = [((end['xmax'].values.tolist()[0] + end['xmin'].values.tolist()[0])/2),
                ((end['ymax'].values.tolist()[0] + end['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found maximum point'
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
            
            
    def needle_tips_point_detect(self):
        coordinates = []
        results = self.needle_model.predict(self.file_name,self.conf,retina_masks=True)
        for result in results:
            masks = result.masks.xy
            for mask in masks[0]:
                coordinates.append((mask[0].astype(int), mask[1].astype(int)))

            def distance(xy, ct):
                x1,y1 = xy
                x2,y2 = ct
                return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

            max_temp = 0
            tip = (0,0)
            dist = 0
            for i in coordinates:
                dist = distance((i[0], i[1]), self.a)
                if dist > max_temp:
                    max_temp = dist
                    tip = (i[0], i[1])

            self.d = [tip[0], tip[1]]


    def find_intersection_point(self, m1, c1, m2, c2):
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1

        return x, y
        
    def cal_center(self):
        x = self.c[0]
        y = self.c[1]

        endxc = x + 600 * math.cos(math.radians(self.angle))
        
        endyc = y + 600 * math.sin(math.radians(self.angle)) 

        m1 = (endyc - self.c[1]) / (endxc - self.c[0])
        c1 = self.c[1] - m1 * self.c[0]

        m2 = (self.b[1] - self.b[1]) / (0 - self.b[0])
        c2 = self.b[1] - m2 * self.b[0]
        self.a = self.find_intersection_point(m1, c1, m2, c2)
        


    def draw_img(self):
        draw = ImageDraw.Draw(self.img)
        # draw.ellipse(((self.c[0]-10, self.c[1]-10), ((self.c[0]+10,self.c[1]+10))), fill=(255,0,0,255))

        # draw.ellipse(((self.b[0]-10, self.b[1]-10), ((self.b[0]+10,self.b[1]+10))), fill=(255,0,0,255))
        
        # draw.ellipse(((self.a[0]-10, self.a[1]-10), ((self.a[0]+10,self.a[1]+10))), fill=(255,0,0,255))

        # draw.ellipse(((self.d[0]-10, self.d[1]-10), ((self.d[0]+10,self.d[1]+10))), fill=(0,255,0,255))

        draw.line(((self.a[0],self.a[1]), (self.d[0],self.d[1])), fill=(0,255,0), width=3)
        

    def show_result(self):
        plt.imshow(self.img)
        plt.title("{:.1f}".format(self.predicted_value))
        plt.axis(False)
        plt.show()

    def predict_value(self,):
        point_b = np.arctan2((self.b[1]-self.a[1]),(self.b[0]-self.a[0]))* 180 / np.pi
        point_c = np.arctan2((self.a[1]-self.c[1]),(self.c[0]-self.a[0]))* 180 / np.pi

        end_point = (abs(point_b) + abs(point_c+90))-90
        incre = (self.end_value+abs(self.start_value))/end_point
        point_d = np.arctan2((self.a[1]-self.d[1]),(self.d[0]-self.a[0]))* 180 / np.pi
        if self.d[1] > self.b[1]:
            self.predicted_value = 0
        else:
            self.predicted_value = incre * abs(point_d+abs(point_b)) + self.start_value
                


       
a = OneQuadant(ONE_QUADANT_MODEL_CONFIG.MAX_VALUE,file_name = join(ONE_QUADANT_MODEL_CONFIG.TEST_IMAGE_DIRECTORY, 'test1q_3.jpg'), conf=GENERAL_CONFIG.CONFIDENCE)
print(a.predicted_value)
