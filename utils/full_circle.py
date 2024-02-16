from os import getcwd
import sys
sys.path.append(getcwd())
from config.configuration import GENERAL_CONFIG, FULL_CIRCLE_MODEL_CONFIG
from config.libaries import *
class ValuePredict:

    def __init__(self, end_value : float, file_name : str = None, frame :np.ndarray = None, start_value : float = 0, conf : float = 0.3) -> None:
        self.file_name = file_name
        self.end_value = end_value
        self.start_value = start_value
        self.conf = conf
        self.model = FULL_CIRCLE_MODEL_CONFIG.MODEL
        self.model_whole = FULL_CIRCLE_MODEL_CONFIG.WHOLE_GUAGE_MODEL
        self.needle_model = FULL_CIRCLE_MODEL_CONFIG.NEEDLE_MODEL
        self.angleb = 45 
        self.anglec = 135
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
        
        if self.d[1] > self.b[1]:
            self.predicted_value = 0
        else:
            self.predict_value()
        
        # self.show_result()
    
    def distance(self, xy, ct):
        x1,y1 = xy
        x2,y2 = ct
        return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

    def find_point(self):
        # ============= Detect and get coordinate =============
        results = self.model([self.file_name], conf=self.conf)
        for result in results:
            name = result.names
            names = []
            for c in result.boxes.cls:
                names.append(name[int(c)])

            df = pd.DataFrame(np.zeros([len(names), 4]))
            df.columns = ['xmin', 'ymin', 'xmax', 'ymax']

            boxes = result.boxes  
            for idx, box in enumerate(boxes):
                coordinate = box.xyxy
                row = np.array(coordinate, dtype=float)
                df.loc[idx] = row
            
            if 'middle' not in names:
                try:
                    names, df = self.cal_center(names, df)
                except:
                    self.error_state = False
                    self.predicted_value = 'Not found middle point'

            df['predict'] = names

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
            
            try:
                middle = df[df['predict'] == 'middle']
                self.a = [((middle['xmax'].values.tolist()[0] + middle['xmin'].values.tolist()[0])/2),
                    ((middle['ymax'].values.tolist()[0] + middle['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found middle point'
                return None
            
            
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
            
    def cal_center(self, names :list, df :pd.DataFrame):
        r = self.model_whole(self.file_name, conf=0.3)
        for rt in r:
            name_temp = rt.names
            names_temp = []
            for c in rt.boxes.cls:
                names_temp.append(name_temp[int(c)])

            df_temp = pd.DataFrame(np.zeros([len(names_temp), 4]))
            df_temp.columns = ['xmin', 'ymin', 'xmax', 'ymax']

            boxes = rt.boxes  
            for idx, box in enumerate(boxes):
                coordinate = box.xyxy
                row = np.array(coordinate, dtype=float)
                df_temp.loc[idx] = row

        df.loc[len(df)] = df_temp.iloc[0].tolist()
        names.append('middle')

        return names, df

    def needle_tips_point_detect(self):
        coordinates = []
        results = self.needle_model.predict(self.file_name,conf=0.5,save=False,retina_masks=True)
        for result in results:
            masks = result.masks.xy
            for mask in masks[0]:
                coordinates.append((mask[0].astype(int), mask[1].astype(int)))
            max_temp = 0
            tip = (0,0)
            dist = 0
            for i in coordinates:
                dist = self.distance((i[0], i[1]), self.a)
                if dist > max_temp:
                    max_temp = dist
                    tip = (i[0], i[1])

                self.d = [tip[0], tip[1]]
                          
    def predict_value(self):

        im_shape = np.array(self.img).shape
        xmax = im_shape[1]
        ymin = im_shape[0]

        point_b = abs(np.arctan2((self.a[1]-self.b[1]),(self.b[0]-self.a[0]))* 180 / np.pi) -90
        point_c = np.arctan2((self.a[1]-self.c[1]),(self.c[0]-self.a[0]))* 180 / np.pi

        end_point = 360 - (abs(point_b) + abs(point_c+90))
        incre = (self.end_value-self.start_value)/end_point

        if self.d[0] > xmax/2 and self.d[1] > ymin/2:
            point_d = np.arctan2((self.a[1]-self.d[1]),(self.d[0]-self.a[0]))* 180 / np.pi
            
            self.predicted_value = incre * (abs(point_d)+270-(point_b)) + self.start_value
            # predict_value = incre * (abs(point_d)+270-(90-point_b)) + self.start_value
            print('Quadant 4')
        else:
            point_d = np.arctan2((self.a[1]-self.d[1]),(self.a[0]-self.d[0]))* 180 / np.pi
            self.predicted_value = incre * abs(point_d+(90-abs(point_b))) + self.start_value

    def draw_img(self, show :bool = False, save : bool = False):
        draw = ImageDraw.Draw(self.img)
        # draw.ellipse(((self.d[0]-10, self.d[1]-10), ((self.d[0]+10,self.d[1]+10))), fill=(0,255,0,255))
        # draw.ellipse(((self.a[0]-10, self.a[1]-10), ((self.a[0]+10,self.a[1]+10))), fill=(0,255,0,255))
        # draw.ellipse(((self.b[0]-10, self.b[1]-10), ((self.b[0]+10,self.b[1]+10))), fill=(0,255,0,255))
        # draw.ellipse(((self.c[0]-10, self.c[1]-10), ((self.c[0]+10,self.c[1]+10))), fill=(0,255,0,255))
        draw.line(((self.a[0],self.a[1]), (self.d[0],self.d[1])), fill=(0,255,0), width=10)
        if show:
            plt.imshow(self.img)
            plt.axis(False)
            plt.title("Result = {:.1f}".format(self.predicted_value))
            plt.show()

        if save:
            self.img.save('src/results/test_save.png')


    

a = ValuePredict(FULL_CIRCLE_MODEL_CONFIG.MAX_VALUE,join(FULL_CIRCLE_MODEL_CONFIG.TEST_IMAGE_DIRECTORY, 'test11.jpg'), conf=GENERAL_CONFIG.CONFIDENCE)
print(a.predicted_value)                