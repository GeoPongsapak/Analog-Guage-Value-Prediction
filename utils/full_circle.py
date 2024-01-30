from os import getcwd
import sys
sys.path.append(getcwd())
from config.configuration import GENERAL_CONFIG, FULL_CIRCLE_MODEL_CONFIG
from config.libaries import *
class ValuePredict:

    def __init__(self, end_value, file_name, start_value = 0, conf=0.3, angleb = 45, anglec = 135) -> None:
        self.file_name = file_name
        self.end_value = end_value
        self.start_value = start_value
        self.conf = conf
        self.model = FULL_CIRCLE_MODEL_CONFIG.MODEL
        self.model_whole = FULL_CIRCLE_MODEL_CONFIG.WHOLE_GUAGE_MODEL
        self.needle_model = FULL_CIRCLE_MODEL_CONFIG.NEEDLE_MODEL
        self.angleb = angleb 
        self.anglec = anglec
        self.error_state = True
        self.img = Image.open(file_name)

        self.get_result()
        if not self.error_state:
            return None
        
        if self.d[1] > self.b[1]:
            self.predicted_value = 0
        else:
            self.predict_value()
        
        self.show_result()
    
    def distance(self, xy, ct):
        x1,y1 = xy
        x2,y2 = ct
        return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

    def get_result(self):
        
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
            
            
            

            if 'tips' not in names:
                try:

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
                except:
                    self.error_state = False
                    self.predicted_value = 'Not found needle tips'
                    return None
            else:
                try:
                    tips = df[df['predict'] == 'tips']
                    self.d = [((tips['xmax'].values.tolist()[0] + tips['xmin'].values.tolist()[0])/2),
                    ((tips['ymax'].values.tolist()[0] + tips['ymin'].values.tolist()[0])/2)]
                except:
                    self.error_state = False
                    self.predicted_value = 'Not found needle tips'
                    return None

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



    def show_result(self):
        draw = ImageDraw.Draw(self.img)
        # draw.line(((self.a[0],self.a[1]), (self.b[0],self.b[1])), fill=128, width=3)
        draw.line(((self.a[0],self.a[1]), (self.d[0],self.d[1])), fill=(0,255,0), width=6)
        # draw.line(((self.a[0],self.a[1]), (self.c[0],self.c[1])), fill=(0,0,255), width=3)
        plt.imshow(self.img)
        plt.title("{:.1f}".format(self.predicted_value))
        plt.axis(False)
        plt.show()

    

a = ValuePredict(FULL_CIRCLE_MODEL_CONFIG.MAX_VALUE,join(FULL_CIRCLE_MODEL_CONFIG.TEST_IMAGE_DIRECTORY, 'test11.jpg'), conf=GENERAL_CONFIG.CONFIDENCE)
print(a.predicted_value)                