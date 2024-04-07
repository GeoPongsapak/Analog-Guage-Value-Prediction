from os import getcwd
import sys
sys.path.append(getcwd())
from config.configuration import GENERAL_CONFIG, HALF_CIRCLE_MODEL_CONFIG
from config.libaries import *

from needle_tips_calculation import needle_tips_point_detect
from center_calculation import cal_center
# from value_prediction import value_calculation
class HalfCircle:

    def __init__(self,end_value : float, file_name : str = None, frame :np.ndarray = None, start_value : float = 0, conf : float = 0.3) -> None:
        self.file_name = file_name
        self.model = HALF_CIRCLE_MODEL_CONFIG.MODEL
        self.conf = conf
        self.angleb = 50
        self.anglec = 130
        self.start_value = start_value
        self.end_value = end_value
        self.needle_model = HALF_CIRCLE_MODEL_CONFIG.NEEDLE_MODEL
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
            # self.predicted_value = value_calculation(self.b, self.c, self.d, self.a, self.start_value, self.end_value)
        # self.draw_img()
        # self.show_result()


    def find_point(self):
        r = self.model(self.img, self.conf)
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

             # ============= Extract minimum point coordinate =============
            try:
                start = df[df['predict'] == 'min']
                self.b = [((start['xmax'].values.tolist()[0] + start['xmin'].values.tolist()[0])/2),
                    ((start['ymax'].values.tolist()[0] + start['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found minimum point'
                return None
            
             # ============= Extract maximum point coordinate =============
            try:
                end = df[df['predict'] == 'max']
                self.c = [((end['xmax'].values.tolist()[0] + end['xmin'].values.tolist()[0])/2),
                ((end['ymax'].values.tolist()[0] + end['ymin'].values.tolist()[0])/2)]
            except:
                self.error_state = False
                self.predicted_value = 'Not found maximum point'
                return None
            
            # ============= Extract middle point coordinate =============
            try:
                self.a = cal_center(self.b, self.c, 1200, 600, self.angleb, self.anglec)
            except:
                self.predicted_value = 'Not found middle point'
                return None
            
            # ============= Extract needle tips point coordinate =============

            try:
                if 'tips' not in names:
                    self.d = needle_tips_point_detect(self.img, self.needle_model, self.a)
                else:
                    tips = df[df['predict'] == 'tips']
                    self.d = [((tips['xmax'].values.tolist()[0] + tips['xmin'].values.tolist()[0])/2),
                    ((tips['ymax'].values.tolist()[0] + tips['ymin'].values.tolist()[0])/2)]
            except Exception as err:
                    self.error_state = False
                    self.predicted_value = 'Not found needle tips'
                    return None
            
    # def needle_tips_point_detect(self):
    #     coordinates = []
    #     results = self.needle_model.predict(self.img,self.conf,retina_masks=True)
    #     for result in results:
    #         masks = result.masks.xy
    #         for mask in masks[0]:
    #             coordinates.append((mask[0].astype(int), mask[1].astype(int)))

    #         def distance(xy, ct):
    #             x1,y1 = xy
    #             x2,y2 = ct
    #             return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

    #         max_temp = 0
    #         tip = (0,0)
    #         dist = 0
    #         for i in coordinates:
    #             dist = distance((i[0], i[1]), self.a)
    #             if dist > max_temp:
    #                 max_temp = dist
    #                 tip = (i[0], i[1])

    #         self.d = [tip[0], tip[1]]


    # def find_intersection_point(self, m1, c1, m2, c2):
    #     x = (c2 - c1) / (m1 - m2)
    #     y = m1 * x + c1

    #     return x, y
        
    # def cal_center(self):

    #     x=self.b[0]
    #     y=self.b[1]
    #     angle = 50
    #     endxb = x + 1200 * math.cos(math.radians(self.angleb))
        
    #     endyb = y + 1200 * math.sin(math.radians(self.angleb))

    
    #     x = self.c[0]
    #     y = self.c[1]

    #     endxc = x + 600 * math.cos(math.radians(self.anglec))
        
    #     endyc = y + 600 * math.sin(math.radians(self.anglec)) 

    #     m1 = (endyc - self.c[1]) / (endxc - self.c[0])
    #     c1 = self.c[1] - m1 * self.c[0]

    #     m2 = (self.b[1] - endyb) / (self.b[0] - endxb)
    #     c2 = self.b[1] - m2 * self.b[0]
    #     self.a = self.find_intersection_point(m1, c1, m2, c2)
        


    def draw_img(self):
        draw = ImageDraw.Draw(self.img)
        # draw.ellipse(((self.c[0]-10, self.c[1]-10), ((self.c[0]+10,self.c[1]+10))), fill=(255,0,0,255))

        # draw.ellipse(((self.b[0]-10, self.b[1]-10), ((self.b[0]+10,self.b[1]+10))), fill=(255,0,0,255))
        
        # draw.ellipse(((self.a[0]-10, self.a[1]-10), ((self.a[0]+10,self.a[1]+10))), fill=(255,0,0,255))

        # draw.ellipse(((self.d[0]-10, self.d[1]-10), ((self.d[0]+10,self.d[1]+10))), fill=(0,255,0,255))

        # draw.line((self.a[0],self.a[1], (0,self.a[1])), fill=(255,255,255), width=20)

        # draw.line((self.a[0],self.a[1], (self.a[0],0)), fill=(255,255,255), width=20)

        # draw.line((self.a[0],self.a[1], (self.b[0], self.b[1])), fill=(255,255,255), width=20)

        # draw.line((self.a[0],self.a[1], (2000,self.a[1])), fill=(255,255,255), width=20)

        draw.line((self.a[0],self.a[1], (self.d[0], self.d[1])), fill=(0,255,0), width=20)
        
    def predict_value(self):
        point_b = np.arctan2((self.b[1]-self.a[1]),(self.a[0]-self.b[0]))* 180 / np.pi
        point_c = np.arctan2((self.a[1]-self.c[1]),(self.c[0]-self.a[0]))* 180 / np.pi
        # print('point B :',point_b)
        # print('point C :',point_c)
        end_point = 180 - (abs(point_b) + (abs(point_c)))
        # print('point End :',end_point)
        incre = (self.end_value+abs(self.start_value))/end_point
        # point_d = np.arctan2((self.a[1]-self.d[1]),(self.d[0]-self.a[0]))* 180 / np.pi
        point_d = abs(np.arctan2((self.d[1]-self.a[1]),(self.a[0]-self.d[0]))* 180 / np.pi)
        # print('point D :',point_d)
        self.predicted_value = incre * abs(point_d-abs(point_b)) + self.start_value
                

    def show_result(self, draw : bool = False, show_image :bool = False, save : bool = False, to_base64 : bool = False):
        if draw:
            self.draw_img()

        if show_image:
            plt.imshow(self.img)
            plt.axis(False)
            try:
                plt.title("Result = {:.1f}".format(self.predicted_value))
            except:
                plt.title("Result = {}".format(self.predicted_value))
            plt.show()

        if save:
            self.img.save('src/results/test_save.png')

        if to_base64:
            pil_im = self.img.convert('RGB')
            cv2_img = np.array(pil_im)
            cv2_img = cv2_img[:, :, ::-1].copy()
            _, img_encoded = cv2.imencode('.jpg', cv2_img)
            base64_encoded = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            return base64_encoded
       
a = HalfCircle(HALF_CIRCLE_MODEL_CONFIG.MAX_VALUE, file_name = join(HALF_CIRCLE_MODEL_CONFIG.TEST_IMAGE_DIRECTORY, 'testhc_1.jpg'), conf=GENERAL_CONFIG.CONFIDENCE)
a.show_result(draw=True, show_image=True)
print(a.predicted_value)

