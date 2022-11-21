from utils import *


# pathxml = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/annotations/*.xml"
# pathImages = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/images"

# class_dict = {'speedlimit':0, 'stop':1, 'crosswalk':2, 'trafficlight':3}

# df_train = generateDfFromXML(pathxml, pathImages)
# df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])
# df_train = add_bbox_coordinate(df_train)

# X = df_train[['filename', 'new_bb']]
# Y = df_train['class']

# print(X['filename'][20])
# print(Y[20])

pathxml = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/annotations/*.xml"
pathImages = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/images"

df_train = generateDfFromXML(pathxml, pathImages)

class_dict = {'speedlimit':0, 'stop':1, 'crosswalk':2, 'trafficlight':3}

df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])
df_train = add_bbox_coordinate(df_train)
df_train = add_bbox_per_pixel(df_train)

chosen_img = cv2.imread(df_train.iloc[0][0])
img_res = cv2.resize(chosen_img,(224,224))

bboxes = [df_train.iloc[0][4]*chosen_img.shape[1],df_train.iloc[0][5]*chosen_img.shape[0],
            df_train.iloc[0][6]*chosen_img.shape[1],df_train.iloc[0][7]*chosen_img.shape[0]]

bboxes_res = [df_train.iloc[0][4]*img_res.shape[1],df_train.iloc[0][5]*img_res.shape[0],
            df_train.iloc[0][6]*img_res.shape[1],df_train.iloc[0][7]*img_res.shape[0]]

