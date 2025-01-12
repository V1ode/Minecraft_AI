import pandas as pd
from sklearn.preprocessing import LabelEncoder


my_data = pd.read_csv('_annotations.csv', delimiter=',')

# classes = []
# new_class_names = {
#     'Polar Bear': 'Polar_bear',
#     'iron golem': 'Iron_golem',
#     'Zombified Piglin': 'Zombified_piglin',
#     'Wither Skeleton': 'Wither_skeleton',
# }
#
# for i in range(len(my_data['class'])):
#     if(my_data['class'][i] not in classes):
#         classes.append(my_data['class'][i])

house_indexes = []

for i in range(len(my_data['class'])):
    if(my_data['class'][i] == 'house'):
        house_indexes.append(i)


labelencoder = LabelEncoder()
my_data = my_data.drop(house_indexes)
my_data = my_data.reset_index()
my_data['class'][:] = labelencoder.fit_transform(my_data['class'][:])
print(my_data['class'])

for i in range(len(my_data['class'])):
    filename = my_data['filename'][i][:len(my_data['filename'][i])-3] + "txt"
    my_file = open(filename, "w+")

    # if (my_data['class'][i] in classes):
    #     my_data['class'][i] = new_class_names[my_data['class'][i]]

    string = str(my_data['class'][i]) + " " + str(my_data['xmin'][i] / 640) + " " + str(my_data['ymin'][i] / 640) + " " + str(my_data['xmax'][i] / 640) + " " + str(my_data['ymax'][i] / 640)

    my_file.write(string)
    my_file.close()