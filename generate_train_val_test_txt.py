import pandas as pd
import numpy as np
import os


class generate_train_test:
    def __init__(self, path, label=None):
        super(generate_train_test, self).__init__()
        self.path = path  # E:\RongPiaoCup\data
        self.label = label
        # modify
        self.label_dic = {
            'mudstone': 0,  # mudstone  mudstone
            'limestone': 1,  # limestone #limestone
            # 'dolomite': 2,  # dolomite dolomite
            # 'sandstone': 3,  # sandstone 砂岩
            'algal_limestone': 2  # algal_limestone algal_limestone
        }

        # create model input directory
        if not os.path.exists('model_input'):
            os.mkdir('model_input')

    def __call__(self):
        """
        :return:
        """
        train_list = []
        test_list = []
        val_list = []
        categories = os.path.join(self.path, 'data_delete')
        print(categories)
        for core_label in os.listdir(categories):
            index = 0
            label = self.label_dic[core_label]
            temp_path = os.path.join(categories, core_label)  # E:\RongPiaoCup\data\mudstone
            current_core_list = os.listdir(temp_path)
            print(temp_path)
            for image in current_core_list:
                index += 1
                print(image)
                image_path = os.path.join(temp_path, image)
                write_line = image_path + " " + str(label) + "\n"

                # 8:1:1
                # if index % 9 == 0:
                #     val_list.append(write_line)
                # elif index % 10 == 0:
                #     test_list.append(write_line)
                # else:
                #     train_list.append(write_line)

                # 7:3
                if index % 8 == 0 or index % 9 == 0 or index % 10 == 0:
                    val_list.append(write_line)
                else:
                    train_list.append(write_line)
        with open('model_input/train.txt', "w", encoding="utf-8") as f:
            for line in train_list:
                f.write(line)

        # with open('model_input/test.txt', "w", encoding="utf-8") as f:
        #     for line in test_list:
        #         f.write(line)

        with open('model_input/val.txt', 'w', encoding="utf-8") as f:
            for line in val_list:
                f.write(line)


if __name__ == '__main__':
    g = generate_train_test(path=os.getcwd())
    g()
