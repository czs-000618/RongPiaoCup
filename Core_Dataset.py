from PIL import Image
from torch.utils.data import Dataset


class CoreDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.image_labels = self._get_image_label()

    def __getitem__(self, index):
        image_path, label = self.image_labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def _get_image_label(self):
        global label, image_path
        image_list = []
        label_list = []
        data = open(self.data_path, encoding="utf-8").readlines()
        for line in data:
            # print(line.strip().split(' '))
            line_data = line.strip().split(' ')
            # 图片命名的过程中存在空格 需要特殊处理
            if len(line_data) == 3:
                image_path, label = line_data[0] + ' ' + line_data[1], line_data[2]
            else:
                image_path, label = line_data[0], line_data[1]
            image_list.append(image_path)
            label_list.append(int(label))

        return list(zip(image_list, label_list))

    def __len__(self):
        return len(self.image_labels)

if __name__ == '__main__':
    dataset = CoreDataset(data_path='model_input/train.txt')
    # print(dataset.image_labels)
    print('over')