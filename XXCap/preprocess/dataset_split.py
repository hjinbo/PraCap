import os
import json

def makejson(dataset):
    out_files_root = f'./data/{dataset}/outfiles/'
    if not os.path.exists(out_files_root):
        os.makedirs(out_files_root)
    tokenpath = f'./data/{dataset}/text/dataset_{dataset}.json'
    trainset_filename = f'captions_{dataset}_train.json'
    valset_filename = f'captions_{dataset}_val.json'
    testset_filename = f'captions_{dataset}_test.json'

    train_dict = {'images': [], 'annotations': []}
    valid_dict = {'images': [], 'annotations': []}
    test_dict = {'images': [], 'annotations': []}
    id = 0
    with open(tokenpath, 'r') as f:
        images = json.load(f)
        data = images['images']
        for d in data:
            image_file = d['filename']
            sentences = d['sentences']
            split = d['split']
            if split == 'train':
                train_dict['images'].append({'id': image_file, 'file_name': image_file})
                for sen in sentences:
                    ann_info = {}
                    caption = sen['raw']
                    ann_info['caption'] = caption
                    ann_info['image_id'] = image_file
                    ann_info['id'] = id
                    train_dict['annotations'].append(ann_info)
            elif split == 'val':
                valid_dict['images'].append({'id': image_file, 'file_name': image_file})
                for sen in sentences:
                    ann_info = {}
                    caption = sen['raw']
                    ann_info['caption'] = caption
                    ann_info['image_id'] = image_file
                    ann_info['id'] = id
                    valid_dict['annotations'].append(ann_info)
            elif split == 'test':
                test_dict['images'].append({'id': image_file, 'file_name': image_file})
                for sen in sentences:
                    ann_info = {}
                    caption = sen['raw']
                    ann_info['caption'] = caption
                    ann_info['image_id'] = image_file
                    ann_info['id'] = id
                    test_dict['annotations'].append(ann_info)
            id += 1

    print("Saving %d train images %d train annotations" % (len(train_dict["images"]), len(train_dict["annotations"])))
    with open(os.path.join(out_files_root, trainset_filename), "w") as f:
        json.dump(train_dict, f)

    print("Saving %d val images, %d val annotations" % (len(valid_dict["images"]), len(valid_dict["annotations"])))
    with open(os.path.join(out_files_root, valset_filename), "w") as f:
        json.dump(valid_dict, f)

    print("Saving %d test images %d test annotations" % (len(test_dict["images"]), len(test_dict["annotations"])))
    with open(os.path.join(out_files_root, testset_filename), "w") as f:
        json.dump(test_dict, f)


if __name__ == '__main__':
    makejson(dataset='coco')