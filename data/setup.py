import os

import json
import pickle

def make_videodatainfo_msvd():
    train_split = (0, 1199)  # train_size = 1200
    val_split = (1200, 1299)  # val_size = 100
    test_split = (1300, 1969)  # test_size = 670

    splits = ["train", "validate", "test"]

    in_video_path = "dataset/MSVD/YouTubeClips"
    MSVD_idx_map_file_path = "input/msvd/msvd_videoname2idx_map.pkl"
    MSVD_ref_file_path = "input/msvd/msvd_ref.pkl"
    out_video_path = "dataset/MSVD/videos"

    if not os.path.exists(out_video_path):
        os.mkdir(out_video_path)


    def update_index(idx):
        for split_i, split in enumerate([train_split, val_split, test_split]):
            if split[1] >= idx >= split[0]:
                return split_i, idx - split[0]


    MSVD_idx_map_file = open(MSVD_idx_map_file_path, 'rb')
    MSVD_idx_map = pickle.load(MSVD_idx_map_file)
    MSVD_ref_file = open(MSVD_ref_file_path, 'rb')
    MSVD_ref = pickle.load(MSVD_ref_file)

    output = {"info": {"contributor": "anonymous", "data_created": "2021-08-23"}, "videos": [], "sentences": []}

    video_list = os.listdir(in_video_path)
    print(MSVD_idx_map)
    print(len(MSVD_ref[2]))

    sen_id = 0
    for i, video_name in enumerate(MSVD_idx_map):
        # print(i, MSVD_idx_map[video_name])
        split_idx, new_idx = update_index(i)
        split_name = splits[split_idx]
        # print(split_idx, MSVD_ref[split_idx][new_idx])
        output["videos"].append({"video_id": "video%d" % i, "split": split_name, "id": i})
        for sentence in MSVD_ref[split_idx][new_idx]:
            output["sentences"].append({"caption": sentence, "video_id": "video%d" % i, "sen_id": sen_id})
            sen_id += 1

    # print(output)
    json.dump(output, open("input/msvd/train_videodatainfo.json", 'w'))


# Get Stanford CoreNLP 3.6.0 models for coco-caption/
def get_stanford_models():
    if not os.path.exists("coco-caption/pycocoevalcap/spice/lib/stanford-corenlp-3.6.0.jar"):
        os.system("cd coco-caption && ./get_stanford_models.sh")

# Get evaluation codes for Microsoft COCO Caption Evaluation 
# (BLUE_1, BLUE_2, BLUE_3, BLUE_4, METEOR, ROUGE_L, SPICE) 
# and put them under the coco-caption directory
def install_coco_caption():
    if not os.path.exists("coco-caption"):
        os.system("git clone https://github.com/tylin/coco-caption.git")

# Get evaluation codes for Consensus-based Image Description 
# Evaluation (CIDEr) and put them under the cider directory
def install_cider():
    if not os.path.exists("cider"):
        os.system("git clone https://github.com/plsang/cider.git")

# Set up environment if this is 
# the first time to use this code
def setup():
    install_coco_caption()
    install_cider()
    get_stanford_models()