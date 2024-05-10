
import argparse
import json
import numpy as np
import re
import os

from evaluation import evaluation_suit
from collections import Counter

import language_evaluation


# right structure
# answer= "There is a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the front of the ego vehicle, and a black sedan to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,888.0,512.0>, <c2,CAM_FRONT,1024.0,512.0>, <c3,CAM_FRONT,1024.0,512.0>, <c4,CAM_FRONT,1024.0,512.0>, and <c5,CAM_FRONT,1024.0,512.0>."
# GT= [888.0, 512.0, 1024.0, 512.0, 1024.0, 512.0, 1024.0, 512.0, 1024.0, 512.0]

# false structure
# answer= "There is a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, and a black sedan to the front of the ego vehicle. The IDs of these objects are <c1,CAM_FRONT,1055.5,500.0>, <c2,CAM_BACK,1055.5,500.0>, <c3,CAM_FRONT,1055.5,500.0>, <c4,CAM_BACK,1055.5,500.0>, <c5,CAM_FRONT,1055.5,500.0>, <c6,CAM_BACK,1055.5,500.0>, and <c7,CAM_FRONT,1055.5,500"
# GT= [1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0]

# answer_nums = re.findall(r'\d+\.\d+', answer)
# GT_nums = re.findall(r'\d+\.\d+', GT)


def find_most_frequent_word(text):
    # 使用空格分割字符串,获取所有单词
    words = text.split()
    # 统计单词出现的频率
    word_counts = Counter(words)
    # 获取出现频率最高的单词
    most_frequent_word, count = word_counts.most_common(1)[0]
    return most_frequent_word, count

def find_nth_occurrence(text, word, n):
    start = 0
    count = 0
    while count < n:
        index = text.find(word, start)
        if index == -1:
            return -1
        start = index + 1
        count += 1
    return index

def test():
    # 不完整的answer    
    answer = "There is a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, and a black sedan to the front of the ego vehicle. The IDs of these objects are <c1,CAM_FRONT,1055.5,501.0>, <c2,CAM_BACK,1055.5,502.0>, <c3,CAM_FRONT,1055.5,503.0>, <c4,CAM_BACK,1055.5,504.0>, <c5,CAM_FRONT,1055.5,505.0>, <c6,CAM_BACK,1055.5,506.0>, and <c7,CAM_FRONT,1055.5,507"
                                    
    answer_nums = re.findall(r'\d+\.\d+', answer)

    if 1:
        if len(answer_nums) %2 == 1:
            answer_nums = answer_nums[:-1]

    answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)

    # 提取answer中的坐标数据
    answer_coords = re.findall(r'<.*?>', answer)
    print(answer_coords)

    # 检查每个坐标数据的完整性并提取x和y值
    valid_coords = []
    for coord in answer_coords:
        numbers = re.findall(r'\d+\.\d+', coord)
        if len(numbers) == 2:  # 确保坐标是完整的（包含x和y值）
            valid_coords.append(numbers)

    # 将提取的坐标数据转换为NumPy数组并进行reshape操作
    valid_coords_np = np.array([list(map(float, coord)) for coord in valid_coords]).reshape(-1, 2)

    # 输出处理后的坐标数据
    print(valid_coords_np)

def tt():
    # 原始的不完整answer
    answer = "There is a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, and a black sedan to the front of the ego vehicle. The IDs of these objects are <c1,CAM_FRONT,1055.5,500.0>, <c2,CAM_BACK,1055.5,500.0>, <c3,CAM_FRONT,1055.5,500.0>, <c4,CAM_BACK,1055.5,500.0>, <c5,CAM_FRONT,1055.5,500.0>, <c6,CAM_BACK,1055.5,503.0>, and <c7,CAM_FRONT,1055.5,500"

    # 提取answer中的坐标数据
    answer_coords = re.findall(r'<.*?>', answer)
    answer2_coords = re.findall(r'<.', answer)
    print(answer_coords)
    print(answer2_coords)
    # 修复不完整的坐标
    last_coord = answer_coords[-1]
    last_coord_pos = answer.find(last_coord)
    new_answer = answer[:last_coord_pos + len(last_coord)]
    
    # 输出修复后的answer
    print(answer)
    print()
    print(new_answer)



def debug_match(answer):
   
    answer_nums = re.findall(r'\d+\.\d+', answer)
    # transform string into float
    print([list(map(float, x.split()))[0] for x in answer_nums])
    # print([list(map(float, x.split()))[0] for x in GT_nums])
    # raise
    answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
    


def main():
    root_path1 = "./pi_test/submit/output_internlm-xcomposer2-7b-chat_0509_1455.json"
    
    new_name = "refine_" + root_path1.split("/")[-1]
    print(root_path1.split("/")[:-1])
    new_root_path1 = os.path.join(*root_path1.split("/")[:-1], new_name)
    
    root_path2 = "v1_1_val_nus_q_only.json"
    
    
    language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
    # root_path1 = "mini_output.json"
    # new_root_path1 = "refine_mini_output.json"
    # root_path2 = "mini_test_eval.json"
    
    # parser = argparse.ArgumentParser(description='Evaluation')
    # parser.add_argument('--root_path1', type=str, default=".", help='path to prediction file')
    # parser.add_argument('--root_path2', type=str, default=".", help='path to test file')
    # args = parser.parse_args()
    
    with open(root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
        
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    
    with open(root_path2, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
    error_format_count = 0
    bad_answer_count = 0
    no_answer_count = 0
    ch_answer_count = 0
    repeat_count = 0
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]['QA']
            first_flag = True

            for i, qa in enumerate(frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]):
                question = qa['Q']
                GT = qa['A']
                tag = qa['tag']
                idx = scene_id + "_" + frame_id + "_" + str(i)
                predict = pred_file[idx]["answer"]
                # print(question)
                if question == "What's your comment on this scene?":
                    # print(f"---{predict}")
                    if predict == "":
                        predict = "the ego vehicle is driving on the road"
                        pred_file[idx]["answer"] = predict
                        bad_answer_count += 1
                    # elif len(predict) > 
                    
                    
                # if question == "":
                #     print(f"question == ''")
                #     predict = ""
                #     pred_file[idx]["answer"] = predict
                
                if 1:
                    pattern = r'[\u4e00-\uffff]+'
                    chinese_chars = re.findall(pattern, predict)
                    if chinese_chars:
                        ch_answer_count += 1
                        pos = predict.find(chinese_chars[0])
                        pos = max(0, pos - 8)
                        predict = predict[:pos]
                        # print(predict)
                        # print(set(chinese_chars))
                        if pos == 0:
                            predict = "None"
                        pred_file[idx]["answer"] = predict
                
                if 1:
                    m_w, m_c = find_most_frequent_word(predict)
                    if m_c > 10:
                        pos = find_nth_occurrence(predict, m_w, 3)
                        predict = predict[:pos]
                        pred_file[idx]["answer"] = predict
                        repeat_count += 1
                
                if 1:
                    answer_coords = re.findall(r'<.*?>', predict)
                    answer2_coords = re.findall(r'<.', predict)
                    if(len(answer_coords) != len(answer2_coords)):
                        error_format_count += 1
                        # print(len(answer_coords) , len(answer2_coords))
                        last_coord = answer_coords[-1]
                        last_coord_pos = predict.find(last_coord)
                        predict = predict[:last_coord_pos + len(last_coord)]
                        pred_file[idx]["answer"] = predict
                        print(f"ef_id:{idx}; ",end='')
                        # print(predict)
                        # print(new_predict)
                    # break
                    
                if predict == "":
                    no_answer_count += 1
                    print(f"no answer: {idx}, question:{question}")
                    # predict = "the ego vehicle is driving on the road"
                    # pred_file[idx]["answer"] = predict
                    
                results_gen = language_eval.run_evaluation(predict, GT)
                

    
    new_pred_file = [pred_file[key] for key in pred_file.keys()]
    with open(new_root_path1, 'w') as f :
        json.dump(new_pred_file, f, indent=4)
    print(f"bad_answer_count: {bad_answer_count}")
    print(f"error_format_count: {error_format_count}")
    print(f"no_answer_count: {no_answer_count}")
    print(f"ch_answer_count: {ch_answer_count}")
    print(f"repeat_count: {repeat_count}")
    print(f"new_path: {new_root_path1}")
    
if __name__ == '__main__':
    main()
    # tt()