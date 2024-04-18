
import json
import numpy as np
import re

from evaluation import evaluation_suit


# right structure
# answer= "There is a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the front of the ego vehicle, and a black sedan to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,888.0,512.0>, <c2,CAM_FRONT,1024.0,512.0>, <c3,CAM_FRONT,1024.0,512.0>, <c4,CAM_FRONT,1024.0,512.0>, and <c5,CAM_FRONT,1024.0,512.0>."
# GT= [888.0, 512.0, 1024.0, 512.0, 1024.0, 512.0, 1024.0, 512.0, 1024.0, 512.0]

# false structure
# answer= "There is a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black sedan to the back of the ego vehicle, and a black sedan to the front of the ego vehicle. The IDs of these objects are <c1,CAM_FRONT,1055.5,500.0>, <c2,CAM_BACK,1055.5,500.0>, <c3,CAM_FRONT,1055.5,500.0>, <c4,CAM_BACK,1055.5,500.0>, <c5,CAM_FRONT,1055.5,500.0>, <c6,CAM_BACK,1055.5,500.0>, and <c7,CAM_FRONT,1055.5,500"
# GT= [1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0, 1055.5, 500.0]

# answer_nums = re.findall(r'\d+\.\d+', answer)
# GT_nums = re.findall(r'\d+\.\d+', GT)


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
    print(answer_coords[-1])
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
    root_path1 = "output_val.json"
    with open(root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
        
    old_pred_file = pred_file
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    print(pred_file == old_pred_file)
    
    root_path2 = "v1_1_val_nus_q_only.json"
    with open(root_path2, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
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
                # assert pred_file[idx]["gt_answer"] == GT, print(pred_file[idx]["gt_answer"], GT)
                if first_flag:
                    first_flag = False
                    evaluation.set_graph(predict, GT)
                    break
                
            raise

if __name__ == '__main__':
    # main()
    tt()