import json
import os
import sys

scene_numbers = 80
frame_numbers = 2
# filepath = '../test_llama.json'
# mini_filepath = '../mini_test_llama.json'

# filepath = './test_eval.json'
# mini_filepath = './mini_test_eval.json'


with open(filepath, 'r') as f:
    data = json.load(f)

if type(data) == list:
    raise ValueError("Data type not supported")
elif type(data) == dict:
    final_data = {}
    print(f"scene_numbers: {scene_numbers/len(data.keys())*100:.1f}%: {scene_numbers}/{len(data.keys())}")
    new_data = {k: data[k] for k in list(data.keys())[:scene_numbers]}
    for k in new_data.keys():
        final_data[k] = {}
        final_data[k]["key_frames"] = {}
        frame_keys = new_data[k]["key_frames"].keys()
        for key in (list(frame_keys)[:frame_numbers]):
            final_data[k]["key_frames"][key] = new_data[k]["key_frames"][key]
else:
    raise ValueError("Data type not supported")
    
print(len(data.keys()))
with open(mini_filepath, 'w') as f :
    json.dump(final_data, f, indent=4)
    
# 获取当前脚本所在的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将上一级目录添加到模块搜索路径中
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# print(current_dir)
# print(parent_dir)

from convert2llama import convert2llama

root = "mini_test_eval.json"
dst = "mini_test_llama.json"
convert2llama(root, dst)
print(f"Generated {dst} from {root}, ", end='')


with open(dst, 'r') as f:
    data = json.load(f)
    print(f"len: {len(data)}")
