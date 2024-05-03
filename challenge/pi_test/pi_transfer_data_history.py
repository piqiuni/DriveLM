import json
import os
import sys

''' From:
[
    {
        "id": "f0f120e4d4b0441da90ec53b16ee169d_4a0798f849ca477ab18009c3a20b7df2_1",
        "image": [
            "data/nuscenes/samples/CAM_FRONT/xxx.jpg",
            "data/nuscenes/samples/CAM_FRONT_LEFT/xxx.jpg",
            "data/nuscenes/samples/CAM_FRONT_RIGHT/xxx.jpg",
            "data/nuscenes/samples/CAM_BACK/xxx.jpg",
            "data/nuscenes/samples/CAM_BACK_LEFT/xxx.jpg",
            "data/nuscenes/samples/CAM_BACK_RIGHT/xxx.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n111."
            },
            {
                "from": "gpt",
                "value": "2222"
            }
        ]
    },
    {
        "id": "f0f120e4d4b0441da90ec53b16ee169d_4a0798f849ca477ab18009c3a20b7df2_1",
        "image": [
            "data/nuscenes/samples/CAM_FRONT/xxx.jpg",
            "data/nuscenes/samples/CAM_FRONT_LEFT/xxx.jpg",
            "data/nuscenes/samples/CAM_FRONT_RIGHT/xxx.jpg",
            "data/nuscenes/samples/CAM_BACK/xxx.jpg",
            "data/nuscenes/samples/CAM_BACK_LEFT/xxx.jpg",
            "data/nuscenes/samples/CAM_BACK_RIGHT/xxx.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n222"
            },
            {
                "from": "gpt",
                "value": "2222"
            }
        ]
    },
]
'''

''' To: 仅在最后一个问答保留图片路径，滑动窗口，选择当前帧(key_frames)前60个问答作为history
[
    {"conversations": [
        {"from": "user", "value": "111"},
        {"from": "assistant", "value": "1111"}
        {"from": "user", "value": "<img> samples/CAM_FRONT/xxx.jpg</img><img> samples/CAM_FRONT/xxx.jpg</img><img> samples/CAM_FRONT/xxx.jpg</img><img>samples/CAM_FRONT/xxx.jpg</img><img>samples/CAM_FRONT/xxx.jpg</img><img>samples/CAM_FRONT/xxx.jpg</img>222"},
        {"from": "assistant", "value": "2222"}
    ]},
]
'''


filepath = '../test_val_llama.json'
output_filepath = ''
# output_filepath = './val_llama.json'
# output_filepath2 = ''
output_filepath2 = '/home/ldl/pi_code/swift/pi_code/history_trainning_llama.json'
# filepath = '../mini_test_llama.json'
# output_filepath = './mini_trainning_llama.json'
# output_filepath2 = '/home/ldl/pi_code/swift/pi_code/mini_trainning_llama.json'
image_path = '/home/ldl/pi_code/DriveLM/challenge/llama_adapter_v2_multimodal7b'


# 读取输入文件
with open(filepath, 'r') as f:
    data = json.load(f)

# 处理数据
output_data = []
for item in data:
    conversations = []
    id = item['id']
    images = item['image']
    picture_paths = [os.path.join(image_path, image) for image in images]
    # print(picture_paths)
    # raise
    picture_str = ''
    for i in range(len(picture_paths)):
        picture_path = picture_paths[i]
        picture_str += f"Picture {i+1}:<img>{picture_path}</img>\n"
    # print(picture_str)
    
    for conv in item['conversations']:
        if conv['from'] == 'human':
            # print(conv['value'][8:])
            # raise
            conv_from = 'user'
            value = picture_str + f"{conv['value'][8:]}"
        elif conv['from'] == 'gpt':
            conv_from = 'assistant'
            value = conv['value']
        
        conversations.append({
            'id': id,
            'from': conv_from,
            'value': value
        })
    # print(conversations)
    # raise
    output_data.append({
        'conversations': conversations
    })

# 写入输出文件
if output_filepath:
    with open(output_filepath, 'w') as f:
        json.dump(output_data, f, indent=4)

if output_filepath2:
    with open(output_filepath2, 'w') as f:
        json.dump(output_data, f, indent=4)

print(f"Data transfered from {filepath} to {output_filepath} successfully!")