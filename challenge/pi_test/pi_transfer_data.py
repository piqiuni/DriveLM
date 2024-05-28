import json
import os
import sys

''' From:
[
    {
        "id": "f0f120e4d4b0441da90ec53b16ee169d_4a0798f849ca477ab18009c3a20b7df2_1",
        "image": [
            "data/nuscenes/samples/CAM_FRONT/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg",
            "data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_LEFT__1537291010604799.jpg",
            "data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_RIGHT__1537291010620482.jpg",
            "data/nuscenes/samples/CAM_BACK/n008-2018-09-18-13-10-39-0400__CAM_BACK__1537291010637558.jpg",
            "data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-09-18-13-10-39-0400__CAM_BACK_LEFT__1537291010647405.jpg",
            "data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__1537291010628113.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat is the moving status of object <c1,CAM_BACK,1088.3,497.5>? Please select the correct answer from the following options: A. Reverse parking. B. Turn right. C. Going ahead. D. Turn left."
            },
            {
                "from": "gpt",
                "value": "D"
            }
        ]
    },
]
'''

''' To:
[
    {"conversations": [
        {"from": "user", "value": "<img>img_path</img>11111"},
        {"from": "assistant", "value": "22222"}
    ]},
]
'''

filepath = '../test_val_llama.json'
output_filepath = './val_llama.json'

filepath = '/home/ldl/pi_code/DriveLM/challenge/pi_test/demo_data/demo_test_llama.json'
output_filepath = './demo_data/demo_history_data.json'


output_filepath2 = ''
# output_filepath2 = '/home/ldl/pi_code/swift/pi_code/trainning_llama.json'
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
with open(output_filepath, 'w') as f:
    json.dump(output_data, f, indent=4)

if output_filepath2:
    with open(output_filepath2, 'w') as f:
        json.dump(output_data, f, indent=4)

print(f"Data transfered from {filepath} to {output_filepath} successfully!")