import numpy as np
import json


def convert2llama(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    output = []
    scene_description_flag = False
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']
        if 'scene_description' in test_file[scene_id]:
            scene_description_flag = True
            description = test_file[scene_id]['scene_description']
            des_question = "Please describe the current scene."
        # print(description)
        # print(len(scene_data.keys()))
        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", "data") for key in image_paths.keys()]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
            idx = 0
            if scene_description_flag:
                output.append(
                        {
                            "id": scene_id + "_" + frame_id + "_" + str(idx),
                            "image": image_paths,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": "<image>\n" + des_question
                                },
                                {
                                    "from": "gpt",
                                    "value": description
                                },
                            ]
                        }
                    )
                idx += 1
            
            for qa in QA_pairs:
                question = qa['Q']
                answer = qa['A']
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "image": image_paths,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\n" + question
                            },
                            {
                                "from": "gpt",
                                "value": answer
                            },
                        ]
                    }
                )
                idx += 1

    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    root = "test_eval.json"
    dst = "test_llama.json"
    # root = "mini_test_eval.json"
    # dst = "mini_test_llama.json"
    # root = "v1_1_val_nus_q_only.json"
    # dst = "test_val_llama.json"
    
    # root = "pi_test/demo_data/demo_test_eval.json"
    # dst = "pi_test/demo_data/demo_test_llama.json"
    
    convert2llama(root, dst)
