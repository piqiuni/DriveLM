import numpy as np
import json


def convert2llama(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    output = []
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", "data") for key in image_paths.keys()]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
            
            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']
                tag = qa['tag']
                if 0 in tag:
                    if 'select' in question:
                        ans = 'A'
                    else:
                        ans = 'No'
                elif 1 in tag:
                    ans = 'None'
                elif 2 in tag:
                    ans = 'None'
                elif 3 in tag:
                    ans = 'None'
                
                if question == "What's your comment on this scene?":
                    ques = []
                    ques.append("Please describe the current scene.")
                    ques.append("What are the important objects in the current scene, determine their status, and predict their future status.")
                    ques.append("What object should the ego vehicle notice?")
                    ques.append("What is the priority of the objects that the ego vehicle should consider?")
                    ques.append("Are there any safety issues in the current scene?")
                    ques.append("What are the safe actions to take for the ego vehicle?")
                    ques.append("What are the dangerous actions to take for the ego vehicle?")
                    ques.append("Predict the behavior of the ego vehicle.")
                    n_q = "".join(ques)
                    
                    
                    ans = "the ego vehicle is driving on the road. There are no important objects in the current scene, the prediction of them are none. The ego vehicle should notice the front objects in ego lane. There is no safety issue, the safe action to take for the ego vehicle is to keep going at the same speed, the dangerous action to take for the ego vehicle is to stop. The ego vehicle should keep going at the same speed based on the traffic rules, which has a high probability."    
                    
                    # ans = "the ego vehicle is driving on the road. There are no important objects in the current scene, the prediction of them are none. The ego vehicle should notice the front objects in ego lane. \
                    # Nothing will affect driving judgment. \
                    # There is no safety issue and the probability is high, the safe action to take for the ego vehicle is to keep going at the same speed, the dangerous action to take for the ego vehicle is to stop. The ego vehicle should keep going at the same speed based on the traffic rules, which has a high probability."    
                    
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "question": question,
                        "answer": ans
                    }
                )

    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    # root = "test_eval.json"
    # dst = "test_llama.json"
    # root = "mini_test_eval.json"
    # dst = "mini_test_llama.json"
    # root = "v1_1_val_nus_q_only.json"
    # dst = "test_val_llama.json"
    
    
    root = "v1_1_val_nus_q_only.json"
    dst = "./pi_test/submit/res_gen.json"
    convert2llama(root, dst)
