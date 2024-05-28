import json
import os
import random
import sys
from typing import List

from tqdm import tqdm

''' From:
[
    {
    "f0f120e4d4b0441da90ec53b16ee169d": {
        "scene_description": "The ego vehicle proceeds through the intersection, continuing along the current roadway.",
        "key_frames": {
            "4a0798f849ca477ab18009c3a20b7df2": {
                "QA": {
                    "perception": [
                        {
                            "Q": "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
                            "A": "There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,1088.3,497.5>, <c2,CAM_BACK,864.2,468.3>, and <c3,CAM_FRONT,1043.2,82.2>.",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null,
                            "tag": [
                                2
                            ]
                        },
                        {
                            "Q": "What is the moving status of object <c1,CAM_BACK,1088.3,497.5>? Please select the correct answer from the following options: A. Back up. B. Stopped. C. Turn left. D. Going ahead.",
                            "A": "C",
                            "C": null,
                            "con_up": null,
                            "con_down": null,
                            "cluster": null,
                            "layer": null,
                            "tag": [
                                0
                            ]
                        }
                    ],
                    "prediction": [
                        ......
    }
    
    to:
=
]
'''

''' To: 仅在最后一个问答保留图片路径，滑动窗口，选择当前帧(key_frames)前60个问答作为history
[
    don't use this:
    {"conversations": [
        {"from": "user", "value": "111"},
        {"from": "assistant", "value": "1111"}
        {"from": "user", "value": "<img> samples/CAM_FRONT/xxx.jpg</img><img> samples/CAM_FRONT/xxx.jpg</img><img> samples/CAM_FRONT/xxx.jpg</img><img>samples/CAM_FRONT/xxx.jpg</img><img>samples/CAM_FRONT/xxx.jpg</img><img>samples/CAM_FRONT/xxx.jpg</img>222"},
        {"from": "assistant", "value": "2222"}
    ]},
    
    use this format:
    {"id": "1", "query": "<img>img_path</img><img>img_path2</img><img>img_path3</img>aaaaa", "response": "66666"}
    {"id": "2", "query": "<img>img_path</img><img>img_path2</img><img>img_path3</img>bbb", "response": "777", "history": [["aaaaa","66666"]]}
    {"id": "3", "query": "<img>img_path</img><img>img_path2</img><img>img_path3</img>ccc", "response": "888", "history": [["aaaaa","66666"], ["bbbbb","77777"]]}
    
]
'''

comment_question = "What's your comment on this scene?"
des_question = "Please describe the current scene."

des_question
io_question = "What are the important objects in the current scene"
nt_question = "What object should the ego vehicle notice"
ca_question = "What actions taken by the ego vehicle can lead to a collision" # with <c1,CAM_BACK,1088.3,497.5>
ea_question = "What actions could the ego vehicle take"
es_question = "what are safe actions to take for the ego vehicle"
ep_question = "Predict the behavior of the ego vehicle"                       # select


class data_transfer_history(object):
    def __init__(self, filepath, output_paths:List, his_len) -> None:
        with open(filepath, 'r') as f:
            self.test_file = json.load(f)
        self.output_paths = output_paths
        self.his_len = his_len
        self.output_data = []
        self.des_question = des_question
        self.task_keys = ["perception", "prediction", "planning", "behavior"]
        
    def new_frame(self, scene_id, frame_id, image_paths):
        image_paths = [image_paths[key].replace("../nuscenes/", "") for key in image_paths.keys()]
        image_paths_str = ''
        for i in range(len(image_paths)):
            image_paths_str += f"<img>{image_paths[i]}</img>"
            
        self.idx_head = scene_id + "_" + frame_id + "_"
        self.id_num = 0
        self.image_path_str = image_paths_str
        
    def add_qa_pair(self, question, answer, history):
        self.output_data.append(
            {
                "id": self.idx_head + str(self.id_num),
                "query": self.image_path_str + question,
                # "query": question,
                "response": answer,
                "history": history
            }
        )
        self.id_num += 1

    def get_history(self, frame_data_qa, task_idx, qa_idx):
        history = []
        now_len = sum(self.task_qa_lens[:task_idx]) + qa_idx
        if now_len < self.his_len:
            qas = self.total_qas[:now_len]
            for qa in qas:
                history.append([qa['Q'], qa['A']])
        else:
            # randomly get qa from each task(same num in each task)
            # 20, [8,8,8,3] ->[6,6,6,2]
            qa_num_each_task = self.his_len // (task_idx+1)
            for task_key in self.task_keys:
                qas = frame_data_qa[task_key]
                sampled_qas = random.sample(qas, min(qa_num_each_task, len(qas)))
                for qa in sampled_qas:
                    history.append([qa['Q'], qa['A']])
        return history
        
    def add_comment_qa(self, history):
        def get_qa_list(qas, str):
            return [qa for qa in qas if str.lower() in qa['Q'].lower()]
        question = comment_question
        # answer = ""
        answers = []
        self.total_qas
        list = [des_question, io_question, nt_question, ca_question, ea_question, es_question, ep_question]
        count_list = []
        des_res = get_qa_list(self.total_qas, des_question)
        if des_res:
            # answer += des_res[0]['A']
            answers.append(des_res[0]['A'])
            count_list.append(1)
        else:
            count_list.append(0)
        
        io_res = get_qa_list(self.total_qas, io_question)
        if io_res:
            # answer += io_res[0]['A']
            answers.append(io_res[0]['A'])
            count_list.append(1)
        else:
            count_list.append(0)
            
        nt_res = get_qa_list(self.total_qas, nt_question)
        if nt_res:
            # answer += nt_res[0]['A']
            answers.append(nt_res[0]['A'])
            count_list.append(1)
            # print(nt_res[0]['A'], nt_question)
            # What object should the ego vehicle notice
            # Firstly, notice that <c3,CAM_FRONT,1365.8,567.5>. The object is going ahead, so the ego vehicle should keep going ahead at the same speed. Secondly, notice that <c2,CAM_BACK,809.3,553.3>. The object is turning right, so the ego vehicle should keep going ahead at the same speed. Thirdly, notice that <c1,CAM_BACK_LEFT,639.2,403.3>. The object is stationary, so the ego vehicle should keep going ahead at the same speed. 
            count_list.append(0)
            
        ca_res = get_qa_list(self.total_qas, ca_question)
        if ca_res:
            start = ca_res[0]["Q"].find(ca_question) + len(ca_question)
            others = ca_res[0]["Q"][start:].strip()
            ans = ca_res[0]['A']
            if others.split(" ")[0] == "with":
                if "No such action will lead to a collision." == ans:
                    na_ca = "No such action will lead ego vehicle to a collision " + others[:-1] + '.'
                elif "collision" in ans:
                    na_ca = ans[:-1] + '.'
                else:
                    na_ca = ca_res[0]['A'][:-1].strip() + ' will lead ego vehicle to a collision ' + others[:-1] + '.'
                # answer += na_ca
                # print(na_ca)
                answers.append(na_ca)
            else:
                # answer += ca_res[0]['A']
                answers.append(ca_res[0]['A'])
            count_list.append(1)
        else:
            count_list.append(0)
            
        ea_res = get_qa_list(self.total_qas, ea_question)
        if ea_res:
            eq_ans = 'The action that the ego vehicle can take' + ea_res[0]['A'][10:]
            # answer += ea_res[0]['A']
            answers.append(eq_ans)
            count_list.append(1)
        else:
            count_list.append(0)
        
        es_res = get_qa_list(self.total_qas, es_question)
        if es_res:
            es_ans = 'The safe actions for the ego vehicle to take are ' + es_res[0]['A'].lower()
            # answer += es_res[0]['A']
            answers.append(es_ans)
            # print(es_ans)
            count_list.append(1)
        else:
            count_list.append(0)
        
        ep_res = get_qa_list(self.total_qas, ep_question)
        if ep_res:
            answer_key = ep_res[0]['A']
            answer_options = ep_res[0]['Q'].split(": ")[1]
            a_pos = answer_options.find(answer_key)
            next_letter = chr(ord(answer_key) + 1)
            na_pos = answer_options.find(next_letter)
            na_ep = answer_options[a_pos+3:na_pos]
            # answer += na_ep
            ss = na_ep.split('.')
            na_ans = ss[0][19:] + " and" + ss[1][19:] + '.'
            na_ans = 'The behavior of the ego vehicle is predicted to be ' + na_ans
            # print(na_ans)
            # Predict the behavior of the ego vehicle.
            answers.append(na_ans)
            count_list.append(1)
        else:
            count_list.append(0)
        
        answer = " ".join(answers)
        self.add_qa_pair(question, answer, history)
        print(answer)
        # print(answers)
        print(count_list)

    def start(self, ):
        keys = list(self.test_file.keys())
        for index in tqdm(range(len(keys))):
        # for scene_id in self.test_file.keys():
            scene_id = keys[index]
            # print(scene_id)
            scene_data = self.test_file[scene_id]['key_frames']
            description_flag = False
            if "scene_description" in self.test_file[scene_id]:
                description_flag = True
                description = self.test_file[scene_id]['scene_description']
            qa_lens = []
            for frame_id in scene_data.keys():
                image_paths = scene_data[frame_id]['image_paths']
                self.new_frame(scene_id, frame_id, image_paths)
                frame_data_qa = scene_data[frame_id]['QA']
                # self.add_qa_pair(self.des_question, description, [])
                if description_flag:
                    frame_data_qa["perception"].insert(0, {'Q':self.des_question, 'A':description})
                self.task_qa_lens = [len(frame_data_qa[task]) for task in self.task_keys]
                self.total_qas = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
                for i, task in enumerate(self.task_keys):
                    qas = frame_data_qa[task]
                    for j, qa in enumerate(qas):
                        question = qa['Q']
                        answer = qa['A']
                        history = self.get_history(frame_data_qa, i, j)
                        # for k in range(j-1, j-self.his_len-1, -1):
                        #     if k >= 0:
                        #         history.append([qas[k]['Q'], qas[k]['A']])
                        self.add_qa_pair(question, answer, history)   
                        # print(history)
                        # print()
                    # print(len(history))
                qa_lens.append(sum(self.task_qa_lens))
                history.append([question, answer])
                self.add_comment_qa(history)
            # print(f"len of scene:{sum([len(self.test_file[key]['key_frames']) for key in self.test_file.keys()])}")
            # print(qa_lens)
            # raise
            # break
        # raise
        for output_path in self.output_paths:
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(self.output_data, f, indent=4)
                print(f"Data transfered from {filepath} to {output_path} successfully!")
        


if __name__ == "__main__":
    
    filepath = './test_eval.json'
    output_filepath = './temp_history_trainning_data.json'
    output_filepath2 = '/home/ldl/pi_code/swift/pi_code/history_trainning_llama.json'
    his_len = 20
    output_filepaths = [output_filepath, output_filepath2]
    
    # filepath = './demo_data/demo_test_eval.json'
    # output_filepath = './demo_data/demo_history_data.json'
    # his_len = 20
    # output_filepaths = [output_filepath]
    
    trans = data_transfer_history(filepath, [output_filepath, output_filepath2], his_len)
    trans.start()
    