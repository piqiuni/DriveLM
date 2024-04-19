import json
import os
import textwrap
from typing import List
import cv2
import numpy as np
from tqdm import tqdm

class ReadOutput(object):
    def __init__(self, abs_data_path, save_path) -> None:
        self.abs_data_path = abs_data_path
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.image_paths = []
        self.images: List[cv2.Mat] = []
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.box_sizes = [(50, 50), (80, 80), (100, 100)]
        self.order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        # 绘制 question 和 answer 文本
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.2
        self.font_thickness = 2
    
    def read_json(self, filepath):
        """
        Reads a JSON file and returns the data as a dictionary.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        data_dict = {data["id"]: data for data in data}
        return data_dict

    def draw_box(self, image, box_pos, box_size, color):
        """
        Draws a box on an image based on the provided box information.
        """
        x, y = box_pos
        w, h = box_size
        cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)


    def get_obj_infos(self, answer):
        object_ids = []
        for word in answer.split():
            if word.startswith("<c"):
                object_ids.append(word)
        return object_ids

    def draw_obj_infos(self, object_ids, box_size, color):
        for object_id in object_ids:
            # Extract object information
            object_info = object_id.split('<')[1].split('>')[0].split(',')
            obj_id, camera, x, y = object_info
            x, y = float(x), float(y)

            # Find corresponding image based on camera
            filtered_paths = [path for path in self.image_paths if f"{camera}" in path]
            # image_index = image_paths.index(f"{camera}")
            image_index = self.image_paths.index(filtered_paths[0])
            # Draw box on the image
            self.draw_box(self.images[image_index], (int(x), int(y)), box_size, color)
            
    def get_matching_indexes(self, ):
        matching_indexes = []
        for o in self.order:
            for i, path in enumerate(self.image_paths):
                if str('__'+o+'__') in path:
                    matching_indexes.append(i)
                    break
        return matching_indexes

    def add_info(self, combined_image, question, answer, ground_truth):
        # 创建空白区域
        blank_row = np.zeros((1000, combined_image.shape[1], 3), dtype=np.uint8)   # 调整空白区域的高度和通道数


        question_text = "Que: {}".format(question)
        answer_text = "Ans: {}".format(answer)
        GT_text = "GT : {}".format(ground_truth)
        
        y = 50
        width = combined_image.shape[1]/20
        
        for i, text in enumerate([question_text, answer_text, GT_text]):
            for line in textwrap.wrap(text, width=width):
                (line_width, line_height), _ = cv2.getTextSize(line, self.font, self.font_scale, self.font_thickness)
                cv2.putText(blank_row, line, (10, y), self.font, self.font_scale, self.colors[i], self.font_thickness, cv2.LINE_AA)
                y += line_height*2
        
            # 增加一行间距
            y += line_height

        final_y = y - line_height
        blank_row = blank_row[:final_y+line_height, :, :]   # 裁剪空白区域，使其与 combined_image 高度一致
        # 将空白区域和 combined_image 垂直拼接
        combined_image_with_text = cv2.vconcat([combined_image, blank_row])
        return combined_image_with_text

    def get_combined_image(self, cimage_name, image_paths, question, answer, ground_truth):
        # Load images
        self.image_paths = image_paths
        self.images = []
        for image_path in image_paths:
            abs_image_path = os.path.join(self.abs_data_path, image_path)
            image = cv2.imread(abs_image_path)
            cv2.imwrite(os.path.join('./pi_test', 'test.png'), image)
            raise
            image_camera = image_path.split('/')[3]
            (text_width, text_height), baseline = cv2.getTextSize(image_camera, self.font, self.font_scale, self.font_thickness)
            img_height, img_width = image.shape[:2]
            text_x = int((img_width - text_width) / 2)
            if image_camera in self.order[:3]:
                text_y = int(text_height*1.5)
            else :
                text_y = int(img_height - text_height*1)
            cv2.putText(image, image_camera, (text_x,text_y), self.font, self.font_scale, (255,255,255), self.font_thickness, cv2.LINE_AA)
            self.images.append(image)
        qes_object_ids = self.get_obj_infos(question)
        ans_object_ids = self.get_obj_infos(answer)
        GT_object_ids = self.get_obj_infos(ground_truth)
        self.draw_obj_infos(qes_object_ids, self.box_sizes[0], self.colors[0])
        self.draw_obj_infos(ans_object_ids, self.box_sizes[1], self.colors[1])
        self.draw_obj_infos(GT_object_ids, self.box_sizes[2], self.colors[2])
        
        
        matching_indexes = self.get_matching_indexes()
        
        
        # print(matching_indexes)
        top_row = cv2.hconcat([self.images[i] for i in matching_indexes[:3]])
        bottom_row = cv2.hconcat([self.images[i] for i in matching_indexes[3:]])
        combined_image = cv2.vconcat([top_row, bottom_row])
        # print(combined_image.shape)
        
        combined_image_with_text = self.add_info(combined_image, question, answer, ground_truth)
        
        # 保存带有文本的图像
        cv2.imwrite(os.path.join(self.save_path, cimage_name), combined_image_with_text)
        
            
        
        
    
    def process(self, testdata_path, output_path):
        data1_dict = self.read_json("pi_test/pi_test_llama.json")
        data2_dict = self.read_json("pi_test/pi_output.json")
        if len(data1_dict) != len(data2_dict):
            print("Data length mismatch!")
        common_keys = data1_dict.keys() & data2_dict.keys()
        for key in tqdm(common_keys):
            cimage_name = 'Output_' + str(key) + '.png'
            image_paths = data1_dict[key]["image"]
            question = data2_dict[key]["question"]
            answer = data2_dict[key]["answer"]
            ground_truth = data1_dict[key]["conversations"][1]["value"]
            self.get_combined_image(cimage_name, image_paths, question, answer, ground_truth)
            # return

def main():
    """
    Main function to read data, process and display images.
    """
    testdata_path = "pi_test/pi_test_llama.json"
    output_path = "pi_test/pi_output.json"
    abs_data_path = "/home/ldl/pi_code/DriveLM/challenge/llama_adapter_v2_multimodal7b"
    save_path = "./pi_test/outputImages"
    
    reader = ReadOutput(abs_data_path, save_path)
    reader.process(testdata_path, output_path)
    
    absolute_path = os.path.abspath(save_path)
    print("Absolute path:", absolute_path)

if __name__ == "__main__":
    main()