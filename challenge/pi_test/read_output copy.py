import json
import os
import textwrap
import cv2
import numpy as np

class ReadOutput(object):
    def __init__(self) -> None:
        pass
        self.image_paths = []
        self.images = []
    
    def read_json(self, filepath):
        """
        Reads a JSON file and returns the data as a dictionary.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    def draw_box(self, image, box_pos, box_size, color):
        """
        Draws a box on an image based on the provided box information.
        """
        x, y = box_pos
        w, h = box_size
        cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
        print((x, y), (x + w, y + h))


    def get_obj_infos(self, answer):
        object_ids = []
        for word in answer.split():
            if word.startswith("<c"):
                object_ids.append(word)
        return object_ids

    def draw_infos(self, object_ids, box_size, color):
        for object_id in object_ids:
            # Extract object information
            object_info = object_id.split('<')[1].split('>')[0].split(',')
            obj_id, camera, x, y = object_info
            x, y = float(x), float(y)

            # Find corresponding image based on camera
            filtered_paths = [path for path in self.image_paths if f"{camera}" in path]
            # image_index = image_paths.index(f"{camera}")
            image_index = self.image_paths.index(filtered_paths[0])
            print(x, y, image_index)
            # Draw box on the image
            self.draw_box(self.images[image_index], (int(x), int(y)), box_size, color)

    def get_combined_image(self, cimage_name, images_path, question, answer, GT):
        pass

def main():
    """
    Main function to read data, process and display images.
    """
    reader = ReadOutput()
    # Read data from JSON files
    data1 = reader.read_json("pi_test/pi_test_llama.json")
    data2 = reader.read_json("pi_test/pi_output.json")
    abs_data_path = "/home/ldl/pi_code/DriveLM/challenge/llama_adapter_v2_multimodal7b"
    save_path = "./pi_test"


    if len(data1) != len(data2):
        print("Data length mismatch!")
        

    # Get image paths and questions/answers
    #   dataset_path = 
    image_paths = data1[0]["image"]
    question = data2[0]["question"]
    answer = data2[0]["answer"]
    GT = data1[0]["conversations"][1]["value"]

    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0)]
    box_sizes = [(50, 50), (80, 80), (100, 100)]
    order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    # print(image_paths)

    # Load images
    images = []
    for image_path in image_paths:
        abs_image_path = os.path.join(abs_data_path, image_path)
        image = cv2.imread(abs_image_path)
        images.append(image)
        # print(abs_image_path, image.shape)
        # cv2.imwrite(os.path.join(save_path, image_path.split("/")[-1]), image)
    # Find object IDs in answer
    # object_ids = []
    # for word in answer.split():
    #     if word.startswith("<c"):
    #         object_ids.append(word)
    #         print(word)
    #     print(word)
    # print(answer)
    # print(answer.split())
    # print(object_ids)
    
    qes_object_ids = get_obj_infos(question)
    

    # Draw boxes on images for each object ID
    for object_id in qes_object_ids:
        # Extract object information
        object_info = object_id.split('<')[1].split('>')[0].split(',')
        obj_id, camera, x, y = object_info
        x, y = float(x), float(y)

        # Find corresponding image based on camera
        filtered_paths = [path for path in image_paths if f"{camera}" in path]
        # image_index = image_paths.index(f"{camera}")
        image_index = image_paths.index(filtered_paths[0])
        print(x, y, image_index)
        # Draw box on the image
        draw_box(images[image_index], (int(x), int(y)), box_sizes[0], colors[0])
    
    ans_object_ids = get_obj_infos(answer)

    # Draw boxes on images for each object ID
    for object_id in ans_object_ids:
        # Extract object information
        object_info = object_id.split('<')[1].split('>')[0].split(',')
        obj_id, camera, x, y = object_info
        x, y = float(x), float(y)

        # Find corresponding image based on camera
        filtered_paths = [path for path in image_paths if f"{camera}" in path]
        # image_index = image_paths.index(f"{camera}")
        image_index = image_paths.index(filtered_paths[0])
        print(x, y, image_index)
        # Draw box on the image
        draw_box(images[image_index], (int(x), int(y)), box_sizes[1], colors[1])
        
    GT_object_ids = get_obj_infos(GT)

    # Draw boxes on images for each object ID
    for object_id in GT_object_ids:
        # Extract object information
        object_info = object_id.split('<')[1].split('>')[0].split(',')
        obj_id, camera, x, y = object_info
        x, y = float(x), float(y)

        # Find corresponding image based on camera
        filtered_paths = [path for path in image_paths if f"{camera}" in path]
        # image_index = image_paths.index(f"{camera}")
        image_index = image_paths.index(filtered_paths[0])
        print(x, y, image_index)
        # Draw box on the image
        draw_box(images[image_index], (int(x), int(y)), box_sizes[2], colors[2])

    matching_indexes = []
    for o in order:
        for i, path in enumerate(image_paths):
            if str('__'+o+'__') in path:
                print(i, path)
                matching_indexes.append(i)
                break
            
    # print(matching_indexes)
    top_row = cv2.hconcat([images[i] for i in matching_indexes[:3]])
    bottom_row = cv2.hconcat([images[i] for i in matching_indexes[3:]])
    combined_image = cv2.vconcat([top_row, bottom_row])
    
    
    # print(combined_image.shape)
    cv2.imwrite(os.path.join(save_path, "CombinedImage.png"), combined_image)
    
    print("Que:", question)
    print("Ans:", answer)
    print("GT :", GT)
    
    
    # 创建空白区域
    blank_row = np.zeros((1000, combined_image.shape[1], 3), dtype=np.uint8)   # 调整空白区域的高度和通道数

    # 绘制 question 和 answer 文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (255, 255, 255)   # 白色文本颜色

    question_text = "Question: {}".format(question)
    answer_text = "Answer: {}".format(answer)
    GT_text = "GT: {}".format(GT)

    # 在空白区域上方绘制 question 和 answer，自动换行
    y = 50

    width = combined_image.shape[1]/20
    # 绘制 question 文本
    for line in textwrap.wrap(question_text, width=width):
        (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.putText(blank_row, line, (10, y), font, font_scale, colors[0], font_thickness, cv2.LINE_AA)
        y += line_height*2

    # 增加一行间距
    y += line_height

    # 绘制 answer 文本
    for line in textwrap.wrap(answer_text, width=width):
        (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.putText(blank_row, line, (10, y), font, font_scale, colors[1], font_thickness, cv2.LINE_AA)
        y += line_height*2
        print(line, y)
        
    # 增加一行间距
    y += line_height

    # 绘制 answer 文本
    for line in textwrap.wrap(GT_text, width=width):
        (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.putText(blank_row, line, (10, y), font, font_scale, colors[2], font_thickness, cv2.LINE_AA)
        y += line_height*2
        print(line, y)
        
    final_y = y
    blank_row = blank_row[:final_y+line_height, :, :]   # 裁剪空白区域，使其与 combined_image 高度一致

    # 将空白区域和 combined_image 垂直拼接
    combined_image_with_text = cv2.vconcat([combined_image, blank_row])

    # 保存带有文本的图像
    cv2.imwrite(os.path.join(save_path, "CombinedImage_with_text.png"), combined_image_with_text)
    
    

if __name__ == "__main__":
    main()