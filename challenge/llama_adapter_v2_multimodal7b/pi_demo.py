import cv2
import llama
import torch
from PIL import Image
from tqdm import tqdm
import json
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from threading import Thread
import math

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class LLamaDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        filename = data_item['image']
        ids = data_item['id']
        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value']
        
        prompt = llama.format_prompt(question)

        if isinstance(filename, list):
            image_all = []
            for img_path in filename:
                image = cv2.imread(img_path)
                image = Image.fromarray(image)
                if self.transform:
                    image = self.transform(image)
                image_all.append(image)
            image = torch.stack(image_all, dim=0)
        else:
            image = cv2.imread(filename)
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

        return image, prompt, ids, question, answer

def worker(rank, gpu_id, args, data_dict):
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda")
    llama_dir = args.llama_dir

    model, preprocess = llama.load(args.checkpoint, llama_dir, llama_type="7B", device=device)
    model.eval()

    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    with open(args.data, 'r') as f:
        data_all = json.load(f)

    num_processes = args.num_processes
    data_per_process = math.ceil(len(data_all) / num_processes)
    start_idx = rank * data_per_process
    end_idx = min((rank + 1) * data_per_process, len(data_all))
    data_to_process = data_all[start_idx:end_idx]

    dataset = LLamaDataset(data_to_process, transform=transform_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    for batch in tqdm(dataloader):
        images, prompts, ids, questions, gt_answers = batch
        images = images.to(device)
        results = model.generate(images, prompts, temperature=0.2, top_p=0.1)
        
        for i, result in enumerate(results):
            print(f"Thread {rank}: Result - {result}")
            data_dict.append({'id': ids[i], 'question': questions[i], 'answer': result})
    
    print(f"Thread {rank} finished")

# add args
parser = argparse.ArgumentParser(description='LLAMA Adapter')
parser.add_argument('--llama_dir', type=str, default="/path/to/llama_model_weights", help='path to llama model weights')
parser.add_argument('--checkpoint', type=str, default="/path/to/pre-trained/checkpoint.pth", help='path to pre-trained checkpoint')
parser.add_argument('--data', type=str, default="../pi_test/pi_test_llama.json", help='path to test data')
parser.add_argument('--output', type=str, default="../pi_test/pi_output.json", help='path to output file')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for parallel processing')
parser.add_argument('--num_processes', type=int, default=8, help='number of gpus to use')
args = parser.parse_args()


def debug():
    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    pic = cv2.imread("../pi_test/test.png")
    
    image = Image.fromarray(pic)
    image = transform_train(image)
    pil_image = transforms.ToPILImage()(image)
    # Save the image
    pil_image.save("../pi_test/image.jpg")

# python pi_demo.py --llama_dir ./weights/ --checkpoint ./check_points/0325-checkpoint-3.pth --batch_size 1 --num_processes 2
# python pi_demo.py --llama_dir ./weights/ --checkpoint ./check_points/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth --batch_size 1 --num_processes 2

def main():
    num_gpus = args.num_processes
    print(f"Using {num_gpus} GPUs")
    
    data_dict = []
    threads = []
    for rank in range(num_gpus):
        t = Thread(target=worker, args=(rank, rank, args, data_dict))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    with open(args.output, "w") as f:
        json.dump(data_dict, f, indent=4)



if __name__ == '__main__':
    debug()
    # main()