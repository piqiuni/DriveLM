import json

# Please fill in your team information here
method = "Model1"  # <str> -- name of the method
team = "BIT-ININ"  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = ["Delun Li, Yang Chen, Wenchao Huang, Herui Li, Zihao Mao, Mingyu Hou, Rui Zhang, Chongshang Yan"]  # <list> -- list of str, authors
email = "piqiuni@qq.com"  # <str> -- e-mail address
institution = "Beijing Institute of Technology"  # <str> -- institution or company
country = "China"  # <str> -- country or region


def main():
    input_file = "./pi_test/submit/refine_output_internlm-xcomposer2-7b-chat_0523_2344.json"
    with open(input_file, 'r') as file:
        output_res = json.load(file)

    submission_content = {
        "method": method,
        "team": team,
        "authors": authors,
        "email": email,
        "institution": institution,
        "country": country,
        "results": output_res
    }

    with open('submission.json', 'w') as file:
        json.dump(submission_content, file, indent=4)   
        
    hf_path = "/home/ldl/pi_code/LLaMA-BaseLine/submission.json"
    with open(hf_path, 'w') as file:
        json.dump(submission_content, file, indent=4)   
    print(f"Submission file from '{input_file}' has been saved to 'submission.json' and '{hf_path}'")

if __name__ == "__main__":
    main()
