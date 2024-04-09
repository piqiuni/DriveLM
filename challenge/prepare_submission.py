import json

# Please fill in your team information here
method = "LLaMA-BaseLine"  # <str> -- name of the method
team = "BIT-ININ"  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = ["Delun Li, Yang Chen, Wenchao Huang, Herui Li, Zihao Mao, Mingyu Hou, Rui Zhang, Chongshang Yan"]  # <list> -- list of str, authors
email = "piqiuni@qq.com"  # <str> -- e-mail address
institution = "Beijing Institute of Technology"  # <str> -- institution or company
country = "China"  # <str> -- country or region


def main():
    with open('output_val.json', 'r') as file:
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

if __name__ == "__main__":
    main()
