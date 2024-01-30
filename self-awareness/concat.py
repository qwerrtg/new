import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--template", type=str, help="Path to the template file.")
parser.add_argument("--attribute", type=str, default="attribute.json", help="Path to the model system.")
parser.add_argument("--output", type=str, default="output.txt", help="Path to the output file.")

args = parser.parse_args()


prompt = """假设你是一个用户可自定义的讯飞星火开源的AI助手，在给定的人设背景下回复用户问题<ret>##人设背景如下：{attr}##用户：{{{input}}}##参考答案：{{{target}}}##回答：{{}}"""
attribute = json.load(open(args.attribute, "r"))

assert len(attribute) > 0, "The number of attribute must greater than one."
for k, v in attribute.items():
    print(f"\t{k}: {v}")

joined_att = '，'.join([f'(你的{key}：{value})' for key, value in attribute.items()])

data = open(args.template, "r").readlines()
fw = open(args.output, "w")
for i in range(len(data)):
    line = json.loads(data[i])
    new_line = prompt.format(attr=joined_att, input=line["input"], target=line["target"])
    fw.write(new_line + "\n")
    