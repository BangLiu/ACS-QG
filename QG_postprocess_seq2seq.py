import argparse


# parser used to read argument
parser = argparse.ArgumentParser(description='QG_postprocess')

# experiment
parser.add_argument(
    '--input_file', type=str, default="", help='input file')
parser.add_argument(
    '--output_file', type=str, default="", help='output file')


def post_process(input):
    input_split = input.split()
    ulist = []
    [ulist.append(x) for x in input_split if x not in ulist]
    return " ".join(ulist)


def main(args):
    fin = open(args.input_file, "r", encoding="utf8")
    fout = open(args.output_file, "w", encoding="utf8")
    for line in fin:
        line_split = line.split("\t")
        question = line_split[2]
        processed_q = post_process(question)
        line_split[2] = processed_q
        fout.write("\t".join(line_split))
    fin.close()
    fout.close()


if __name__ == '__main__':
    main(parser.parse_args())
