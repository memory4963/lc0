from glob import glob
import os
import argparse
from utils import ChunkParserInner
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_dir", type=str, default='../data/training*/*.gz')
    parser.add_argument("--out_dir", type=str, default='../data/preproc_train')
    parser.add_argument("--parts", type=str, default="-1/-1")
    return parser.parse_args()


def main():
    args = parse_args()

    file_list = sorted(glob(args.ori_dir))

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if not args.parts.startswith("-1"):
        part = int(args.parts.split("/")[0])
        total = int(args.parts.split("/")[1])
    else:
        part = 0
        total = 1
    start = len(file_list) * part // total
    end = len(file_list) * (part + 1) // total

    chunk_parser = ChunkParserInner(file_list[start:end], 1, 524288, 32, 1, 0, 6.0, 3.5)

    for i, file in enumerate(file_list[start:end]):
        data = chunk_parser.task(file)
        former_moves_left = b'\x00\x00\x00'
        pickle_data = []
        for j, d in enumerate(data):
            planes, probs, winner, best_q, moves_left = chunk_parser.convert_v6_to_tuple(d)
            proc_data = {
                'planes': planes,
                'probs': probs,
                'winner': winner,
                'best_q': best_q,
                'moves_left': moves_left,
                'former_moves_left': former_moves_left,
            }
            former_moves_left = moves_left
            pickle_data.append(proc_data)
        with open(os.path.join(args.out_dir, file.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.pickle'), 'wb') as f:
            pickle.dump(pickle_data, f)

if __name__ == '__main__':
    main()
