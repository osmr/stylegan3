import argparse
import copy
import pickle

# import numpy as np


def blend_models(low, high, level):
    assert (low.state_dict().keys() == high.state_dict().keys())

    model_out = copy.deepcopy(low)
    params_src = high.named_parameters()
    dict_dest = model_out.state_dict()

    for name, param in params_src:
        name_parts_list = name.split(".")
        if (len(name_parts_list) < 1) or (name_parts_list[0] != "synthesis") or\
                (not name_parts_list[1].startswith("L")):
            continue
        real_level = int(name_parts_list[1].split("_")[0][1:])
        # if real_level < level:  # a
        if real_level > level:  # b
            continue
        dict_dest[name].data.copy_(param.data)

    model_out_dict = model_out.state_dict()
    model_out_dict.update(dict_dest)
    model_out.load_state_dict(dict_dest)

    return model_out


def run_blending(model1_path, model2_path, out_path, level=8, device='cuda'):
    with open(model1_path, 'rb') as f:
        model1 = pickle.load(f)['G_ema'].requires_grad_(False).to(device)
    with open(model2_path, 'rb') as f:
        model2 = pickle.load(f)['G_ema'].requires_grad_(False).to(device)

    model_final = blend_models(model1, model2, level)
    model_dict = dict()
    model_dict['G_ema'] = model_final

    with open(out_path, 'wb') as f:
        pickle.dump(model_dict, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-1', type=str, required=True)
    parser.add_argument('--model-2', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--level', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    run_blending(
        model1_path=args.model_1,
        model2_path=args.model_2,
        out_path=args.out_path,
        level=args.level)


if __name__ == '__main__':
    main()
