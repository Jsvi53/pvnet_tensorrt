from lib.config import cfg
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
import sys

def main():
    dummy_input = torch.ones((1, 3, 480, 640), device="cuda")
    torch.manual_seed(0)
    network = make_network(cfg).cuda()
    print("begin to load the {}th model".format(cfg.test.epoch))
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()
    input_names = ["input0"]
    output_names = [ "output0" ]
    torch.onnx.export(network, dummy_input, "clean_pvnet_green.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11)

if __name__ == '__main__':
    main()