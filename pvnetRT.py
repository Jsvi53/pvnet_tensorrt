from utils import TensorRTInfer, DataProcess, PostProcess
from torch.utils.data import DataLoader
import numpy as np


def main():
    results = {}
    results['K'] = np.loadtxt('/home/lmx/project/pvnet_tensorrt/dataset/custom/camera.txt')
    results['kpts'] = np.loadtxt('/home/lmx/project/pvnet_tensorrt/dataset/custom/fps.txt')
    results['model_path'] = '/home/lmx/project/pvnet_tensorrt/dataset/custom/model.ply'
    engine_path = '/home/lmx/project/pvnet_tensorrt/model//cpn_sim.engine'
    data_path = '/home/lmx/project/pvnet_tensorrt/dataset/custom/rgb'
    postprocessor = PostProcess(results)
    dataset = DataProcess(data_path)
    pvnet = TensorRTInfer(engine_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for data in data_loader:
        # data = next(iter(data_loader))
        results['img_path'] = data['img_path']
        results['output'] = pvnet.infer(data['img'].float().numpy())
        results['time'] = pvnet.time
        postprocessor.processResults(results)
        postprocessor.draw(single=False)


if __name__ == "__main__":
    main()
