import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time
# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit


class TensorRTInfer:
    """
    Implements inference for the PVNet TensorRT engine.
    """

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)  # Setup logger, parameters are optional
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize     # use numpy dtype for allocation size
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def infer(self, input):
        """
        Execute inference on a batch of images. The images should
        already be batched and preprocessed, as prepared by the
        ImageBatcher class. Memory copying to and from the GPU device
        will be performed here.

        :param batch: A numpy array holding the image batch.
        :return:
            - raw mask: (1, 2, H, W), e.g., (1, 2, 480, 640)
            - vector field: (1, 18, H, W), e.g., (1, 18, 480, 640)
            - mask: (1, H, W), e.g., (1, 480, 640)
            - kpts: (1, 9, 2)
        """
        output = [None] * len(self.outputs)
        # Prepare the output data
        for i in range(len(self.outputs)):
            output[i] = np.empty(self.outputs[i]['shape'], dtype=self.outputs[i]['dtype'])

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(input))
        # inference
        start_time = time.time()
        self.context.execute_v2(self.allocations)
        end_time = time.time()
        self.time = (end_time - start_time) * 1000
        # Copy the output data back to the host
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh(output[i], self.outputs[i]['allocation'])
        self.stream.synchronize()

        # Process the reslults
        return output
