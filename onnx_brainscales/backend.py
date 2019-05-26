from onnx import ModelProto
from onnx.backend.base import Backend, DeviceType
import onnx

class BackendRep(object):
    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        pass

class BrainScaleSBackend(Backend):
    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):
        super(BrainScaleSBackend, cls).prepare(model, device, **kwargs)
        
        return None
        # return BrainScaleSRep(brainscales_model, onnx_outputs_info, device == 'BrainScaleS')

    @classmethod
    def is_compatible(cls,
                       model,  # type: ModelProto
                       device='CPU',  # type: Text
                       **kwargs  # type: Any
                       ):  # type: (...) -> bool
        node_set = set()
        initializer_set = set()
        graph = model.graph

        for node in graph.node:
            if node not in ['LIFLayer', 'ADEXLayer', 'LSNNLayer']:
                return False

        return True

    @classmethod
    def supports_device(cls,
                        device,  # type: Text
                        ):
        # type: (...) -> bool
        return device == 'BrainScaleS'

prepare = BrainScaleSBackend.prepare
run_node = BrainScaleSBackend.run_node
run_model = BrainScaleSBackend.run_model